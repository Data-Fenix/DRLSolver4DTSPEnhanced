"""
eval_heuristics.py
Evaluate all heuristics on a fixed validation dataset (valid_data.txt).

Run from repo root (DRLSolver4DTSP-main/):
    python m1_graph_dependent/eval_heuristics.py --val_dataset valid_data_19.txt
    python m1_graph_dependent/eval_heuristics.py --val_dataset valid_data_49.txt

Note: Heuristics are inherently sequential (one city at a time), so they run on CPU
even on a GPU cluster. The distance matrix is loaded on CPU accordingly.
"""

import torch
import numpy as np
import random
import argparse
import time
import os
import csv
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# Heuristics are sequential - CPU is correct regardless of cluster GPU availability
device = torch.device('cpu')


# ============================================================
# DATA STRUCTURES
# ============================================================

class Cities:
    def __init__(self, n_cities=100):
        self.n_cities = n_cities
        self.cities = torch.rand((n_cities, 2))


class DistanceMatrix:
    def __init__(self, ci, max_time_step=100, load_dir=None):
        self.n_c = ci.n_cities
        self.max_time_step = max_time_step
        with torch.no_grad():
            self.mat = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m2  = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m3  = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m4  = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            if load_dir is not None:
                temp = np.loadtxt(load_dir, delimiter=',', skiprows=0)
                x = np.arange(max_time_step + 1)
                for k in range(self.n_c):
                    for j in range(self.n_c):
                        i = k * self.n_c + j
                        cs = CubicSpline(
                            x,
                            np.concatenate((temp[i], [temp[i, 0]]), axis=0),
                            bc_type='periodic'
                        )
                        self.m4[i * max_time_step: i * max_time_step + 12] = torch.tensor(cs.c[0], device=device)
                        self.m3[i * max_time_step: i * max_time_step + 12] = torch.tensor(cs.c[1], device=device)
                        self.m2[i * max_time_step: i * max_time_step + 12] = torch.tensor(cs.c[2], device=device)
                        self.mat[i * max_time_step: i * max_time_step + 12] = torch.tensor(cs.c[3], device=device)

    def __getd__(self, st, a, b, t):
        a  = torch.gather(st, 1, a)
        b  = torch.gather(st, 1, b)
        tt = torch.floor(t * self.max_time_step) % self.max_time_step
        zz = (torch.floor(t * self.max_time_step) + 1) % self.max_time_step
        c  = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + tt.squeeze().long()
        d  = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + zz.squeeze().long()
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2,  0, c)
        a2 = torch.gather(self.m3,  0, c)
        a3 = torch.gather(self.m4,  0, c)
        b0 = torch.gather(self.mat, 0, d)
        z  = (t.squeeze() * self.max_time_step - torch.floor(t.squeeze() * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        res    = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res, _ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim=-1), dim=-1)
        res, _ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim=-1), dim=-1)
        return res


class TSPDataset(Dataset):
    """Loads a fixed validation dataset from file (e.g. valid_data_19.txt)."""
    def __init__(self, filename, n_cities=100):
        super(TSPDataset, self).__init__()
        ff  = np.loadtxt(filename, delimiter=' ')
        ind = torch.tensor(ff, dtype=torch.long).unsqueeze(2)   # [N, seq_len, 1]
        num_samples = ind.size(0)
        seq_len     = ind.size(1)
        self.data   = torch.zeros(num_samples, seq_len, n_cities)
        self.data   = self.data.scatter_(2, ind, 1.)
        self.size   = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# HEURISTIC FUNCTIONS
# ============================================================

def nearest_neighbor_tour(sample, mat):
    x        = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    tour     = [node_ids[0]]
    unvisited = set(node_ids[1:])   # only the cities in this instance
    current  = node_ids[0]
    t        = 0.0
    tour_time = 0
    st       = torch.arange(n).unsqueeze(0)

    while unvisited:
        min_cost, next_node = float('inf'), None
        for j in unvisited:
            a  = torch.tensor([[current]], dtype=torch.long)
            b  = torch.tensor([[j]],       dtype=torch.long)
            tt = torch.tensor([[t]])
            cost_val = mat.__getd__(st, a, b, tt).item()
            if cost_val < min_cost:
                min_cost, next_node = cost_val, j
        tour_time += min_cost
        t         += min_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    a  = torch.tensor([[current]],      dtype=torch.long)
    b  = torch.tensor([[node_ids[0]]], dtype=torch.long)
    tt = torch.tensor([[t]])
    tour_time += mat.__getd__(st, a, b, tt).item()
    return tour_time


def _build_nn_tour(node_ids, mat, st, n):
    tour      = [node_ids[0]]
    unvisited = set(node_ids[1:])
    current   = node_ids[0]
    t         = 0.0
    while unvisited:
        min_cost, next_node = float('inf'), None
        for j in unvisited:
            a  = torch.tensor([[current]], dtype=torch.long)
            b  = torch.tensor([[j]],       dtype=torch.long)
            tt = torch.tensor([[t]])
            cost_ij = mat.__getd__(st, a, b, tt).item()
            if cost_ij < min_cost:
                min_cost, next_node = cost_ij, j
        tour.append(next_node)
        t += min_cost
        unvisited.remove(next_node)
        current = next_node
    return tour


def _tour_cost(tour, mat, st):
    total = 0.0
    t     = 0.0
    for i in range(len(tour) - 1):
        a  = torch.tensor([[tour[i]]],     dtype=torch.long)
        b  = torch.tensor([[tour[i + 1]]], dtype=torch.long)
        tt = torch.tensor([[t]])
        travel = mat.__getd__(st, a, b, tt).item()
        total += travel
        t     += travel
    a  = torch.tensor([[tour[-1]]],  dtype=torch.long)
    b  = torch.tensor([[tour[0]]], dtype=torch.long)
    tt = torch.tensor([[t]])
    total += mat.__getd__(st, a, b, tt).item()
    return total


def nn_plus_one_tour(sample, mat):
    x        = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    tour     = [node_ids[0]]
    unvisited = set(node_ids[1:])   # only the cities in this instance
    current  = node_ids[0]
    t        = 0.0
    tour_time = 0
    st       = torch.arange(n).unsqueeze(0)
    lookahead_weight = 0.5

    while unvisited:
        unvisited_list = list(unvisited)
        if len(unvisited_list) == 1:
            next_node = unvisited_list[0]
        else:
            costs_to = []
            for j in unvisited_list:
                a  = torch.tensor([[current]], dtype=torch.long)
                b  = torch.tensor([[j]],       dtype=torch.long)
                tt = torch.tensor([[t]])
                costs_to.append(mat.__getd__(st, a, b, tt).item())

            combined = []
            for idx, j in enumerate(unvisited_list):
                cost_j = costs_to[idx]
                ahead  = []
                for k in unvisited_list:
                    if k != j:
                        a  = torch.tensor([[j]], dtype=torch.long)
                        b  = torch.tensor([[k]], dtype=torch.long)
                        tt = torch.tensor([[t + cost_j]])
                        ahead.append(mat.__getd__(st, a, b, tt).item())
                combined.append((cost_j + lookahead_weight * (min(ahead) if ahead else 0), j))
            next_node = min(combined, key=lambda x: x[0])[1]

        a  = torch.tensor([[current]],   dtype=torch.long)
        b  = torch.tensor([[next_node]], dtype=torch.long)
        tt = torch.tensor([[t]])
        cost_ij = mat.__getd__(st, a, b, tt).item()
        tour_time += cost_ij
        t         += cost_ij
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    a  = torch.tensor([[current]],      dtype=torch.long)
    b  = torch.tensor([[node_ids[0]]], dtype=torch.long)
    tt = torch.tensor([[t]])
    tour_time += mat.__getd__(st, a, b, tt).item()
    return tour_time


def two_opt_tour(sample, mat, max_iterations=10):
    x        = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    st       = torch.arange(n).unsqueeze(0)
    tour     = _build_nn_tour(node_ids, mat, st, n)
    best_cost = _tour_cost(tour, mat, st)

    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour  = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                new_cost  = _tour_cost(new_tour, mat, st)
                if new_cost < best_cost - 1e-6:
                    tour      = new_tour
                    best_cost = new_cost
                    improved  = True
                    break       # restart from beginning on first improvement
            if improved:
                break
        if not improved:
            break
    return best_cost


def nnr_tour(sample, mat, top_k=3, probabilities=None):
    if probabilities is None:
        probabilities = [0.7, 0.2, 0.1]
    probabilities = probabilities[:top_k]
    prob_sum      = sum(probabilities)
    probabilities = [p / prob_sum for p in probabilities]

    x        = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    tour     = [node_ids[0]]
    unvisited = set(node_ids[1:])
    current  = node_ids[0]
    t        = 0.0
    tour_time = 0.0
    st       = torch.arange(n).unsqueeze(0)

    while unvisited:
        candidates = []
        for j in unvisited:
            a  = torch.tensor([[current]], dtype=torch.long)
            b  = torch.tensor([[j]],       dtype=torch.long)
            tt = torch.tensor([[t]])
            candidates.append((mat.__getd__(st, a, b, tt).item(), j))
        candidates.sort(key=lambda x: x[0])

        k = min(top_k, len(candidates))
        if k == 1:
            selected_cost, next_node = candidates[0]
        else:
            probs    = probabilities[:k]
            prob_sum = sum(probs)
            probs    = [p / prob_sum for p in probs]
            r        = random.random()
            cumsum   = 0
            selected_idx = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    selected_idx = i
                    break
            selected_cost, next_node = candidates[selected_idx]

        tour_time += selected_cost
        t         += selected_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    a  = torch.tensor([[current]],      dtype=torch.long)
    b  = torch.tensor([[node_ids[0]]], dtype=torch.long)
    tt = torch.tensor([[t]])
    tour_time += mat.__getd__(st, a, b, tt).item()
    return tour_time


def nnr_best_tour(sample, mat, num_runs=10):
    best = float('inf')
    for _ in range(num_runs):
        cost = nnr_tour(sample, mat)
        if cost < best:
            best = cost
    return best


def greedy_edge_tour(sample, mat):
    x        = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    num_nodes = len(node_ids)
    st       = torch.arange(n).unsqueeze(0)
    t_init   = torch.tensor([[0.0]])

    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            a    = torch.tensor([[node_ids[i]]], dtype=torch.long)
            b    = torch.tensor([[node_ids[j]]], dtype=torch.long)
            cost = mat.__getd__(st, a, b, t_init).item()
            edges.append((cost, node_ids[i], node_ids[j]))
    edges.sort(key=lambda x: x[0])

    degree    = {node: 0 for node in node_ids}
    adjacency = {node: [] for node in node_ids}
    parent    = {node: node for node in node_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False

    selected_edges = []
    for cost, u, v in edges:
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        if len(selected_edges) < num_nodes - 1:
            if find(u) == find(v):
                continue
        selected_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        adjacency[u].append(v)
        adjacency[v].append(u)
        union(u, v)
        if len(selected_edges) == num_nodes:
            break

    start = node_ids[0]
    for node in node_ids:
        if degree[node] == 1:
            start = node
            break

    tour          = [start]
    visited_nodes = {start}
    current       = start
    while len(tour) < num_nodes:
        found = False
        for neighbor in adjacency[current]:
            if neighbor not in visited_nodes:
                tour.append(neighbor)
                visited_nodes.add(neighbor)
                current = neighbor
                found   = True
                break
        if not found:
            for node in node_ids:
                if node not in visited_nodes:
                    tour.append(node)
                    visited_nodes.add(node)
                    current = node
                    break

    return _tour_cost(tour, mat, st)


# ============================================================
# MAIN
# ============================================================

HEURISTICS = {
    'nearest_neighbor': nearest_neighbor_tour,
    'greedy_edge':      greedy_edge_tour,
    'nn_plus_one':      nn_plus_one_tour,
    'two_opt':          two_opt_tour,
    'nnr_best':         nnr_best_tour,
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate heuristics on a fixed validation dataset')
    parser.add_argument('--val_dataset', type=str, required=True,
                        help='Path to validation dataset file, e.g. valid_data_19.txt')
    parser.add_argument('--n_cities', type=int, default=100,
                        help='Total city pool size (default 100)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Limit number of instances to evaluate (default: all). Use 5 for a quick check.')
    parser.add_argument('--two_opt_iters', type=int, default=10,
                        help='Max iterations for 2-opt (default 10). Use higher for better quality at the cost of time.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default 1234)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 55)
    print('HEURISTIC EVALUATION')
    print('=' * 55)
    print(f'  Dataset : {args.val_dataset}')
    print(f'  Device  : CPU (heuristics are sequential)')
    print(f'  Seed    : {args.seed}')

    print('\nLoading distance matrix from m1/data.csv ...')
    ci  = Cities(n_cities=args.n_cities)
    mat = DistanceMatrix(ci, load_dir='m1/data.csv', max_time_step=12)
    print('  Done.')

    print(f'\nLoading validation dataset from {args.val_dataset} ...')
    dataset = TSPDataset(filename=args.val_dataset, n_cities=args.n_cities)
    if args.num_samples is not None:
        dataset.data = dataset.data[:args.num_samples]
        dataset.size = len(dataset.data)
    print(f'  Loaded {len(dataset)} instances.')

    HEURISTICS['two_opt'] = lambda s, m: two_opt_tour(s, m, max_iterations=args.two_opt_iters)

    results = {}
    for name, func in HEURISTICS.items():
        costs      = []
        t_start    = time.time()
        for i in tqdm(range(len(dataset)), desc=f'{name:<22}', ncols=80, unit='inst'):
            costs.append(func(dataset[i], mat))
        elapsed    = time.time() - t_start
        costs_min  = np.array(costs) * 1440
        results[name] = {
            'mean':         float(np.mean(costs_min)),
            'std':          float(np.std(costs_min)),
            'min':          float(np.min(costs_min)),
            'max':          float(np.max(costs_min)),
            'elapsed_s':    elapsed,
            'time_per_inst': elapsed / len(dataset),
            'raw':          costs_min,
        }

    print('\n' + '=' * 80)
    print(f'RESULTS  —  dataset: {args.val_dataset}  |  instances: {len(dataset)}')
    print('=' * 80)
    print(f"{'Heuristic':<25} {'Mean (min)':>12} {'Std':>10} {'Min':>10} {'Total (s)':>11} {'Per inst (s)':>13}")
    print('-' * 80)
    for name, r in sorted(results.items(), key=lambda x: x[1]['mean']):
        print(f"{name:<25} {r['mean']:>12.2f} {r['std']:>10.2f} {r['min']:>10.2f} "
              f"{r['elapsed_s']:>11.2f} {r['time_per_inst']:>13.4f}")
    print('=' * 80)

    # ── Output paths ──────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.val_dataset))
    tag     = os.path.splitext(os.path.basename(args.val_dataset))[0]

    # Save per-heuristic raw cost arrays
    for name, r in results.items():
        out_path = os.path.join(out_dir, f'heuristic_{name}_{tag}_costs.txt')
        np.savetxt(out_path, r['raw'], fmt='%.6f')

    # Save summary CSV (one row per heuristic)
    summary_path = os.path.join(out_dir, f'heuristic_summary_{tag}.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'heuristic', 'mean_min', 'std_min', 'min_min', 'max_min',
            'total_time_s', 'time_per_inst_s', 'num_instances'
        ])
        writer.writeheader()
        for name, r in sorted(results.items(), key=lambda x: x[1]['mean']):
            writer.writerow({
                'heuristic':        name,
                'mean_min':         round(r['mean'], 4),
                'std_min':          round(r['std'],  4),
                'min_min':          round(r['min'],  4),
                'max_min':          round(r['max'],  4),
                'total_time_s':     round(r['elapsed_s'],     4),
                'time_per_inst_s':  round(r['time_per_inst'], 6),
                'num_instances':    len(dataset),
            })

    print(f'\nSaved summary : {summary_path}')
    print(f'Saved cost arrays: heuristic_<name>_{tag}_costs.txt')


if __name__ == '__main__':
    main()
