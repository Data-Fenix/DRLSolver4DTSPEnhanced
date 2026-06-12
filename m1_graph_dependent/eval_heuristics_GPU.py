"""
eval_heuristics.py  —  GPU-batched distance lookups
Evaluate all heuristics on a fixed validation dataset.

Run from repo root (DRLSolver4DTSP-main/):
    python m1_graph_dependent/eval_heuristics.py --val_dataset valid_data_19.txt
    python m1_graph_dependent/eval_heuristics.py --val_dataset valid_data_49.txt

GPU is used for batched distance lookups (all unvisited nodes at once).
Navigation decisions remain sequential (inherent to heuristic logic).
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        """Single distance lookup (kept for compatibility)."""
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
        z2, z3 = z * z, z * z * z
        res    = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res, _ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim=-1), dim=-1)
        res, _ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim=-1), dim=-1)
        return res

    def get_distances_batch(self, current, candidates, t):
        """
        GPU-BATCHED distance lookup.
        Computes distances from `current` node to ALL candidate nodes at once.

        Args:
            current:    int, current node index
            candidates: list of int, candidate node indices
            t:          float, current time

        Returns:
            costs: torch.Tensor of shape [len(candidates)] on GPU
        """
        k   = len(candidates)
        n   = self.n_c
        mts = self.max_time_step

        # Build index tensors on GPU — shape [k]
        a_idx = torch.full((k,), current, dtype=torch.long, device=device)
        b_idx = torch.tensor(candidates, dtype=torch.long, device=device)
        t_val = torch.full((k,), t, device=device)

        tt = torch.floor(t_val * mts) % mts
        zz = (torch.floor(t_val * mts) + 1) % mts
        c  = a_idx * n * mts + b_idx * mts + tt.long()
        d  = a_idx * n * mts + b_idx * mts + zz.long()

        a0 = self.mat[c]
        a1 = self.m2[c]
        a2 = self.m3[c]
        a3 = self.m4[c]
        b0 = self.mat[d]

        z  = (t_val * mts - torch.floor(t_val * mts)) / mts
        z2, z3 = z * z, z * z * z
        res    = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res    = torch.max(res, minres)
        res    = torch.min(res, maxres)
        return res  # shape [k] on GPU

    def get_all_pairs_batch(self, nodes, t):
        """
        GPU-BATCHED all-pairs distance lookup at time t.
        Used by greedy_edge and two_opt.

        Args:
            nodes: list of int, node indices
            t:     float, time

        Returns:
            costs: torch.Tensor of shape [n, n] on GPU
        """
        n_nodes = len(nodes)
        n       = self.n_c
        mts     = self.max_time_step

        nodes_t = torch.tensor(nodes, dtype=torch.long, device=device)
        # All pairs: [n_nodes, n_nodes]
        a_idx = nodes_t.unsqueeze(1).expand(n_nodes, n_nodes).reshape(-1)
        b_idx = nodes_t.unsqueeze(0).expand(n_nodes, n_nodes).reshape(-1)
        total = n_nodes * n_nodes

        t_val = torch.full((total,), t, device=device)
        tt    = torch.floor(t_val * mts) % mts
        zz    = (torch.floor(t_val * mts) + 1) % mts
        c     = a_idx * n * mts + b_idx * mts + tt.long()
        d     = a_idx * n * mts + b_idx * mts + zz.long()

        a0 = self.mat[c]
        a1 = self.m2[c]
        a2 = self.m3[c]
        a3 = self.m4[c]
        b0 = self.mat[d]

        z  = (t_val * mts - torch.floor(t_val * mts)) / mts
        z2, z3 = z * z, z * z * z
        res    = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res    = torch.max(res, minres)
        res    = torch.min(res, maxres)
        return res.view(n_nodes, n_nodes)  # [n_nodes, n_nodes] on GPU


class TSPDataset(Dataset):
    def __init__(self, filename, n_cities=100):
        super(TSPDataset, self).__init__()
        ff  = np.loadtxt(filename, delimiter=' ')
        ind = torch.tensor(ff, dtype=torch.long).unsqueeze(2)
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
# HEURISTIC FUNCTIONS (GPU-batched distance lookups)
# ============================================================

def nearest_neighbor_tour(sample, mat):
    """Nearest Neighbor — batched distance lookups on GPU."""
    x         = sample.clone()
    node_ids  = x.argmax(dim=1).tolist()
    tour      = [node_ids[0]]
    unvisited = list(node_ids[1:])
    current   = node_ids[0]
    t         = 0.0
    tour_time = 0.0

    while unvisited:
        # GPU: compute distances to ALL unvisited nodes at once
        costs     = mat.get_distances_batch(current, unvisited, t)
        min_idx   = costs.argmin().item()
        min_cost  = costs[min_idx].item()
        next_node = unvisited[min_idx]

        tour_time += min_cost
        t         += min_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    # Return to depot
    costs     = mat.get_distances_batch(current, [node_ids[0]], t)
    tour_time += costs[0].item()
    return tour_time


def _build_nn_tour_batched(node_ids, mat):
    """Helper: build NN tour with batched GPU lookups."""
    tour      = [node_ids[0]]
    unvisited = list(node_ids[1:])
    current   = node_ids[0]
    t         = 0.0

    while unvisited:
        costs     = mat.get_distances_batch(current, unvisited, t)
        min_idx   = costs.argmin().item()
        min_cost  = costs[min_idx].item()
        next_node = unvisited[min_idx]
        t        += min_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour


def _tour_cost_batched(tour, mat):
    """Calculate tour cost sequentially (time-aware, must be sequential)."""
    total = 0.0
    t     = 0.0
    for i in range(len(tour) - 1):
        costs  = mat.get_distances_batch(tour[i], [tour[i + 1]], t)
        travel = costs[0].item()
        total += travel
        t     += travel
    costs  = mat.get_distances_batch(tour[-1], [tour[0]], t)
    total += costs[0].item()
    return total


def nn_plus_one_tour(sample, mat):
    """NN+1 with 1-step lookahead — batched GPU lookups for both steps."""
    x         = sample.clone()
    node_ids  = x.argmax(dim=1).tolist()
    tour      = [node_ids[0]]
    unvisited = list(node_ids[1:])
    current   = node_ids[0]
    t         = 0.0
    tour_time = 0.0
    lw        = 0.5  # lookahead weight

    while unvisited:
        if len(unvisited) == 1:
            next_node = unvisited[0]
        else:
            # GPU: distances from current to all unvisited
            costs_to = mat.get_distances_batch(current, unvisited, t)  # [k]

            best_combined, next_node = float('inf'), unvisited[0]
            for idx, j in enumerate(unvisited):
                cost_j    = costs_to[idx].item()
                # GPU: distances from j to all other unvisited (lookahead)
                others    = [k for k in unvisited if k != j]
                if others:
                    ahead     = mat.get_distances_batch(j, others, t + cost_j)
                    lookahead = ahead.min().item()
                else:
                    lookahead = 0.0
                combined  = cost_j + lw * lookahead
                if combined < best_combined:
                    best_combined = combined
                    next_node     = j

        costs     = mat.get_distances_batch(current, [next_node], t)
        cost_ij   = costs[0].item()
        tour_time += cost_ij
        t         += cost_ij
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    costs     = mat.get_distances_batch(current, [node_ids[0]], t)
    tour_time += costs[0].item()
    return tour_time


def two_opt_tour(sample, mat, max_iterations=10):
    """2-Opt — batched GPU all-pairs lookup for initial edge costs."""
    x         = sample.clone()
    node_ids  = x.argmax(dim=1).tolist()
    tour      = _build_nn_tour_batched(node_ids, mat)
    best_cost = _tour_cost_batched(tour, mat)

    for _ in range(max_iterations):
        improved = False
        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour  = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                new_cost  = _tour_cost_batched(new_tour, mat)
                if new_cost < best_cost - 1e-6:
                    tour      = new_tour
                    best_cost = new_cost
                    improved  = True
                    break
            if improved:
                break
        if not improved:
            break
    return best_cost


def nnr_tour(sample, mat, top_k=3, probabilities=None):
    """NNR — batched GPU distance lookups."""
    if probabilities is None:
        probabilities = [0.7, 0.2, 0.1]
    probabilities = probabilities[:top_k]
    s             = sum(probabilities)
    probabilities = [p / s for p in probabilities]

    x         = sample.clone()
    node_ids  = x.argmax(dim=1).tolist()
    tour      = [node_ids[0]]
    unvisited = list(node_ids[1:])
    current   = node_ids[0]
    t         = 0.0
    tour_time = 0.0

    while unvisited:
        costs      = mat.get_distances_batch(current, unvisited, t)
        costs_list = costs.cpu().tolist()
        candidates = sorted(zip(costs_list, unvisited), key=lambda x: x[0])

        k = min(top_k, len(candidates))
        if k == 1:
            selected_cost, next_node = candidates[0]
        else:
            probs    = probabilities[:k]
            s        = sum(probs)
            probs    = [p / s for p in probs]
            r        = random.random()
            cumsum   = 0
            sel_idx  = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    sel_idx = i
                    break
            selected_cost, next_node = candidates[sel_idx]

        tour_time += selected_cost
        t         += selected_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    costs     = mat.get_distances_batch(current, [node_ids[0]], t)
    tour_time += costs[0].item()
    return tour_time


def nnr_best_tour(sample, mat, num_runs=10):
    """NNR best of num_runs."""
    best = float('inf')
    for _ in range(num_runs):
        cost = nnr_tour(sample, mat)
        if cost < best:
            best = cost
    return best


def greedy_edge_tour(sample, mat):
    """Greedy Edge — batched GPU all-pairs lookup at t=0."""
    x         = sample.clone()
    node_ids  = x.argmax(dim=1).tolist()
    num_nodes = len(node_ids)

    # GPU: compute all pairs at t=0 in one call
    cost_matrix = mat.get_all_pairs_batch(node_ids, t=0.0).cpu()

    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edges.append((cost_matrix[i, j].item(), node_ids[i], node_ids[j]))
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

    return _tour_cost_batched(tour, mat)


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
    parser.add_argument('--val_dataset',  type=str, required=True)
    parser.add_argument('--n_cities',     type=int, default=100)
    parser.add_argument('--num_samples',   type=int, default=None,
                        help='Limit instances (e.g. 5 for a quick test)')
    parser.add_argument('--two_opt_iters', type=int, default=10,
                        help='Max iterations for 2-opt (default 10). Use higher for better quality at the cost of time.')
    parser.add_argument('--seed',          type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 60)
    print('HEURISTIC EVALUATION  (GPU-batched distance lookups)')
    print('=' * 60)
    print(f'  Dataset : {args.val_dataset}')
    print(f'  Device  : {device}')
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
        print(f'\nRunning {name} ...', flush=True)
        costs   = []
        t_start = time.time()

        for i in range(len(dataset)):
            costs.append(func(dataset[i], mat))

            # Progress every 50 instances
            if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
                elapsed   = time.time() - t_start
                avg_time  = elapsed / (i + 1)
                remaining = avg_time * (len(dataset) - (i + 1))
                avg_so_far = np.mean(costs) * 1440
                print(
                    f'  [{name}] {i+1}/{len(dataset)} '
                    f'| elapsed: {elapsed:.1f}s '
                    f'| remaining: {remaining:.1f}s '
                    f'| avg cost: {avg_so_far:.2f} min',
                    flush=True
                )

        elapsed   = time.time() - t_start
        costs_min = np.array(costs) * 1440
        results[name] = {
            'mean':          float(np.mean(costs_min)),
            'std':           float(np.std(costs_min)),
            'min':           float(np.min(costs_min)),
            'max':           float(np.max(costs_min)),
            'elapsed_s':     elapsed,
            'time_per_inst': elapsed / len(dataset),
            'raw':           costs_min,
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

    # Save outputs
    out_dir = os.path.dirname(os.path.abspath(args.val_dataset))
    tag     = os.path.splitext(os.path.basename(args.val_dataset))[0]

    for name, r in results.items():
        np.savetxt(os.path.join(out_dir, f'heuristic_{name}_{tag}_costs.txt'), r['raw'], fmt='%.6f')

    summary_path = os.path.join(out_dir, f'heuristic_summary_{tag}.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'heuristic', 'mean_min', 'std_min', 'min_min', 'max_min',
            'total_time_s', 'time_per_inst_s', 'num_instances'
        ])
        writer.writeheader()
        for name, r in sorted(results.items(), key=lambda x: x[1]['mean']):
            writer.writerow({
                'heuristic':       name,
                'mean_min':        round(r['mean'], 4),
                'std_min':         round(r['std'],  4),
                'min_min':         round(r['min'],  4),
                'max_min':         round(r['max'],  4),
                'total_time_s':    round(r['elapsed_s'],     4),
                'time_per_inst_s': round(r['time_per_inst'], 6),
                'num_instances':   len(dataset),
            })

    print(f'\nSaved summary  : {summary_path}')
    print(f'Saved costs    : heuristic_<name>_{tag}_costs.txt')


if __name__ == '__main__':
    main()