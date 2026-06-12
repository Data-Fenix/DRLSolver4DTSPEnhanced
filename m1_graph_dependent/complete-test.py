import os
import time
import torch
import math
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
import numpy as np
from transformer import AttentionModel
from scipy.interpolate import CubicSpline
from scipy import stats
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import random

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
import warnings
import pprint as pp
warnings = warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Cities:
    def __init__(self, n_cities=100):
        self.n_cities = n_cities
        self.cities = torch.rand((n_cities, 2))
    def __getdis__(self, i, j):
        return torch.sqrt(torch.sum(torch.pow(torch.sub(self.cities[i], self.cities[j]), 2)))


class DistanceMatrix:
    def __init__(self, ci, max_time_step=100, load_dir=None):
        self.n_c = ci.n_cities
        self.max_time_step = max_time_step
        with torch.no_grad():
            self.mat = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m2 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m3 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m4 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.var = torch.full((ci.n_cities * ci.n_cities, 1), 0.03, device=device).view(-1)
            if load_dir is not None:
                temp = np.loadtxt(load_dir, delimiter=',', skiprows=0)
                x = np.arange(max_time_step + 1)
                for k in range(self.n_c):
                    self.var[k*self.n_c+k] = 0
                    for j in range(self.n_c):
                        i = k * self.n_c + j
                        cs = CubicSpline(x, np.concatenate((temp[i], [temp[i,0]]), axis=0), bc_type='periodic')
                        self.m4[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[0], device=device)
                        self.m3[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[1], device=device)
                        self.m2[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[2], device=device)
                        self.mat[i * max_time_step : i * max_time_step + 12] = torch.tensor(cs.c[3], device=device)

    def __getd__(self, st, a, b, t):
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        tt = torch.floor(t * self.max_time_step) % self.max_time_step
        zz = (torch.floor(t * self.max_time_step) + 1) % self.max_time_step
        c = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + tt.squeeze().long()
        d = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + zz.squeeze().long()
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2, 0, c)
        a2 = torch.gather(self.m3, 0, c)
        a3 = torch.gather(self.m4, 0, c)
        b0 = torch.gather(self.mat, 0, d)
        z = (t.squeeze() * self.max_time_step - torch.floor(t.squeeze() * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res, _ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim=-1), dim=-1)
        res, _ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim=-1), dim=-1)
        return res

    def __getddd__(self, st, a, b, t):
        s0, s1 = a.size(0), a.size(1)
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        tt = torch.round(t * self.max_time_step) % self.max_time_step
        zz = (torch.round(t * self.max_time_step) + 1) % self.max_time_step
        c = a * self.n_c * self.max_time_step + b * self.max_time_step + tt.long()
        c = c.view(-1)
        d = a * self.n_c * self.max_time_step + b * self.max_time_step + zz.long()
        d = d.view(-1)
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2, 0, c)
        a2 = torch.gather(self.m3, 0, c)
        a3 = torch.gather(self.m4, 0, c)
        b0 = torch.gather(self.mat, 0, d)
        tt = tt.view(-1)
        ttt = t.expand(s0, s1).contiguous().view(-1)
        z = (ttt * self.max_time_step - torch.floor(ttt * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res, _ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim=-1), dim=-1)
        res, _ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim=-1), dim=-1)
        return res.view(s0, s1)


def rollout(mat, model, dataset, opts):
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, _ = model(mat, move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)


def roll(mat, model, dataset, opts):
    set_decode_type(model, "greedy")
    model.eval()
    c = []
    p = []

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(mat, move_to(bat, opts.device), return_pi=True)
        return cost.data.cpu(), pi.data.cpu()

    for bat in DataLoader(dataset, batch_size=opts.eval_batch_size):
        cost, pi = eval_model_bat(bat)
        for z in range(cost.size(0)):
            c.append(cost[z])
            p.append(pi[z])
    return torch.stack(p), torch.stack(c)


def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)


def get_inner_model(model):
    return model


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))
    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)
        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)
        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)


class TSPDataset(Dataset):
    def __init__(self, ci, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()
        if filename is None:
            self.data_set = []
            l = torch.rand((num_samples, ci.n_cities - 1))
            sorted, ind = torch.sort(l)
            ind = ind.unsqueeze(2).expand(num_samples, ci.n_cities - 1, 2)
            ind = ind[:, :size, :] + 1
            ff = ci.cities.unsqueeze(0)
            ff = ff.expand(num_samples, ci.n_cities, 2)
            f = torch.gather(ff, dim=1, index=ind)
            f = f.permute(0, 2, 1)
            depot = ci.cities[0].view(1, 2, 1).expand(num_samples, 2, 1)
            self.static = torch.cat((depot, f), dim=2)
            depot = torch.zeros(num_samples, 1, 1, dtype=torch.long)
            ind = ind[:, :, 0:1]
            ind = torch.cat((depot, ind), dim=1)
        else:
            ff = np.loadtxt(filename, delimiter=' ')
            ind = torch.tensor(ff, dtype=torch.long).unsqueeze(2)
            file_num_samples = ind.size(0)
            if file_num_samples > num_samples:
                ind = ind[:num_samples]
                num_samples = ind.size(0)
            elif file_num_samples < num_samples:
                num_samples = file_num_samples

        self.data = torch.zeros(num_samples, size + 1, ci.n_cities)
        self.data = self.data.scatter_(2, ind, 1.)
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# HEURISTIC FUNCTIONS
# FIX: All tensors created with .to(device) to match mat on GPU
# ============================================================

def nearest_neighbor_tour(sample, mat):
    x = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    visited = [node_ids[0]]
    tour = [node_ids[0]]
    unvisited = set(range(n))
    unvisited.remove(node_ids[0])
    current = node_ids[0]
    t = 0.0
    tour_time = 0
    st = torch.arange(n).unsqueeze(0).to(device)  # FIX

    while unvisited:
        min_cost, next_node = float('inf'), None
        for j in unvisited:
            a = torch.tensor([[current]], dtype=torch.long).to(device)  # FIX
            b = torch.tensor([[j]], dtype=torch.long).to(device)        # FIX
            tt = torch.tensor([[t]]).to(device)                         # FIX
            cost_ij = mat.__getd__(st, a, b, tt)
            cost_val = cost_ij.item() if hasattr(cost_ij, 'item') else float(cost_ij)
            if cost_val < min_cost:
                min_cost, next_node = cost_val, j
        tour_time += min_cost
        t += min_cost
        visited.append(next_node)
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    a = torch.tensor([[current]], dtype=torch.long).to(device)      # FIX
    b = torch.tensor([[node_ids[0]]], dtype=torch.long).to(device)  # FIX
    tt = torch.tensor([[t]]).to(device)                              # FIX
    cost_return = mat.__getd__(st, a, b, tt)
    tour_time += cost_return.item() if hasattr(cost_return, 'item') else float(cost_return)
    return tour_time


def nn_plus_one_tour(sample, mat):
    x = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    visited = [node_ids[0]]
    tour = [node_ids[0]]
    unvisited = set(range(n))
    unvisited.remove(node_ids[0])
    current = node_ids[0]
    t = 0.0
    tour_time = 0
    st = torch.arange(n).unsqueeze(0).to(device)  # FIX
    lookahead_weight = 0.5

    while unvisited:
        unvisited_list = list(unvisited)
        num_unvisited = len(unvisited_list)

        if num_unvisited == 1:
            next_node = unvisited_list[0]
        else:
            current_tensor = torch.tensor([[current]], dtype=torch.long).to(device)  # FIX
            costs_to_candidates = []
            for j in unvisited_list:
                j_tensor = torch.tensor([[j]], dtype=torch.long).to(device)  # FIX
                tt_tensor = torch.tensor([[t]]).to(device)                    # FIX
                cost = mat.__getd__(st, current_tensor, j_tensor, tt_tensor).item()
                costs_to_candidates.append(cost)

            combined_costs = []
            for idx, j in enumerate(unvisited_list):
                cost_to_j = costs_to_candidates[idx]
                lookahead_costs = []
                for k in unvisited_list:
                    if k != j:
                        j_tensor = torch.tensor([[j]], dtype=torch.long).to(device)      # FIX
                        k_tensor = torch.tensor([[k]], dtype=torch.long).to(device)      # FIX
                        tt2_tensor = torch.tensor([[t + cost_to_j]]).to(device)          # FIX
                        cost_jk = mat.__getd__(st, j_tensor, k_tensor, tt2_tensor).item()
                        lookahead_costs.append(cost_jk)
                lookahead_cost = min(lookahead_costs) if lookahead_costs else 0
                combined_cost = cost_to_j + lookahead_weight * lookahead_cost
                combined_costs.append((combined_cost, j))
            next_node = min(combined_costs, key=lambda x: x[0])[1]

        a = torch.tensor([[current]], dtype=torch.long).to(device)    # FIX
        b = torch.tensor([[next_node]], dtype=torch.long).to(device)  # FIX
        tt = torch.tensor([[t]]).to(device)                            # FIX
        cost_ij = mat.__getd__(st, a, b, tt).item()

        tour_time += cost_ij
        t += cost_ij
        visited.append(next_node)
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    a = torch.tensor([[current]], dtype=torch.long).to(device)      # FIX
    b = torch.tensor([[node_ids[0]]], dtype=torch.long).to(device)  # FIX
    tt = torch.tensor([[t]]).to(device)                              # FIX
    cost_return = mat.__getd__(st, a, b, tt).item()
    tour_time += cost_return
    return tour_time


def two_opt_tour(sample, mat, max_iterations=100):
    x = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    st = torch.arange(n).unsqueeze(0).to(device)  # FIX

    tour = _build_nn_tour(node_ids, mat, st, n)
    best_cost = _calculate_tour_cost_time_aware(tour, mat, st)
    best_tour = tour.copy()

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(1, len(tour) - 1):
            for j in range(i + 1, len(tour)):
                new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                new_cost = _calculate_tour_cost_time_aware(new_tour, mat, st)

                if new_cost < best_cost - 1e-6:
                    best_cost = new_cost
                    best_tour = new_tour.copy()
                    tour = new_tour.copy()
                    improved = True
                    break
            if improved:
                break

    return best_cost


def _build_nn_tour(node_ids, mat, st, n):
    tour = [node_ids[0]]
    unvisited = set(node_ids[1:])
    current = node_ids[0]
    t = 0.0

    while unvisited:
        min_cost, next_node = float('inf'), None
        for j in unvisited:
            a = torch.tensor([[current]], dtype=torch.long).to(device)  # FIX
            b = torch.tensor([[j]], dtype=torch.long).to(device)        # FIX
            tt = torch.tensor([[t]]).to(device)                         # FIX
            cost_ij = mat.__getd__(st, a, b, tt).item()
            if cost_ij < min_cost:
                min_cost, next_node = cost_ij, j
        t += min_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    return tour


def _calculate_tour_cost_time_aware(tour, mat, st):
    total_time = 0.0
    t = 0.0

    for i in range(len(tour) - 1):
        a = torch.tensor([[tour[i]]], dtype=torch.long).to(device)      # FIX
        b = torch.tensor([[tour[i + 1]]], dtype=torch.long).to(device)  # FIX
        tt = torch.tensor([[t]]).to(device)                              # FIX
        travel_time = mat.__getd__(st, a, b, tt).item()
        total_time += travel_time
        t += travel_time

    a = torch.tensor([[tour[-1]]], dtype=torch.long).to(device)  # FIX
    b = torch.tensor([[tour[0]]], dtype=torch.long).to(device)   # FIX
    tt = torch.tensor([[t]]).to(device)                           # FIX
    travel_time = mat.__getd__(st, a, b, tt).item()
    total_time += travel_time

    return total_time


def nnr_tour(sample, mat, top_k=3, probabilities=None):
    if probabilities is None:
        probabilities = [0.7, 0.2, 0.1]

    probabilities = probabilities[:top_k]
    prob_sum = sum(probabilities)
    probabilities = [p / prob_sum for p in probabilities]

    x = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()

    tour = [node_ids[0]]
    unvisited = set(node_ids[1:])
    current = node_ids[0]
    t = 0.0
    tour_time = 0.0
    st = torch.arange(n).unsqueeze(0).to(device)  # FIX

    while unvisited:
        candidates = []
        for j in unvisited:
            a = torch.tensor([[current]], dtype=torch.long).to(device)  # FIX
            b = torch.tensor([[j]], dtype=torch.long).to(device)        # FIX
            tt = torch.tensor([[t]]).to(device)                         # FIX
            cost_ij = mat.__getd__(st, a, b, tt).item()
            candidates.append((cost_ij, j))

        candidates.sort(key=lambda x: x[0])

        k = min(top_k, len(candidates))
        if k == 1:
            selected_cost, next_node = candidates[0]
        else:
            probs = probabilities[:k]
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]

            r = random.random()
            cumsum = 0
            selected_idx = 0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    selected_idx = i
                    break

            selected_cost, next_node = candidates[selected_idx]

        tour_time += selected_cost
        t += selected_cost
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node

    a = torch.tensor([[current]], dtype=torch.long).to(device)      # FIX
    b = torch.tensor([[node_ids[0]]], dtype=torch.long).to(device)  # FIX
    tt = torch.tensor([[t]]).to(device)                              # FIX
    cost_return = mat.__getd__(st, a, b, tt).item()
    tour_time += cost_return

    return tour_time


def nnr_tour_averaged(sample, mat, num_runs=10, top_k=3, probabilities=None):
    best_cost = float('inf')
    for _ in range(num_runs):
        cost = nnr_tour(sample, mat, top_k, probabilities)
        if cost < best_cost:
            best_cost = cost
    return best_cost


def greedy_edge_tour(sample, mat):
    x = sample.clone()
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    num_nodes = len(node_ids)
    st = torch.arange(n).unsqueeze(0).to(device)       # FIX
    t_initial = torch.tensor([[0.0]]).to(device)        # FIX

    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            a = torch.tensor([[node_ids[i]]], dtype=torch.long).to(device)  # FIX
            b = torch.tensor([[node_ids[j]]], dtype=torch.long).to(device)  # FIX
            cost = mat.__getd__(st, a, b, t_initial).item()
            edges.append((cost, node_ids[i], node_ids[j]))

    edges.sort(key=lambda x: x[0])

    degree = {node: 0 for node in node_ids}
    adjacency = {node: [] for node in node_ids}
    selected_edges = []

    parent = {node: node for node in node_ids}

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

    tour = [start]
    visited_nodes = {start}
    current = start

    while len(tour) < num_nodes:
        found = False
        for neighbor in adjacency[current]:
            if neighbor not in visited_nodes:
                tour.append(neighbor)
                visited_nodes.add(neighbor)
                current = neighbor
                found = True
                break
        if not found:
            for node in node_ids:
                if node not in visited_nodes:
                    tour.append(node)
                    visited_nodes.add(node)
                    current = node
                    break

    total_time = 0.0
    t = 0.0

    for i in range(len(tour) - 1):
        a = torch.tensor([[tour[i]]], dtype=torch.long).to(device)      # FIX
        b = torch.tensor([[tour[i + 1]]], dtype=torch.long).to(device)  # FIX
        tt = torch.tensor([[t]]).to(device)                              # FIX
        travel_time = mat.__getd__(st, a, b, tt).item()
        total_time += travel_time
        t += travel_time

    a = torch.tensor([[tour[-1]]], dtype=torch.long).to(device)  # FIX
    b = torch.tensor([[tour[0]]], dtype=torch.long).to(device)   # FIX
    tt = torch.tensor([[t]]).to(device)                           # FIX
    travel_time = mat.__getd__(st, a, b, tt).item()
    total_time += travel_time

    return total_time


def run_heuristic(val_dataset, mat, heuristic_type='nearest_neighbor'):
    tour_costs = []

    if heuristic_type in ['nearest_neighbor', 'linear_time', 'greedy']:
        tour_func = nearest_neighbor_tour
    elif heuristic_type == 'nn_plus_one':
        tour_func = nn_plus_one_tour
    elif heuristic_type == 'two_opt':
        tour_func = two_opt_tour
    elif heuristic_type == 'nnr':
        tour_func = nnr_tour
    elif heuristic_type == 'nnr_best':
        tour_func = lambda s, m: nnr_tour_averaged(s, m, num_runs=10)
    elif heuristic_type == 'greedy_edge':
        tour_func = greedy_edge_tour
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")

    total_samples = len(val_dataset)
    print(f"\nProcessing {total_samples} instances with {heuristic_type} heuristic...")

    for i in tqdm(range(total_samples), desc=f"{heuristic_type}", unit="inst", ncols=80):
        sample = val_dataset[i]
        cost = tour_func(sample, mat)
        tour_costs.append(cost)

    return np.array(tour_costs)


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def validate(mat, model, dataset, opts):
    print('Validating...')
    cost = rollout(mat, model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost


def train_batch(mat, model, optimizer, baseline, epoch, batch_id, step,
                batch, tb_logger, opts):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
    cost, log_likelihood, _ = model(mat, x)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss
    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)


def train_epoch(mat, ci, model, optimizer, baseline, lr_scheduler, epoch, val_dataset, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    training_dataset = baseline.wrap_dataset(TSPDataset(ci, size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)

    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(training_dataloader):
        train_batch(mat, model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts)
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(mat, model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)


def load_args_from_checkpoint(checkpoint_dir):
    """
    FIX: Load saved args.json from checkpoint directory to recover
    step feature flags used during training.
    """
    args_path = os.path.join(checkpoint_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            return json.load(f)
    return {}


def compare_models(model_paths, model_names, mat, val_dataset, opts,
                   include_heuristics=True, heuristic_types=None):
    print("\n" + "="*60)
    print("STARTING MODEL COMPARISON")
    print("="*60)

    results = {}

    if heuristic_types is None:
        heuristic_types = [
            'nearest_neighbor',
            'greedy_edge',
            'nn_plus_one',
            'two_opt',
            'nnr_best'
        ]

    if include_heuristics:
        print("\n=== Evaluating Heuristic Baselines ===")
        for h_type in heuristic_types:
            print(f"\nRunning {h_type} heuristic...")
            try:
                h_costs = run_heuristic(val_dataset, mat, heuristic_type=h_type)
                h_costs_minutes = h_costs * 1440
                results[f'Heuristic-{h_type}'] = {
                    'avg_cost': np.mean(h_costs_minutes),
                    'std_cost': np.std(h_costs_minutes),
                    'min_cost': np.min(h_costs_minutes),
                    'max_cost': np.max(h_costs_minutes),
                    'costs': h_costs_minutes,
                    'routes': None
                }
                print(f'  {h_type} mean cost: {np.mean(h_costs_minutes):.2f} ± {np.std(h_costs_minutes):.2f} minutes')
            except Exception as e:
                print(f"  ERROR running {h_type}: {e}")
                continue

    for name, path in zip(model_names, model_paths):
        print(f"\n=== Evaluating {name} model ===")
        print(f"Loading from: {path}")

        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path}")
            continue

        try:
            load_data_eval = torch.load(path, map_location='cpu')

            has_step_mlp = any('step_mlp' in key for key in load_data_eval['model'].keys())
            has_temp_mlp = any('temp_mlp' in key for key in load_data_eval['model'].keys())
            has_cost_aware_gating = any('lambda_heuristic' in key or 'heuristic_computer' in key
                                        for key in load_data_eval['model'].keys())
            has_time_slicing = any('embed_windowed_traffic' in key
                                   for key in load_data_eval['model'].keys())

            print(f"  Detected Step-MLP: {has_step_mlp}")
            print(f"  Detected Temp-MLP: {has_temp_mlp}")
            print(f"  Detected Cost-Aware Gating: {has_cost_aware_gating}")
            print(f"  Detected Time Slicing: {has_time_slicing}")

            # FIX: Load saved args from checkpoint directory to recover step feature flags
            checkpoint_dir = os.path.dirname(path)
            saved_args = load_args_from_checkpoint(checkpoint_dir)
            if saved_args:
                print(f"  Loaded saved args from {checkpoint_dir}/args.json")
            else:
                print(f"  WARNING: No args.json found in {checkpoint_dir}, using opts defaults")

            def get_flag(flag_name, fallback=False):
                """Get flag from saved_args first, then opts, then fallback."""
                if flag_name in saved_args:
                    return saved_args[flag_name]
                return getattr(opts, flag_name, fallback)

            checkpoint_n_cities = saved_args.get('n_cities', load_data_eval.get('n_cities', opts.n_cities))

            eval_model = AttentionModel(
                opts.embedding_dim,
                opts.hidden_dim,
                n_encode_layers=opts.n_encode_layers,
                mask_inner=True,
                mask_logits=True,
                normalization=opts.normalization,
                tanh_clipping=opts.tanh_clipping,
                checkpoint_encoder=opts.checkpoint_encoder,
                shrink_size=opts.shrink_size,
                step_mlp_dim=opts.step_mlp_dim if (has_step_mlp or has_temp_mlp) else None,
                use_step_mlp=has_step_mlp,
                use_temp_mlp=has_temp_mlp,
                # FIX: Use saved_args to recover exact flags used during training
                use_step_ratio=get_flag('use_step_ratio') if (has_step_mlp or has_temp_mlp) else False,
                use_last_3_nodes=get_flag('use_last_3_nodes') if (has_step_mlp or has_temp_mlp) else False,
                use_visited_mean=get_flag('use_visited_mean') if (has_step_mlp or has_temp_mlp) else False,
                use_unvisited_mean=get_flag('use_unvisited_mean') if (has_step_mlp or has_temp_mlp) else False,
                use_sin_cos_time=get_flag('use_sin_cos_time') if (has_step_mlp or has_temp_mlp) else False,
                use_linear_time=get_flag('use_linear_time') if (has_step_mlp or has_temp_mlp) else False,
                use_depot_distance=get_flag('use_depot_distance') if (has_step_mlp or has_temp_mlp) else False,
                use_tour_length=get_flag('use_tour_length') if (has_step_mlp or has_temp_mlp) else False,
                use_mean_dist_unvisited=get_flag('use_mean_dist_unvisited') if (has_step_mlp or has_temp_mlp) else False,
                step_features_v1=get_flag('step_features_v1') if (has_step_mlp or has_temp_mlp) else False,
                step_features_v2=get_flag('step_features_v2') if (has_step_mlp or has_temp_mlp) else False,
                step_features_v1_light=get_flag('step_features_v1_light') if (has_step_mlp or has_temp_mlp) else False,
                step_features_v2_light=get_flag('step_features_v2_light') if (has_step_mlp or has_temp_mlp) else False,
                step_features_minimal=get_flag('step_features_minimal') if (has_step_mlp or has_temp_mlp) else False,
                input_size=opts.graph_size + 1,
                max_t=12,
                n_cities=checkpoint_n_cities,
                use_cost_aware_gating=has_cost_aware_gating,
                heuristic_type=get_flag('heuristic_type', 'linear_time') if has_cost_aware_gating else 'linear_time',
                lambda_heuristic=get_flag('lambda_heuristic', 1.0) if has_cost_aware_gating else 1.0,
                use_nonlinear_transform=get_flag('use_nonlinear_transform') if has_cost_aware_gating else False,
                transform_type=get_flag('transform_type', 'piecewise') if has_cost_aware_gating else 'piecewise',
                use_time_slicing=has_time_slicing,
                window_size_W=get_flag('window_size_W', 12) if has_time_slicing else 12,
                use_decoder_mlp=get_flag('use_decoder_mlp'),
                decoder_mlp_hidden=get_flag('decoder_mlp_hidden', 512),
                use_decoder_mlp_pre=get_flag('use_decoder_mlp_pre'),
                decoder_mlp_pre_hidden=get_flag('decoder_mlp_pre_hidden', 512),
                lambda_heuristic_learnable=get_flag('lambda_heuristic_learnable') if has_cost_aware_gating else False,
            ).to(opts.device)

            eval_model.load_state_dict(load_data_eval['model'])
            eval_model.eval()

            ans, cost = roll(mat, eval_model, val_dataset, opts)
            cost_minutes = cost.numpy() * 1440
            avg_cost = np.mean(cost_minutes)
            std_cost = np.std(cost_minutes)

            results[name] = {
                'avg_cost': avg_cost,
                'std_cost': std_cost,
                'min_cost': np.min(cost_minutes),
                'max_cost': np.max(cost_minutes),
                'costs': cost_minutes,
                'routes': ans.numpy()
            }

            print(f'{name} Results:')
            print(f'  Average cost: {avg_cost:.2f} ± {std_cost:.2f} minutes')
            print(f'  Min cost: {np.min(cost_minutes):.2f} minutes')
            print(f'  Max cost: {np.max(cost_minutes):.2f} minutes')

        except Exception as e:
            print(f"ERROR loading {name} model: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*70)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_cost'])

    print(f"\n{'Method':<30} {'Mean (min)':<12} {'Std (min)':<12} {'Min':<12} {'vs Best':<12}")
    print("-"*78)

    best_cost = sorted_results[0][1]['avg_cost'] if sorted_results else 0
    for name, result in sorted_results:
        gap = ((result['avg_cost'] - best_cost) / best_cost) * 100 if best_cost > 0 else 0
        print(f"{name:<30} {result['avg_cost']:<12.2f} {result['std_cost']:<12.2f} "
              f"{result['min_cost']:<12.2f} {gap:>+8.2f}%")

    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test)")
    print("="*70)

    if len(sorted_results) >= 2:
        best_name, best_result = sorted_results[0]

        print(f"\n--- vs Best: {best_name} ({best_result['avg_cost']:.2f} min) ---")
        print("-"*50)
        for name, result in sorted_results[1:]:
            try:
                t_stat, p_value = stats.ttest_rel(best_result['costs'], result['costs'])
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  {name:<30} p={p_value:.6f} {sig}")
            except Exception as e:
                print(f"  {name:<30} Statistical test failed: {e}")

        if model_names and model_names[0] in results:
            designated_baseline = model_names[0]
            bl_result = results[designated_baseline]
            print(f"\n--- vs Designated Baseline: {designated_baseline} ({bl_result['avg_cost']:.2f} min) ---")
            print("-"*50)
            for name, result in sorted(results.items(), key=lambda x: x[1]['avg_cost']):
                if name == designated_baseline:
                    continue
                try:
                    t_stat, p_value = stats.ttest_rel(bl_result['costs'], result['costs'])
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    direction = "better" if result['avg_cost'] < bl_result['avg_cost'] else "worse"
                    print(f"  {name:<30} p={p_value:.6f} {sig}  ({direction} than baseline)")
                except Exception as e:
                    print(f"  {name:<30} Statistical test failed: {e}")

    for name, result in results.items():
        safe_name = name.replace(' ', '_').replace('/', '_').replace('-', '_')
        if result['routes'] is not None:
            np.savetxt(f'comparison_{safe_name}_routes.txt', result['routes'], fmt='%d')
        np.savetxt(f'comparison_{safe_name}_costs.txt', result['costs'], fmt='%.6f')

    excel_filename = f'comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        summary_data = []
        for name, result in sorted_results:
            gap = ((result['avg_cost'] - best_cost) / best_cost) * 100 if best_cost > 0 else 0
            summary_data.append({
                'Method': name,
                'Mean Cost (min)': result['avg_cost'],
                'Std Cost (min)': result['std_cost'],
                'Min Cost (min)': result['min_cost'],
                'Max Cost (min)': result['max_cost'],
                'Gap vs Best (%)': gap,
                'Num Samples': len(result['costs'])
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        max_len = max((len(result['costs']) for _, result in sorted_results), default=0)
        all_costs_data = {
            name: np.pad(result['costs'].astype(float),
                         (0, max_len - len(result['costs'])),
                         mode='constant', constant_values=np.nan)
            for name, result in sorted_results
        }
        df_all_costs = pd.DataFrame(all_costs_data)
        df_all_costs.to_excel(writer, sheet_name='All Costs', index=False)

        if len(sorted_results) >= 2:
            pairwise_data = []
            for i, (name1, result1) in enumerate(sorted_results):
                for j, (name2, result2) in enumerate(sorted_results):
                    if i >= j:
                        continue
                    try:
                        t_stat, p_value = stats.ttest_rel(result1['costs'], result2['costs'])
                        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                        pairwise_data.append({
                            'Method A': name1,
                            'Mean A (min)': round(result1['avg_cost'], 2),
                            'Method B': name2,
                            'Mean B (min)': round(result2['avg_cost'], 2),
                            'Diff (A-B)': round(result1['avg_cost'] - result2['avg_cost'], 2),
                            't-statistic': round(t_stat, 4),
                            'p-value': round(p_value, 6),
                            'Significance': sig
                        })
                    except Exception as e:
                        pairwise_data.append({
                            'Method A': name1,
                            'Mean A (min)': round(result1['avg_cost'], 2),
                            'Method B': name2,
                            'Mean B (min)': round(result2['avg_cost'], 2),
                            'Diff (A-B)': round(result1['avg_cost'] - result2['avg_cost'], 2),
                            't-statistic': None,
                            'p-value': None,
                            'Significance': f'Error: {e}'
                        })
            df_pairwise = pd.DataFrame(pairwise_data)
            df_pairwise.to_excel(writer, sheet_name='Pairwise Comparisons', index=False)

    print(f"\n" + "="*70)
    print(f"Results saved to: {excel_filename}")
    print("="*70)

    return results


def run(opts):
    pp.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    ci = Cities()
    mat = DistanceMatrix(ci, load_dir='./m1/data.csv', max_time_step=12)
    np.savetxt('mat.txt', mat.mat.cpu().numpy(), fmt='%.6f')
    np.savetxt('m2.txt', mat.m2.cpu().numpy(), fmt='%.6f')
    np.savetxt('m3.txt', mat.m3.cpu().numpy(), fmt='%.6f')
    np.savetxt('m4.txt', mat.m4.cpu().numpy(), fmt='%.6f')

    model_class = AttentionModel
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        step_mlp_dim=opts.step_mlp_dim,
        use_step_mlp=opts.use_step_mlp,
        use_temp_mlp=getattr(opts, 'use_temp_mlp', False),
        use_step_ratio=getattr(opts, 'use_step_ratio', False),
        use_last_3_nodes=getattr(opts, 'use_last_3_nodes', False),
        use_visited_mean=getattr(opts, 'use_visited_mean', False),
        use_unvisited_mean=getattr(opts, 'use_unvisited_mean', False),
        use_sin_cos_time=getattr(opts, 'use_sin_cos_time', False),
        use_linear_time=getattr(opts, 'use_linear_time', False),
        use_depot_distance=getattr(opts, 'use_depot_distance', False),
        use_tour_length=getattr(opts, 'use_tour_length', False),
        use_mean_dist_unvisited=getattr(opts, 'use_mean_dist_unvisited', False),
        step_features_v1=getattr(opts, 'step_features_v1', False),
        step_features_v2=getattr(opts, 'step_features_v2', False),
        step_features_v1_light=getattr(opts, 'step_features_v1_light', False),
        step_features_v2_light=getattr(opts, 'step_features_v2_light', False),
        step_features_minimal=getattr(opts, 'step_features_minimal', False),
        input_size=opts.graph_size + 1,
        max_t=12,
        n_cities=opts.n_cities,
        use_cost_aware_gating=getattr(opts, 'use_cost_aware_gating', False),
        heuristic_type=getattr(opts, 'heuristic_type', 'linear_time'),
        lambda_heuristic=getattr(opts, 'lambda_heuristic', 1.0),
        lambda_heuristic_learnable=getattr(opts, 'lambda_heuristic_learnable', False),
        use_nonlinear_transform=getattr(opts, 'use_nonlinear_transform', False),
        transform_type=getattr(opts, 'transform_type', 'piecewise'),
        use_decoder_mlp=getattr(opts, 'use_decoder_mlp', False),
        decoder_mlp_hidden=getattr(opts, 'decoder_mlp_hidden', 512),
        use_decoder_mlp_pre=getattr(opts, 'use_decoder_mlp_pre', False),
        decoder_mlp_pre_hidden=getattr(opts, 'decoder_mlp_pre_hidden', 512),
    ).to(opts.device)

    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(mat, ci, model, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    val_dataset = TSPDataset(ci, size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    _, ind = torch.max(val_dataset.data, dim=2)

    if opts.resume:
        epoch_resume = 999
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    print("\n" + "="*60)
    print("EVALUATING CURRENT MODEL")
    print("="*60)
    model2 = baseline.baseline.model
    ans, cost = roll(mat, model2, val_dataset, opts)
    print('Current model average cost:', torch.mean(cost).item() * 1440, 'minutes')
    np.savetxt('answer.txt', ans.numpy(), fmt='%d')
    np.savetxt('costs.txt', cost.numpy(), fmt='%.6f')

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    if opts.compare_models and opts.compare_names:
        if len(opts.compare_models) != len(opts.compare_names):
            print(f"ERROR: Number of model paths ({len(opts.compare_models)}) does not match "
                  f"number of names ({len(opts.compare_names)})")
        else:
            model_paths = opts.compare_models
            model_names = opts.compare_names
            print(f"Comparing {len(model_paths)} models from command-line arguments...")
            comparison_results = compare_models(
                model_paths,
                model_names,
                mat,
                val_dataset,
                opts,
                include_heuristics=not opts.no_heuristics
            )
    else:
        print("No --compare_models specified, using hardcoded paths.")
        model_paths = [
            'outputs/tsp_19/baseline_20251030T082105/epoch-99.pt',
            'outputs/tsp_19/mlp_gating_linear_model_20251022T113722/epoch-9.pt'
        ]
        model_names = ['Baseline', 'MLP+Gating']
        comparison_results = compare_models(
            model_paths,
            model_names,
            mat,
            val_dataset,
            opts,
            include_heuristics=not opts.no_heuristics
        )


if __name__ == "__main__":
    run(get_options())