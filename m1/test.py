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

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger



from options import get_options
from baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
import warnings
import pprint as pp
warnings = warnings.filterwarnings("ignore")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Cities:
    def __init__(self, n_cities = 100):
        self.n_cities = n_cities
        self.cities = torch.rand((n_cities, 2))
    def __getdis__(self,i, j):
        return torch.sqrt(torch.sum(torch.pow(torch.sub(self.cities[i], self.cities[j]), 2)))


class DistanceMatrix:
    def __init__(self, ci, max_time_step = 100, load_dir = None):
        self.n_c = ci.n_cities
        self.max_time_step = max_time_step
        with torch.no_grad():
            self.mat = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m2 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m3 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.m4 = torch.zeros(self.n_c * self.n_c * max_time_step, device=device)
            self.var = torch.full((ci.n_cities * ci.n_cities, 1), 0.03, device = device).view(-1)
            #self.var = torch.rand(ci.n_cities * ci.n_cities, device = device) * 0.06
            #self.var = torch.randn(ci.n_cities * ci.n_cities, device = device) * 0.05 + 0.03
            if (load_dir is not None):
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
        res,_ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim = -1), dim = -1)
        res,_ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim = -1), dim = -1)
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
        res,_ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim = -1), dim = -1)
        res,_ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim = -1), dim = -1)
        return res.view(s0, s1)
def rollout(mat, model, dataset, opts):
    # Put in greedy evaluation mode!
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
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()
    c = []
    p = []
    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _, pi = model(mat, move_to(bat, opts.device))
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
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


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


    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))


    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))


    # Log values to tensorboard
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


        if (filename is None):
            self.data_set = []
            l = torch.rand((num_samples, ci.n_cities - 1))
            sorted, ind = torch.sort(l)
            ind = ind.unsqueeze(2).expand(num_samples, ci.n_cities - 1, 2)
            ind = ind[:,:size,:] + 1
            ff = ci.cities.unsqueeze(0)
            ff = ff.expand(num_samples, ci.n_cities, 2)
            f = torch.gather(ff, dim = 1, index = ind)
            f = f.permute(0,2,1)
            depot = ci.cities[0].view(1, 2, 1).expand(num_samples, 2, 1)
            self.static = torch.cat((depot, f), dim = 2)
            depot = torch.zeros(num_samples, 1, 1, dtype=torch.long)
            ind = ind[:,:,0:1]
            ind = torch.cat((depot, ind), dim=1)
        else:
            ff = np.loadtxt(filename, delimiter = ' ')
            ind = torch.tensor(ff, dtype=torch.long).unsqueeze(2)
            # Limit to num_samples if file has more
            file_num_samples = ind.size(0)
            if file_num_samples > num_samples:
                ind = ind[:num_samples]  # Take only first num_samples
                num_samples = ind.size(0)  # Update to actual size used
            elif file_num_samples < num_samples:
                # File has fewer samples than requested - use what's available
                num_samples = file_num_samples
        
        self.data = torch.zeros(num_samples, size+1, ci.n_cities)
        self.data = self.data.scatter_(2, ind, 1.)
        self.size = len(self.data)
    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        return self.data[idx]

def run_nearest_neighbor_heuristic(val_dataset, mat):
    """
    Compute NN/GR solution for all samples in val_dataset.
    Returns a numpy array of tour costs.
    """
    tour_costs = []
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        cost = nearest_neighbor_tour(sample, mat)
        tour_costs.append(cost)
    return np.array(tour_costs)

def nearest_neighbor_tour(sample, mat):
    x = sample.clone()  # [tour_len, n]
    seq_len, n = x.shape
    node_ids = x.argmax(dim=1).tolist()
    visited = [node_ids[0]]
    tour = [node_ids[0]]
    unvisited = set(range(n))
    unvisited.remove(node_ids[0])
    current = node_ids[0]
    t = 0.0
    tour_time = 0
    st = torch.arange(n).unsqueeze(0)  # shape [1, n]
    while unvisited:
        min_cost, next_node = float('inf'), None
        for j in unvisited:
            a = torch.tensor([[current]], dtype=torch.long)
            b = torch.tensor([[j]], dtype=torch.long)
            tt = torch.tensor([[t]])
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
    # Return to depot
    a = torch.tensor([[current]], dtype=torch.long)
    b = torch.tensor([[node_ids[0]]], dtype=torch.long)
    tt = torch.tensor([[t]])
    cost_return = mat.__getd__(st, a, b, tt)
    tour_time += cost_return.item() if hasattr(cost_return, 'item') else float(cost_return)
    return tour_time

def linear_time_tour(sample, mat):
    """
    Linear time heuristic: always pick the node with shortest travel time.
    This is essentially the same as nearest_neighbor for your case.
    """
    # For DTSP, linear_time and nearest_neighbor are the same
    return nearest_neighbor_tour(sample, mat)

def nn_plus_one_tour(sample, mat):
    """
    NN+1 heuristic: OPTIMIZED VERSION with vectorization
    """
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
    st = torch.arange(n).unsqueeze(0)
    lookahead_weight = 0.5
    
    while unvisited:
        unvisited_list = list(unvisited)
        num_unvisited = len(unvisited_list)
        
        if num_unvisited == 1:
            # Only one node left - no need for lookahead
            next_node = unvisited_list[0]
        else:
            # Vectorize: compute costs to all candidates at once
            current_tensor = torch.tensor([[current]], dtype=torch.long)
            candidates_tensor = torch.tensor([unvisited_list], dtype=torch.long)
            tt_tensor = torch.tensor([[t]])
            
            # Compute costs to all candidates (vectorized)
            costs_to_candidates = []
            for j in unvisited_list:
                j_tensor = torch.tensor([[j]], dtype=torch.long)
                cost = mat.__getd__(st, current_tensor, j_tensor, tt_tensor).item()
                costs_to_candidates.append(cost)
            
            # For each candidate, compute lookahead cost
            combined_costs = []
            for idx, j in enumerate(unvisited_list):
                cost_to_j = costs_to_candidates[idx]
                
                # Compute lookahead: min cost from j to other unvisited nodes
                lookahead_costs = []
                for k in unvisited_list:
                    if k != j:
                        j_tensor = torch.tensor([[j]], dtype=torch.long)
                        k_tensor = torch.tensor([[k]], dtype=torch.long)
                        tt2_tensor = torch.tensor([[t + cost_to_j]])
                        cost_jk = mat.__getd__(st, j_tensor, k_tensor, tt2_tensor).item()
                        lookahead_costs.append(cost_jk)
                
                lookahead_cost = min(lookahead_costs) if lookahead_costs else 0
                combined_cost = cost_to_j + lookahead_weight * lookahead_cost
                combined_costs.append((combined_cost, j))
            
            # Select node with minimum combined cost
            next_node = min(combined_costs, key=lambda x: x[0])[1]
        
        # Move to next node
        a = torch.tensor([[current]], dtype=torch.long)
        b = torch.tensor([[next_node]], dtype=torch.long)
        tt = torch.tensor([[t]])
        cost_ij = mat.__getd__(st, a, b, tt).item()
        
        tour_time += cost_ij
        t += cost_ij
        visited.append(next_node)
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    # Return to depot
    a = torch.tensor([[current]], dtype=torch.long)
    b = torch.tensor([[node_ids[0]]], dtype=torch.long)
    tt = torch.tensor([[t]])
    cost_return = mat.__getd__(st, a, b, tt).item()
    tour_time += cost_return
    return tour_time

def run_heuristic(val_dataset, mat, heuristic_type='nearest_neighbor'):
    """
    Run any heuristic type on the validation dataset.
    
    Args:
        val_dataset: TSPDataset
        mat: DistanceMatrix
        heuristic_type: 'nearest_neighbor', 'linear_time', 'nn_plus_one'
    
    Returns:
        numpy array of tour costs
    """
    tour_costs = []
    
    if heuristic_type == 'nearest_neighbor' or heuristic_type == 'linear_time':
        tour_func = nearest_neighbor_tour
    elif heuristic_type == 'nn_plus_one':
        tour_func = nn_plus_one_tour
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")
    
    # Add progress bar
    total_samples = len(val_dataset)
    print(f"\nProcessing {total_samples} instances with {heuristic_type} heuristic...")
    
    for i in tqdm(range(total_samples), desc=f"{heuristic_type}", unit="inst", ncols=80):
        sample = val_dataset[i]
        cost = tour_func(sample, mat)
        tour_costs.append(cost)
    
    return np.array(tour_costs)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    #print(len(param_groups[0]['params']))
    #print('param_groups', param_groups)
    #print('group[params]', [group['params'] for group in param_groups])
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    #print(len(param_groups[0]['params']))
    #print('ss', [g_norm for g_norm in grad_norms])
    #print('grad_norms', grad_norms)
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    #print('grad_norms_clipped', grad_norms_clipped)
    return grad_norms, grad_norms_clipped



def validate(mat, model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(mat, model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))


    return avg_cost



def train_batch(
        mat,
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
   # print(x.size())
    # Evaluate model, get costs and log probabilities
    cost, log_likelihood,_ = model(mat, x)


    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)


    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss


    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()


    # Logging
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


    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(TSPDataset(ci, size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size)


    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")


    for batch_id, batch in enumerate(training_dataloader):


        train_batch(
            mat,
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )


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

# Define model comparison function
def compare_models(model_paths, model_names, mat, val_dataset, opts, include_heuristics=True):
    """Compare multiple trained models on the same test dataset"""
    
    print("\n" + "="*60)
    print("STARTING MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    # Evaluate all heuristics first (if requested)
    if include_heuristics:
        print("\n=== Evaluating Heuristic Baselines ===")
        heuristic_types = ['nearest_neighbor', 'nn_plus_one']
        for h_type in heuristic_types:
            print(f"Running {h_type} heuristic...")
            h_costs = run_heuristic(val_dataset, mat, heuristic_type=h_type)
            h_costs_minutes = h_costs * 1440  # Convert to minutes
            results[f'Heuristic-{h_type}'] = {
                'avg_cost': np.mean(h_costs_minutes),
                'std_cost': np.std(h_costs_minutes),
                'costs': h_costs_minutes,
                'routes': None  # Heuristics don't save routes in current implementation
            }
            print(f'  {h_type} mean cost: {np.mean(h_costs_minutes):.2f} ± {np.std(h_costs_minutes):.2f} minutes')
    
    # Then evaluate all models
    for name, path in zip(model_names, model_paths):
        print(f"\n=== Evaluating {name} model ===")
        print(f"Loading from: {path}")
        
        # Load checkpoint
        if not os.path.exists(path):
            print(f"ERROR: Model file not found: {path}")
            continue
            
        try:
            load_data_eval = torch.load(path, map_location='cpu')
            
            # Check if the saved model has Step-MLP components
            has_step_mlp = any('step_mlp' in key for key in load_data_eval['model'].keys())
            
            # Check if the saved model has cost-aware gating components
            has_cost_aware_gating = any('lambda_heuristic' in key or 'heuristic_computer' in key 
                                       for key in load_data_eval['model'].keys())
            
            print(f"Detected Step-MLP in model: {has_step_mlp}")
            print(f"Detected Cost-Aware Gating in model: {has_cost_aware_gating}")
            
            # Initialize model with appropriate architecture
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
                step_mlp_dim=opts.step_mlp_dim if has_step_mlp else None,
                use_step_mlp=has_step_mlp,
                use_temp_mlp=getattr(opts, 'use_temp_mlp', False) if has_step_mlp else False,  # NEW
                input_size=opts.graph_size+1,
                max_t=12,
                # Cost-Aware Gating parameters - Add these
                use_cost_aware_gating=has_cost_aware_gating,
                heuristic_type=getattr(opts, 'heuristic_type', 'linear_time') if has_cost_aware_gating else 'linear_time',
                lambda_heuristic=getattr(opts, 'lambda_heuristic', 1.0) if has_cost_aware_gating else 1.0,
                use_nonlinear_transform=getattr(opts, 'use_nonlinear_transform', False) if has_cost_aware_gating else False,
                transform_type=getattr(opts, 'transform_type', 'piecewise') if has_cost_aware_gating else 'piecewise'
            ).to(opts.device)
            
            # Load trained weights
            eval_model.load_state_dict(load_data_eval['model'])
            eval_model.eval()
            
            # Evaluate on test dataset
            ans, cost = roll(mat, eval_model, val_dataset, opts)
            cost_minutes = cost.numpy() * 1440  # Convert to minutes
            avg_cost = np.mean(cost_minutes)
            std_cost = np.std(cost_minutes)
            
            results[name] = {
                'avg_cost': avg_cost,
                'std_cost': std_cost,
                'costs': cost_minutes,
                'routes': ans.numpy()
            }
            
            print(f'{name} Results:')
            print(f'  Average cost: {avg_cost:.2f} ± {std_cost:.2f} minutes')
            print(f'  Min cost: {np.min(cost_minutes):.2f} minutes')
            print(f'  Max cost: {np.max(cost_minutes):.2f} minutes')
            
        except Exception as e:
            print(f"ERROR loading {name} model: {e}")
            continue
    
    # Print comprehensive comparison
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*60)
    
    # Sort by average cost
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_cost'])
    
    print(f"\n{'Method':<30} {'Mean (min)':<15} {'Std (min)':<15} {'vs Best':<15}")
    print("-"*75)
    
    best_cost = sorted_results[0][1]['avg_cost']
    for name, result in sorted_results:
        improvement = ((best_cost - result['avg_cost']) / best_cost) * 100 if result['avg_cost'] != best_cost else 0.0
        print(f"{name:<30} {result['avg_cost']:<15.2f} {result['std_cost']:<15.2f} {improvement:>+6.2f}%")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    if len(results) >= 2:
        baseline_name = model_names[0]
        if baseline_name in results:
            baseline_cost = results[baseline_name]['avg_cost']
            
            print(f"\nBaseline: {baseline_name}")
            print(f"  Average: {baseline_cost:.2f} minutes")
            
            for name in model_names[1:]:
                if name in results:
                    compare_cost = results[name]['avg_cost']
                    improvement = ((baseline_cost - compare_cost) / baseline_cost) * 100
                    
                    print(f"\n{name}:")
                    print(f"  Average: {compare_cost:.2f} minutes")
                    print(f"  Improvement: {improvement:.2f}%")
                    
                    # Statistical significance (paired t-test)
                    try:
                        t_stat, p_value = stats.ttest_rel(
                            results[baseline_name]['costs'],
                            results[name]['costs']
                        )
                        print(f"  T-statistic: {t_stat:.4f}")
                        print(f"  P-value: {p_value:.6f}")
                        if p_value < 0.05:
                            print(f"  *** Statistically significant difference (p < 0.05)")
                        else:
                            print(f"  Not statistically significant (p >= 0.05)")
                    except Exception as e:
                        print(f"  Statistical test failed: {e}")
    

    # Save detailed results to files
    for name, result in results.items():
        safe_name = name.replace(' ', '_').replace('/', '_')
        if result['routes'] is not None:
            np.savetxt(f'comparison_{safe_name}_routes.txt', result['routes'], fmt='%d')
        np.savetxt(f'comparison_{safe_name}_costs.txt', result['costs'], fmt='%.6f')
    
    # ===== SAVE STATISTICS TO EXCEL =====
    excel_filename = f'comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Sheet 1: Summary Statistics for All Models
        summary_data = []
        for name, result in sorted_results:
            summary_data.append({
                'Model Name': name,
                'Mean Cost (minutes)': result['avg_cost'],
                'Std Cost (minutes)': result['std_cost'],
                'Min Cost (minutes)': np.min(result['costs']),
                'Max Cost (minutes)': np.max(result['costs']),
                'Num Samples': len(result['costs']),
                'Improvement vs Best (%)': ((best_cost - result['avg_cost']) / best_cost) * 100 if result['avg_cost'] != best_cost else 0.0
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Sheet 2: Pairwise Comparisons vs Baseline
        if len(results) >= 2 and baseline_name in results:
            baseline_cost = results[baseline_name]['avg_cost']
            comparison_data = []
            
            for name in model_names[1:]:
                if name in results:
                    compare_cost = results[name]['avg_cost']
                    improvement = ((baseline_cost - compare_cost) / baseline_cost) * 100
                    
                    # Statistical test
                    try:
                        t_stat, p_value = stats.ttest_rel(
                            results[baseline_name]['costs'],
                            results[name]['costs']
                        )
                        comparison_data.append({
                            'Model': name,
                            'Baseline': baseline_name,
                            'Mean Cost (minutes)': compare_cost,
                            'Baseline Cost (minutes)': baseline_cost,
                            'Improvement (%)': improvement,
                            'T-statistic': t_stat,
                            'P-value': p_value,
                            'Statistically Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
                        })
                    except Exception as e:
                        comparison_data.append({
                            'Model': name,
                            'Baseline': baseline_name,
                            'Mean Cost (minutes)': compare_cost,
                            'Baseline Cost (minutes)': baseline_cost,
                            'Improvement (%)': improvement,
                            'T-statistic': 'Error',
                            'P-value': 'Error',
                            'Statistically Significant (p<0.05)': str(e)
                        })
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_excel(writer, sheet_name='Pairwise Comparisons', index=False)
        
        # Sheet 3: All Costs (one column per model)
        all_costs_data = {}
        max_len = max(len(result['costs']) for result in results.values())
        
        for name, result in sorted_results:
            costs = result['costs']
            # Pad with NaN if this model has fewer samples
            padded_costs = np.pad(costs, (0, max_len - len(costs)), 
                                 constant_values=np.nan) if len(costs) < max_len else costs
            all_costs_data[name] = padded_costs
        
        df_all_costs = pd.DataFrame(all_costs_data)
        df_all_costs.to_excel(writer, sheet_name='All Costs', index=False)
        
        # Sheet 4: Metadata
        metadata = {
            'Comparison Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Number of Models': [len(model_names)],
            'Number of Heuristics': [len([r for r in results.keys() if r.startswith('Heuristic-')])],
            'Total Samples': [len(val_dataset)],
            'Baseline Model': [baseline_name if len(results) >= 2 and baseline_name in results else 'N/A']
        }
        df_metadata = pd.DataFrame(metadata)
        df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"\n" + "="*60)
    print(f"Results saved to:")
    print(f"  - Excel file: {excel_filename}")
    print(f"  - Route files: comparison_*_routes.txt")
    print(f"  - Cost files: comparison_*_costs.txt")
    print("="*60)
    
    return results


def run(opts):
    # Pretty print the run args
    print(123)
    pp.pprint(vars(opts))


    # Set the random seed
    torch.manual_seed(opts.seed)


    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))


    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)


    # Set the device
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Figure out what's the problem


    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    ci = Cities()
    mat = DistanceMatrix(ci, load_dir='./m1/data.csv', max_time_step = 12)
    #np.savetxt('var12.txt', mat.var.cpu().numpy(), fmt='%.6f')
    np.savetxt('mat.txt', mat.mat.cpu().numpy(), fmt='%.6f')
    np.savetxt('m2.txt', mat.m2.cpu().numpy(), fmt='%.6f')
    np.savetxt('m3.txt', mat.m3.cpu().numpy(), fmt='%.6f')
    np.savetxt('m4.txt', mat.m4.cpu().numpy(), fmt='%.6f')
    # Initialize model
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
        use_step_mlp=opts.use_step_mlp,  # Add this
        use_temp_mlp=getattr(opts, 'use_temp_mlp', False),  # NEW
        input_size=opts.graph_size+1,
        max_t=12,
        # Cost-Aware Gating parameters - Add these
        use_cost_aware_gating=getattr(opts, 'use_cost_aware_gating', False),
        heuristic_type=getattr(opts, 'heuristic_type', 'linear_time'),
        lambda_heuristic=getattr(opts, 'lambda_heuristic', 1.0),
        use_nonlinear_transform=getattr(opts, 'use_nonlinear_transform', False),
        transform_type=getattr(opts, 'transform_type', 'piecewise')
    ).to(opts.device)



    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})


    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(mat, ci, model, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()


    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)


    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])


    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )


    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)


    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    # Start the actual training loop
    #val_dataset = TSPDataset(ci, size=opts.graph_size, num_samples=opts.val_size, distribution=opts.data_distribution)
    val_dataset = TSPDataset(ci, size=opts.graph_size, num_samples=opts.val_size, filename='data_nodes/node_19.txt', distribution=opts.data_distribution)
    _,ind = torch.max(val_dataset.data, dim=2)
    #np.savetxt('valid_data.txt', ind.numpy(), fmt='%d')
    if opts.resume:
        epoch_resume = 999


        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1


    


    # Evaluate current model (from baseline rollout)
    print("\n" + "="*60)
    print("EVALUATING CURRENT MODEL")
    print("="*60)
    model2 = baseline.baseline.model
    ans, cost = roll(mat, model2, val_dataset, opts)
    print('Current model average cost:', torch.mean(cost).item() * 1440, 'minutes')
    np.savetxt('answer.txt', ans.numpy(), fmt='%d')
    np.savetxt('costs.txt', cost.numpy(), fmt='%.6f')
    
    
    # Example: Compare two models
    # IMPORTANT: Replace these paths with your actual model checkpoint paths
    # print("\n" + "="*60)
    # print("EVALUATING GREEDY NEAREST NEIGHBOR (NN/GR) HEURISTIC")
    # print("="*60)
    # nn_costs = run_nearest_neighbor_heuristic(val_dataset, mat)
    # print('NN/GR mean cost:', np.mean(nn_costs), 'minutes')
    # np.savetxt('nn_gr_heuristic_costs.txt', nn_costs, fmt='%.8f')


    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    # Use command-line arguments if provided, otherwise use hardcoded defaults
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
                include_heuristics=not opts.no_heuristics  # Use command-line flag
            )
    else:
        # Fallback to hardcoded paths if no command-line arguments provided
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
            include_heuristics=not opts.no_heuristics  # Use command-line flag
        )


if __name__ == "__main__":
    run(get_options())
