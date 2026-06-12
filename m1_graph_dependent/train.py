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

import torch.optim as optim
from tensorboard_logger import Logger as TbLogger


from options import get_options
from baselines import NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
from experiment_tracker import ExperimentTracker
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
            self.var = torch.full((ci.n_cities * ci.n_cities, 1), 0.00, device = device).view(-1)
            if (load_dir is not None):
                temp = np.loadtxt(load_dir, delimiter=',', skiprows=0)
                x = np.arange(max_time_step + 1)
                for k in range(self.n_c):
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
    # Check if beam search is requested for evaluation
    if hasattr(opts, 'use_beam_search_eval') and opts.use_beam_search_eval:
        beam_width = getattr(opts, 'beam_width', 5)
        set_decode_type(model, "beam", beam_width=beam_width)
    else:
        # Put in greedy evaluation mode!
        set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            # NEW: Pass start_time if time slicing is enabled
            if getattr(opts, 'use_time_slicing', False):
                # Get start_time_bin - use hasattr to check if it exists
                if hasattr(opts, 'start_time_bin'):
                    start_time_bin = opts.start_time_bin
                    start_time = start_time_bin / 12.0
                else:
                    # If attribute doesn't exist, default to 0
                    start_time = 0.0
                    print(f"[WARNING rollout] start_time_bin attribute not found, using default 0.0")
            else:
                start_time = None
            
            cost, _, _ = model(mat, move_to(bat, opts.device), start_time=start_time)
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat in DataLoader(dataset, batch_size=opts.eval_batch_size)
    ], 0)

def roll(mat, model, dataset, opts):
    # Check if beam search is requested for evaluation
    if hasattr(opts, 'use_beam_search_eval') and opts.use_beam_search_eval:
        beam_width = getattr(opts, 'beam_width', 5)
        set_decode_type(model, "beam", beam_width=beam_width)
        print(f'[TEST EVALUATION] Using BEAM SEARCH (width={beam_width})')
    else:
        # Put in greedy evaluation mode!
        set_decode_type(model, "greedy")
        print('[TEST EVALUATION] Using GREEDY')
    model.eval()
    c = []
    p = []
    
    def eval_model_bat(bat):
        with torch.no_grad():
            # NEW: Pass start_time if time slicing is enabled
            if getattr(opts, 'use_time_slicing', False):
                start_time_bin = getattr(opts, 'start_time_bin', 0)
                start_time = start_time_bin / 12.0
            else:
                start_time = None
            
            cost, _, pi = model(mat, move_to(bat, opts.device), return_pi=True, start_time=start_time)
        return cost.data.cpu(), pi.data.cpu()
    
    for bat in DataLoader(dataset, batch_size=opts.eval_batch_size):
        cost, pi = eval_model_bat(bat)
        for z in range(cost.size(0)):
            c.append(cost[z])
            p.append(pi[z])
    
    return torch.stack(p), torch.stack(c)

def set_decode_type(model, decode_type, beam_width=1):
    model.set_decode_type(decode_type, beam_width=beam_width)
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


class TSPDataset(Dataset):

    def __init__(self, ci, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        if filename is not None:
            print(f'Loading val/test dataset from {filename}')
            saved = torch.load(filename, map_location='cpu')
            self.data = saved['data']
            self.static = saved['static']
            self.size = len(self.data)
            print(f'  Loaded {self.size} instances (graph_size={self.data.size(1)-1})')
            return

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
        self.data = torch.zeros(num_samples, size+1, ci.n_cities)
        self.data = self.data.scatter_(2, ind, 1.)
        self.size = len(self.data)

    def save(self, path):
        torch.save({'data': self.data, 'static': self.static}, path)
        print(f'Saved val/test dataset ({self.size} instances) to {path}')

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]



def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def validate(mat, model, dataset, opts, is_final_eval=False):
    # Validate
    print('Validating...')
    
    # Only use beam search for final evaluation, use greedy for per-epoch validation
    if is_final_eval and hasattr(opts, 'use_beam_search_eval') and opts.use_beam_search_eval:
        # Use beam search for final evaluation
        beam_width = getattr(opts, 'beam_width', 5)
        print(f'[EVALUATION] Using BEAM SEARCH (width={beam_width}) for final evaluation')
        cost = rollout(mat, model, dataset, opts)
    else:
        # Use greedy for per-epoch validation (faster)
        if is_final_eval:
            print('[EVALUATION] Using GREEDY for final evaluation')
        else:
            print('[EVALUATION] Using GREEDY for per-epoch validation')
        original_flag = getattr(opts, 'use_beam_search_eval', False)
        opts.use_beam_search_eval = False  # Temporarily disable beam search
        cost = rollout(mat, model, dataset, opts)
        opts.use_beam_search_eval = original_flag  # Restore original flag
    
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
    
    # NEW: Determine start_time for this batch
    if getattr(opts, 'use_time_slicing', False):
        # Option 1: Use fixed start_time from options (deterministic)
        if hasattr(opts, 'start_time_bin') and opts.start_time_bin is not None:
            start_time = opts.start_time_bin / 12.0  # Convert bin to normalized time
        
        # Option 2: Use random start_time for each batch (adds diversity)
        # Uncomment the lines below to use random start times instead
        # import random
        # start_time = random.uniform(0.0, 1.0)  # Random time across full day
        
        # Option 3: Use random start_time per epoch (same for all batches in epoch)
        # You can add this logic in train_epoch() and pass it as parameter
        else:
            start_time = 0.0  # Default to midnight if not specified
    else:
        start_time = None
    
    cost, log_likelihood, _ = model(mat, x, start_time=start_time)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Evaluate baseline, get baseline loss if any (only for critic)
    #bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

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
                'baseline': baseline.state_dict(),
                'graph_size': opts.graph_size,
                'n_cities': opts.n_cities,
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(mat, model, val_dataset, opts, is_final_eval=False)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    return avg_reward

def run(opts):
    # Pretty print the run args
    pp.pprint(vars(opts))
    
    # Initialize experiment tracker
    tracker = ExperimentTracker('experiment_log.csv')
    
    # Record start time for total training time
    training_start_time = time.time()
    
    # Set the random seed
    torch.manual_seed(opts.seed)

    # NEW: Initialize transfer learning tracking variables
    transfer_info = {
        'training_method': getattr(opts, 'training_method', 'direct'),
        'stage_number': getattr(opts, 'stage_number', 1),
        'transfer_source_path': opts.load_path,
        'transfer_source_graph_size': None,
        'layers_transferred': 0,
        'layers_initialized': 0
    }

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

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)
    
    ci = Cities(n_cities=opts.n_cities)
    mat = DistanceMatrix(ci, load_dir='m1/data.csv', max_time_step = 12)
    np.savetxt('var.txt', mat.var.cpu().numpy(), fmt='%.6f')
    np.savetxt('mat.txt', mat.mat.cpu().numpy(), fmt='%.6f')
    np.savetxt('m2.txt', mat.m2.cpu().numpy(), fmt='%.6f')
    np.savetxt('m3.txt', mat.m3.cpu().numpy(), fmt='%.6f')
    np.savetxt('m4.txt', mat.m4.cpu().numpy(), fmt='%.6f')

    val_dataset = TSPDataset(ci, size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)
    _, ind = torch.max(val_dataset.data, dim=2)
    np.savetxt('valid_data.txt', ind.numpy(), fmt='%d')
    if getattr(opts, 'save_val_dataset', None):
        val_dataset.save(opts.save_val_dataset)

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
        input_size=opts.graph_size + 1,  # Include depot node
        max_t=12,

        # Step-MLP parameters
        step_mlp_dim=opts.step_mlp_dim,
        use_step_mlp=opts.use_step_mlp,
        use_temp_mlp=opts.use_temp_mlp,
        
        # Individual feature flags
        use_step_ratio=getattr(opts, 'use_step_ratio', False),
        use_last_3_nodes=getattr(opts, 'use_last_3_nodes', False),
        use_visited_mean=getattr(opts, 'use_visited_mean', False),
        use_unvisited_mean=getattr(opts, 'use_unvisited_mean', False),
        use_sin_cos_time=getattr(opts, 'use_sin_cos_time', False),
        use_linear_time=getattr(opts, 'use_linear_time', False), # NEW: Use linear time encoding instead of sin/cos (1 dim vs 2 dim, individual flag)
        use_depot_distance=getattr(opts, 'use_depot_distance', False),
        use_tour_length=getattr(opts, 'use_tour_length', False),
        use_mean_dist_unvisited=getattr(opts, 'use_mean_dist_unvisited', False),
        
        # Preset flags (will be mapped to individual flags in model)
        step_features_v1=getattr(opts, 'step_features_v1', False),
        step_features_v2=getattr(opts, 'step_features_v2', False),
        step_features_v1_light=getattr(opts, 'step_features_v1_light', False),
        step_features_v2_light=getattr(opts, 'step_features_v2_light', False),
        step_features_minimal=getattr(opts, 'step_features_minimal', False),
        
        # Cost-Aware Gating parameters
        use_cost_aware_gating=getattr(opts, 'use_cost_aware_gating', False),
        heuristic_type=getattr(opts, 'heuristic_type', 'linear_time'),
        lambda_heuristic=getattr(opts, 'lambda_heuristic', 1.0),
        lambda_heuristic_learnable=getattr(opts, 'lambda_heuristic_learnable', False),
        use_nonlinear_transform=getattr(opts, 'use_nonlinear_transform', False),
        transform_type=getattr(opts, 'transform_type', 'piecewise'),

        # NEW: Time slicing parameters
        use_time_slicing=getattr(opts, 'use_time_slicing', False),
        window_size_W=getattr(opts, 'window_size_W', 12),
        n_cities=opts.n_cities,  # Number of cities in the problem
        use_decoder_mlp=getattr(opts, 'use_decoder_mlp', False),  # Default: False (baseline)
        decoder_mlp_hidden=getattr(opts, 'decoder_mlp_hidden', 512),  # Match encoder
        use_decoder_mlp_pre=getattr(opts, 'use_decoder_mlp_pre', False),  # Option C: Pre-attention MLP
        decoder_mlp_pre_hidden=getattr(opts, 'decoder_mlp_pre_hidden', 512),  # Hidden dim for pre-attention MLP
    ).to(opts.device)

    # NEW: Set refresh parameters on model (if time slicing is enabled)
    # These attributes will be used by _initialize_window_state() and _should_refresh_encoder()
    if getattr(opts, 'use_time_slicing', False):
        model.refresh_strategy = getattr(opts, 'refresh_strategy', 'one_time')
        model.refresh_interval = getattr(opts, 'refresh_interval', 0.5)
        model.buffer_k_moves = getattr(opts, 'buffer_k_moves', 2)
        
        print(f"[Model Init] Time slicing enabled:")
        print(f"  - Refresh strategy: {model.refresh_strategy}")
        print(f"  - Refresh interval: {model.refresh_interval} (normalized time)")
        print(f"  - Buffer k-moves: {model.buffer_k_moves}")

    # Modified safe loading with transfer tracking
    model_ = get_inner_model(model)

    if load_data.get('model', {}):
        loaded_state = load_data['model']
        current_state = model_.state_dict()
        
        # Track source graph size
        if 'graph_size' in load_data:
            transfer_info['transfer_source_graph_size'] = load_data['graph_size']
        
        # Track source n_cities
        if 'n_cities' in load_data and load_data['n_cities'] != opts.n_cities:
            print(f"Warning: checkpoint n_cities {load_data['n_cities']} != current opts.n_cities {opts.n_cities}")
        # Filter out layers with mismatched dimensions
        compatible_state = {}
        incompatible_layers = []
        
        for key, value in loaded_state.items():
            if key in current_state:
                if current_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    incompatible_layers.append(f"{key}: {value.shape} -> {current_state[key].shape}")
            else:
                incompatible_layers.append(f"{key}: not found in current model")
        
        # Load only compatible weights
        model_.load_state_dict({**current_state, **compatible_state})
        
        # Update transfer info
        transfer_info['layers_transferred'] = len(compatible_state)
        transfer_info['layers_initialized'] = len(current_state) - len(compatible_state)
        
        print(f"Loaded {len(compatible_state)}/{len(loaded_state)} layer weights")
        if incompatible_layers:
            print("Skipped incompatible layers:")
            for layer in incompatible_layers:
                print(f"  - {layer}")
    else:
        # Training from scratch
        transfer_info['layers_initialized'] = len(model_.state_dict())


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
    # if 'baseline' in load_data:
    #     baseline.load_state_dict(load_data['baseline'])

    # Load baseline from data, but DON'T load if graph_size changed
    # Because baseline has hard-coded size references
    # Load baseline from data, but DON'T load if graph_size changed
    if 'baseline' in load_data:
        baseline_loaded = False  # Track if we loaded
        
        # Fix: Use 'in' for dictionary, not hasattr
        if 'graph_size' in load_data:
            if load_data['graph_size'] != opts.graph_size:
                print(f"Graph size changed: {load_data['graph_size']} -> {opts.graph_size}")
                print("Skipping baseline load (will use fresh baseline for new size)")
                # baseline_loaded stays False
            else:
                baseline.load_state_dict(load_data['baseline'])
                baseline_loaded = True
        else:
            # No graph_size in checkpoint - check the dataset size
            try:
                saved_dataset = load_data['baseline'].get('dataset', None)
                if saved_dataset and len(saved_dataset) > 0:
                    saved_graph_size = saved_dataset[0].size(0) - 1  # -1 for depot
                    if saved_graph_size != opts.graph_size:
                        print(f"Graph size mismatch: {saved_graph_size} → {opts.graph_size}")
                        print("Skipping baseline load")
                    else:
                        baseline.load_state_dict(load_data['baseline'])
                        baseline_loaded = True
                else:
                    print("Cannot determine previous graph size, skipping baseline load")
            except Exception as e:
                print(f"Error checking baseline compatibility: {e}")
                print("Skipping baseline load for safety")
        
        # CRITICAL FIX: If we skipped loading, reinitialize the baseline completely
        if not baseline_loaded and opts.baseline == 'rollout':
            print("Reinitializing baseline with current model to ensure size compatibility")
            # Get the inner baseline (unwrap WarmupBaseline if present)
            if isinstance(baseline, WarmupBaseline):
                # Force the inner RolloutBaseline to update with current model
                baseline.baseline._update_model(model, epoch=0)
            else:
                baseline._update_model(model, epoch=0)

        
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
        # Skip optimizer loading if graph size changed
        if 'graph_size' in load_data and load_data['graph_size'] != opts.graph_size:
            print(f"Skipping optimizer state load due to graph size mismatch")
            print("Optimizer will start fresh (this is normal for transfer learning)")
        else:
            optimizer.load_state_dict(load_data['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    
    # Start the actual training loop
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        
        # NEW: Reset epoch_start if graph size changed (progressive learning)
        if 'graph_size' in load_data and load_data['graph_size'] != opts.graph_size:
            print("Progressive learning detected: Resetting epoch counter to 0")
            opts.epoch_start = 0  # Start fresh for new graph size
        else:
            opts.epoch_start = epoch_resume + 1  # Continue from checkpoint

    # Initialize tracking variables
    epoch_times = []
    validation_costs = []
    best_epoch = 0
    best_val_cost = float('inf')
    experiment_status = 'completed'
    issues_encountered = []

    # Set up CSV for per-epoch validation costs
    val_cost_csv_path = os.path.join(opts.save_dir, 'val_costs.csv')
    with open(val_cost_csv_path, 'w') as f:
        f.write('epoch,mean_val_cost\n')
    
    # NEW: Track learning dynamics at specific epochs
    epoch_snapshots = {}
    initial_val_cost = None
    
    if opts.eval_only:
        val_cost = validate(mat, model, val_dataset, opts, is_final_eval=True)
    else:
        try:
            for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
                epoch_start_time = time.time()
                
                val_cost = train_epoch(
                    mat,
                    ci,
                    model,
                    optimizer,
                    baseline,
                    lr_scheduler,
                    epoch,
                    val_dataset,
                    tb_logger,
                    opts
                )

                # Track epoch time
                epoch_duration = time.time() - epoch_start_time
                epoch_times.append(epoch_duration)

                validation_costs.append(val_cost.item())
                with open(val_cost_csv_path, 'a') as f:
                    f.write(f'{epoch},{val_cost.item()}\n')
                
                # NEW: Track initial validation cost
                if initial_val_cost is None:
                    initial_val_cost = val_cost.item()
                
                # NEW: Track validation cost at specific epochs
                relative_epoch = epoch - opts.epoch_start
                if relative_epoch == 10:
                    epoch_snapshots['epoch10_val_cost'] = val_cost.item()
                elif relative_epoch == 25:
                    epoch_snapshots['epoch25_val_cost'] = val_cost.item()
                elif relative_epoch == 50:
                    epoch_snapshots['epoch50_val_cost'] = val_cost.item()
                
                # Track best epoch
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    best_epoch = epoch
                    
        except Exception as e:
            experiment_status = 'failed'
            issues_encountered.append(str(e))
            print(f"Training failed with error: {e}")

    # Save the final best model (the baseline's best model, used for final evaluation)
    try:
        inner_baseline = baseline.baseline if isinstance(baseline, WarmupBaseline) else baseline
        if hasattr(inner_baseline, 'model') and inner_baseline.model is not None:
            best_model_path = os.path.join(opts.save_dir, 'best_model.pt')
            torch.save(
                {
                    'model': get_inner_model(inner_baseline.model).state_dict(),
                    'epoch': inner_baseline.epoch,
                    'graph_size': opts.graph_size,
                    'n_cities': opts.n_cities,
                },
                best_model_path
            )
            print('Final best model (baseline epoch {}) saved to {}'.format(inner_baseline.epoch, best_model_path))
    except Exception as e:
        print('Could not save final best model: {}'.format(e))

    # Calculate total training time
    total_training_time = (time.time() - training_start_time) / 60  # Convert to minutes
    
    # Get final validation cost
    final_val_cost = validation_costs[-1] if validation_costs else None
    
    # NEW: Calculate convergence metrics
    convergence_epoch = None
    convergence_speed = None
    if validation_costs and best_val_cost:
        threshold = best_val_cost * 1.05  # Within 5% of best
        for i, cost in enumerate(validation_costs):
            if cost <= threshold:
                convergence_epoch = opts.epoch_start + i
                convergence_speed = i + 1  # Number of epochs to converge
                break
    
    # Run final evaluation to get test metrics
    try:
        # ADD: Check if baseline and model exist
        if not hasattr(baseline, 'baseline') or baseline.baseline is None:
            raise ValueError("baseline.baseline is None")
        if not hasattr(baseline.baseline, 'model') or baseline.baseline.model is None:
            raise ValueError("baseline.baseline.model is None")
        if val_dataset is None:
            raise ValueError("val_dataset is None")
        
        # CRITICAL: Use the baseline's model (best model found during training), not the current model
        model2 = baseline.baseline.model
        
        # CRITICAL: Set ALL time slicing parameters on the baseline's model before evaluation
        if getattr(opts, 'use_time_slicing', False):
            # Ensure the baseline's best model has the correct time slicing configuration
            model2.use_time_slicing = opts.use_time_slicing
            model2.window_size_W = opts.window_size_W
            model2.refresh_strategy = getattr(opts, 'refresh_strategy', 'one_time')
            model2.refresh_interval = getattr(opts, 'refresh_interval', 0.5)
            model2.buffer_k_moves = getattr(opts, 'buffer_k_moves', 2)
            
            # Debug output to verify what we're using
            print(f"\n[Test Evaluation] Using BASELINE'S BEST model with time slicing:")
            print(f"  - use_time_slicing: {model2.use_time_slicing}")
            print(f"  - window_size_W: {model2.window_size_W}")
            print(f"  - refresh_strategy: {model2.refresh_strategy}")
            print(f"  - start_time_bin: {getattr(opts, 'start_time_bin', None)}")
            print(f"  - refresh_interval: {model2.refresh_interval}")
            print(f"  - buffer_k_moves: {model2.buffer_k_moves}\n")
        else:
            # Explicitly disable time slicing if not enabled
            model2.use_time_slicing = False
            print(f"\n[Test Evaluation] Using BASELINE'S BEST model WITHOUT time slicing\n")
        
        print('\n' + '='*50)
        if hasattr(opts, 'use_beam_search_eval') and opts.use_beam_search_eval:
            beam_width = getattr(opts, 'beam_width', 5)
            print(f'[FINAL TEST] Running with BEAM SEARCH (width={beam_width})')
        else:
            print('[FINAL TEST] Running with GREEDY')
        print('='*50 + '\n')
        
        ans, test_costs = roll(mat, model2, val_dataset, opts)

        # Convert to float to avoid Long tensor issues
        test_costs = test_costs.float()
        print(f"Test costs type: {test_costs.dtype}")

        # Convert to minutes (multiply by 1440 as in your original code)
        test_costs_minutes = test_costs * 1440
        
        # Calculate test statistics
        test_mean_cost = torch.mean(test_costs_minutes).item()
        test_std_cost = torch.std(test_costs_minutes).item()
        test_95th_percentile = torch.quantile(test_costs_minutes, 0.95).item()
        test_min_cost = torch.min(test_costs_minutes).item()
        test_max_cost = torch.max(test_costs_minutes).item()
        
        print('Final Avg cost:', test_mean_cost)
        np.savetxt('answer.txt', ans.numpy(), fmt='%d')
        np.savetxt('costs.txt', test_costs.numpy(), fmt='%.6f')
        
    except Exception as e:
        experiment_status = 'partial_failure'
        #issues_encountered.append(f"Final evaluation failed: {str(e)}")
        error_msg = f"Final evaluation failed: {str(e)}"
        issues_encountered.append(error_msg)
        print(f"[ERROR] {error_msg}")
        
        # Set default values
        test_mean_cost = None
        test_std_cost = None
        test_95th_percentile = None
        test_min_cost = None
        test_max_cost = None
    
    # Prepare config dictionary for logging
    config = vars(opts)
    
    # NEW: Calculate total samples seen
    total_samples_seen = opts.batch_size * (opts.epoch_size // opts.batch_size) * opts.n_epochs
    
    # Determine evaluation method used
    eval_method = "greedy"
    eval_details = "Greedy"
    if hasattr(opts, 'use_beam_search_eval') and opts.use_beam_search_eval:
        beam_width = getattr(opts, 'beam_width', 5)
        eval_method = "beam_search"
        eval_details = f"Beam Search (width={beam_width})"

    # Prepare results dictionary with enhanced metrics
    results = {
        'status': experiment_status,
        'final_val_cost': final_val_cost,
        'initial_val_cost': initial_val_cost,
        'test_mean_cost': test_mean_cost,
        'test_std_cost': test_std_cost,
        'test_95th_percentile': test_95th_percentile,
        'test_min_cost': test_min_cost,
        'test_max_cost': test_max_cost,
        'best_epoch': best_epoch,
        'convergence_epoch': convergence_epoch,
        'convergence_speed': convergence_speed,
        'total_training_time': total_training_time,
        'time_per_epoch': np.mean(epoch_times) / 60 if epoch_times else None,
        'total_samples_seen': total_samples_seen,
        'evaluation_method': eval_method,
        'beam_width': getattr(opts, 'beam_width', None) if eval_method == "beam_search" else None,  # Add this line
        
        # Transfer learning info
        **transfer_info,
        
        # Epoch snapshots
        **epoch_snapshots
    }
    
    # Create notes
    method_str = f"{transfer_info['training_method']} (Stage {transfer_info['stage_number']})"
    notes = f"{method_str}. Best epoch: {best_epoch}. "
    if transfer_info['transfer_source_path']:
        notes += f"Transferred {transfer_info['layers_transferred']} layers from {transfer_info['transfer_source_graph_size']}-node model. "
    if experiment_status != 'completed':
        notes += "Training had issues."
    
    issues = "; ".join(issues_encountered) if issues_encountered else ""
    
    # Log the experiment
    experiment_id = tracker.log_experiment(
        config=config,
        results=results,
        notes=notes,
        issues=issues
    )
    
    print(f"\nExperiment logged with ID: {experiment_id}")
    print(f"Results summary:")
    print(f"  - Training method: {transfer_info['training_method']} (Stage {transfer_info['stage_number']})")
    print(f"  - Evaluation method: {eval_details}")
    print(f"  - Final validation cost: {final_val_cost:.4f}" if final_val_cost else "  - Final validation cost: N/A")
    print(f"  - Test mean cost: {test_mean_cost:.4f} ± {test_std_cost:.4f}" if test_mean_cost else "  - Test mean cost: N/A")
    print(f"  - Total training time: {total_training_time:.1f} minutes")
    print(f"  - Convergence speed: {convergence_speed} epochs" if convergence_speed else "")
    print(f"  - Status: {experiment_status}")


if __name__ == "__main__":
    run(get_options())
