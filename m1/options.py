import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving the Travelling Salesman Problem with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', type=int, default=19, help="The size of the problem graph")
    parser.add_argument('--n_cities', type=int, default=100,
                    help='Total number of cities in the pool (default 100). graph_size instances are sampled from this pool.')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=128000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')


    # Model
    parser.add_argument('--model', default='attention', help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=2,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=0.995, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=1000, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--val_seed', type=int, default=0, help='Fixed random seed for generating the validation/test dataset (same across all training seeds)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default=None,
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    
    # Beam Search options (for evaluation)
    parser.add_argument('--use_beam_search_baseline', action='store_true',
                       help='Use beam search for rollout baseline (default: greedy)')
    parser.add_argument('--beam_width', type=int, default=5,
                       help='Beam width for beam search (default: 5)')
    parser.add_argument('--use_beam_search_eval', action='store_true',
                       help='Use beam search for evaluation/validation (default: greedy)')

    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    # NEW: Cost-Aware Gating parameters    # NEW: Cost-Aware Gating parameters
    parser.add_argument('--use_cost_aware_gating', action='store_true',
                       help='Enable cost-aware gating for heuristic blending')
    parser.add_argument('--heuristic_type', type=str, default='linear_time',
                       choices=['linear_time', 'nearest_neighbor', 'nn_plus_one', 'nnr'],
                       help='Type of heuristic to use for cost-aware gating')
    parser.add_argument('--lambda_heuristic', type=float, default=1.0,
                       help='Weight for heuristic blending in cost-aware gating')
    parser.add_argument('--lambda_heuristic_learnable', action='store_true', default=False,
                       help='Make lambda_heuristic a learnable parameter (default: False, uses fixed value)')
    parser.add_argument('--use_nonlinear_transform', action='store_true',
                       help='Enable nonlinear transformation for heuristic blending')
    parser.add_argument('--transform_type', type=str, default='piecewise',
                       choices=['piecewise', 'exponential'],
                       help='Type of nonlinear transformation for heuristic blending')
    # NEW: Step-MLP parameters
    parser.add_argument('--use_step_mlp', action='store_true', 
                       help='Enable Step-MLP enhancement')
    parser.add_argument('--step_mlp_dim', type=int, default=64,
                       help='Hidden dimension for Step-MLP')

    # Step-MLP Feature Flags
    parser.add_argument('--step_features_v1', action='store_true',
                       help='Preset: step_ratio + last_3_nodes + visited_mean')
    parser.add_argument('--step_features_v2', action='store_true',
                       help='Preset: step_ratio + sin_cos_time + depot_distance + tour_length + unvisited_mean + mean_dist_unvisited')
    parser.add_argument('--step_features_v1_light', action='store_true',
                       help='Preset: step_ratio + last_3_nodes (NO visited_mean)')
    parser.add_argument('--step_features_v2_light', action='store_true',
                       help='Preset: step_ratio + sin_cos_time + depot_distance + tour_length + mean_dist_unvisited (NO unvisited_mean)')
    parser.add_argument('--step_features_minimal', action='store_true',  # NEW
                       help='Preset: step_ratio + visited_mean only (minimal with embedding)')
    parser.add_argument('--step_features_step_ratio_only', action='store_true',
                   help='Preset: step_ratio only (1 dim, minimal baseline)')
    parser.add_argument('--step_features_v2_plus_visited', action='store_true',
                   help='Preset: v2 + visited_mean (262 dim, both embeddings)')
    parser.add_argument('--step_features_v2_light_plus_visited', action='store_true',
                   help='Preset: v2_light + visited_mean (134 dim, V2 features with visited_mean)')

    # Individual feature flags (for fine-grained control)
    parser.add_argument('--use_step_ratio', action='store_true',
                       help='Use step ratio feature (individual flag)')
    parser.add_argument('--use_last_3_nodes', action='store_true',
                       help='Use last 3 visited nodes feature (individual flag)')
    parser.add_argument('--use_visited_mean', action='store_true',
                       help='Use visited mean embedding (expensive, individual flag)')
    parser.add_argument('--use_unvisited_mean', action='store_true',
                       help='Use unvisited mean embedding (expensive, individual flag)')
    parser.add_argument('--use_sin_cos_time', action='store_true',
                       help='Use sin/cos time encoding (individual flag)')               
    parser.add_argument('--use_linear_time', action='store_true',                   help='Use linear time encoding instead of sin/cos (1 dim vs 2 dim, individual flag)')
    parser.add_argument('--use_depot_distance', action='store_true',
                       help='Use depot distance feature (individual flag)')
    parser.add_argument('--use_tour_length', action='store_true',
                       help='Use tour length feature (individual flag)')
    parser.add_argument('--use_mean_dist_unvisited', action='store_true',
                       help='Use mean distance to unvisited nodes (individual flag)')

    # NEW: Temperature MLP control
    parser.add_argument('--use_temp_mlp', action='store_true',
                       help='Enable temperature adjustment MLP (only used when --use_step_mlp is enabled)')
    # Add these lines after the existing arguments (around line 130-140)

    # Time Slicing parameters
    parser.add_argument('--use_time_slicing', action='store_true',
                    help='Enable time slicing with windowed encoding')
    parser.add_argument('--window_size_W', type=int, default=12,
                    help='Time window size in bins (default 12 = full day). '
                         'Use -1 for forward window (all bins from start_time_bin to end of day). '
                         'Examples: 3=3 bins, 6=6 bins, 12=full day, -1=forward window')
    parser.add_argument('--start_time_bin', type=int, default=0,
                    help='Starting time bin for the tour (0-11, 0=midnight, 3=6am, 6=noon)')

    # Safe Refresh parameters
    parser.add_argument('--refresh_strategy', type=str, default='one_time',
                   choices=['buffer', 'one_time', 'periodic', 'combined', 'none'],
                   help='Strategy for encoder refresh when window expires: '
                        '"one_time"=refresh when boundary reached (default), '
                        '"buffer"=refresh before boundary using k-moves lookahead, '
                        '"periodic"=refresh at fixed time intervals, '
                        '"combined"=use all strategies, '
                        '"none"=no refresh')
    parser.add_argument('--refresh_interval', type=float, default=0.5,
                   help='Refresh interval for periodic strategy in normalized time / Simulate k greedy moves and check if estimated time exceeds window'
                        '(default 0.5 = 6 hours = 3 time bin). '
                        'Example: 0.167 = 2 hours, 0.333 = 4 hours, 0.5 = 6 hours')
    parser.add_argument('--buffer_k_moves', type=int, default=2,
                   help='Number of moves to look ahead for buffer rule (default 2). '
                        'Higher values = more conservative (refresh earlier)')

    # Transfer Learning Experiment Tracking
    parser.add_argument('--training_method', type=str, default='direct',
                       choices=['direct', '2-stage', '3-stage'],
                       help='Training method for experiment tracking')
    parser.add_argument('--stage_number', type=int, default=1,
                       help='Current stage number for progressive training')

                       

    # Model Comparison arguments
    parser.add_argument('--compare_models', nargs='+', default=None,
                       help='List of model checkpoint paths to compare (e.g., path1.pt path2.pt)')
    parser.add_argument('--compare_names', nargs='+', default=None,
                       help='List of model names for comparison (e.g., Baseline StepMLP). '
                            'Must have same length as --compare_models')
    parser.add_argument('--no_heuristics', action='store_true',
                       help='Disable running heuristic baselines in model comparison')

    # Decoder MLP parameters(Option A: Post-attention)
    parser.add_argument('--use_decoder_mlp', action='store_true',
                   help='Enable decoder MLP (Option 1: post-attention MLP)')
    parser.add_argument('--decoder_mlp_hidden', type=int, default=512,
                   help='Hidden dimension for decoder MLP (default: 512, matches encoder)')

    # Decoder MLP parameters(Option C: Pre-attention)
    parser.add_argument('--use_decoder_mlp_pre', action='store_true',
                   help='Enable decoder MLP pre-attention (Option C: pre-attention MLP)')
    parser.add_argument('--decoder_mlp_pre_hidden', type=int, default=512,
                   help='Hidden dimension for decoder MLP pre-attention (default: 512, matches encoder)')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
