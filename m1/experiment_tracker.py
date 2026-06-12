import csv
import os
from datetime import datetime
import json
import subprocess

class ExperimentTracker:
    """Track experiments automatically and save to CSV with enhanced transfer learning metrics"""
    
    def __init__(self, log_file='experiment_log.csv'):
        self.log_file = log_file
        self.fieldnames = [
            # Metadata
            'experiment_id', 'experiment_name', 'date', 'timestamp', 'git_commit', 'status',
            
            # Model Config
            'model_type', 'use_step_mlp', 'step_mlp_dim', 'step_features_used',
            'use_cost_aware_gating', 'heuristic_type', 'lambda_heuristic', 
            'embedding_dim', 'hidden_dim', 'n_encode_layers',
            
            # Time Slicing Config (ADD THIS SECTION)
            'use_time_slicing', 'window_size_W', 'start_time_bin', 
            'refresh_strategy', 'refresh_interval', 'buffer_k_moves',
            
            # Training Hyperparams
            'graph_size', 'n_epochs', 'batch_size', 'learning_rate', 'lr_decay', 'baseline', 'seed',
            
            # Transfer Learning Info
            'training_method', 'stage_number', 'transfer_source_path', 'transfer_source_graph_size',
            'layers_transferred', 'layers_initialized',
            
            # Performance Metrics
            'final_train_cost', 'final_val_cost', 'test_mean_cost', 'test_std_cost', 
            'test_95th_percentile', 'test_min_cost', 'test_max_cost', 'optimality_gap',
            
            # Stage Performance (on target size)
            'stage1_test_cost', 'stage2_test_cost', 'stage3_test_cost',
            
            # Learning Dynamics
            'best_epoch', 'convergence_epoch', 'epoch10_val_cost', 'epoch25_val_cost', 
            'epoch50_val_cost', 'convergence_speed', 'improvement_rate',
            
            # Training Efficiency
            'total_training_time', 'time_per_epoch', 'total_samples_seen', 
            'samples_per_improvement', 'final_grad_norm',
            
            # Robustness
            'coefficient_of_variation', 'performance_stability',
            
            # Additional
            'notes', 'issues', 'output_path'
        ]
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def get_git_commit(self):
        """Get current git commit hash"""
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except:
            return 'unknown'
    
    def _get_step_features_string(self, config):
        """Build string describing which step features are enabled"""
        features = []
        dim = 0
        
        # Check presets first
        if config.get('step_features_v1'):
            features.append('v1')
            dim = 1 + 3 + config.get('embedding_dim', 128)  # step_ratio + last_3 + visited_mean
        elif config.get('step_features_v2'):
            features.append('v2')
            dim = 1 + 2 + 1 + 1 + config.get('embedding_dim', 128) + 1  # step_ratio + sin_cos + depot + tour + unvisited + mean_dist
        elif config.get('step_features_v1_light'):
            features.append('v1_light')
            dim = 1 + 3  # step_ratio + last_3
        elif config.get('step_features_v2_light'):
            features.append('v2_light')
            dim = 1 + 2 + 1 + 1 + 1  # step_ratio + sin_cos + depot + tour + mean_dist
        elif config.get('step_features_minimal'):
            features.append('minimal')
            dim = 1 + config.get('embedding_dim', 128)  # step_ratio + visited_mean
        else:
            # Individual flags
            if config.get('use_step_ratio'):
                features.append('step_ratio')
                dim += 1
            if config.get('use_last_3_nodes'):
                features.append('last_3_nodes')
                dim += 3
            if config.get('use_visited_mean'):
                features.append('visited_mean')
                dim += config.get('embedding_dim', 128)
            if config.get('use_unvisited_mean'):
                features.append('unvisited_mean')
                dim += config.get('embedding_dim', 128)
            if config.get('use_sin_cos_time'):
                features.append('sin_cos_time')
                dim += 2
            if config.get('use_linear_time'):
                features.append('linear_time')
                dim += 1
            if config.get('use_depot_distance'):
                features.append('depot_distance')
                dim += 1
            if config.get('use_tour_length'):
                features.append('tour_length')
                dim += 1
            if config.get('use_mean_dist_unvisited'):
                features.append('mean_dist_unvisited')
                dim += 1
        
        feature_str = '+'.join(features) if features else 'none'
        return f"{feature_str}({dim}dim)"  # Return with dimension info
    
    def log_experiment(self, config, results, notes='', issues=''):
        """Log a single experiment with enhanced metrics"""
        
        # Generate experiment ID
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            exp_count = sum(1 for _ in reader)
        exp_id = f"exp_{exp_count + 1:03d}"
        
        # Calculate derived metrics
        test_mean = results.get('test_mean_cost', None)
        test_std = results.get('test_std_cost', None)
        test_95th = results.get('test_95th_percentile', None)
        test_min = results.get('test_min_cost', None)
        
        # Coefficient of variation
        cv = (test_std / test_mean) if (test_mean and test_std) else None
        
        # Performance stability (1 - normalized range)
        stability = None
        if test_95th and test_min and test_mean and test_mean > 0:
            normalized_range = (test_95th - test_min) / test_mean
            stability = max(0, 1 - normalized_range)
        
        # Improvement rate (cost improvement per minute)
        improvement_rate = None
        if results.get('initial_val_cost') and results.get('final_val_cost') and results.get('total_training_time'):
            cost_improvement = results['initial_val_cost'] - results['final_val_cost']
            improvement_rate = cost_improvement / results['total_training_time']
        
        # Samples per improvement
        samples_per_improvement = None
        if results.get('total_samples_seen') and results.get('initial_val_cost') and results.get('final_val_cost'):
            cost_improvement = results['initial_val_cost'] - results['final_val_cost']
            if cost_improvement > 0:
                samples_per_improvement = results['total_samples_seen'] / cost_improvement
        
        # Create experiment record
        record = {
            # Metadata
            'experiment_id': exp_id,
            'experiment_name': config.get('run_name', 'unnamed'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().strftime('%Y%m%dT%H%M%S'),
            'git_commit': self.get_git_commit(),
            'status': results.get('status', 'completed'),
            
            # Model Config
            'model_type': config.get('model', 'attention'),
            'use_step_mlp': config.get('use_step_mlp', False),
            'step_mlp_dim': config.get('step_mlp_dim', 0),
            'step_features_used': self._get_step_features_string(config),  # NEW
            'use_cost_aware_gating': config.get('use_cost_aware_gating', False),
            'heuristic_type': config.get('heuristic_type', 'none'),
            'lambda_heuristic': config.get('lambda_heuristic', 0.0),
            'embedding_dim': config.get('embedding_dim', 128),
            'hidden_dim': config.get('hidden_dim', 128),
            'n_encode_layers': config.get('n_encode_layers', 2),

            # Time Slicing Config
            'use_time_slicing': config.get('use_time_slicing', False),
            'window_size_W': config.get('window_size_W', None),
            'start_time_bin': config.get('start_time_bin', None),
            'refresh_strategy': config.get('refresh_strategy', None),
            'refresh_interval': config.get('refresh_interval', None),
            'buffer_k_moves': config.get('buffer_k_moves', None),
            
            # Training Hyperparams
            'graph_size': config.get('graph_size', 19),
            'n_epochs': config.get('n_epochs', 10),
            'batch_size': config.get('batch_size', 512),
            'learning_rate': config.get('lr_model', 0.0001),
            'lr_decay': config.get('lr_decay', 0.995),
            'baseline': config.get('baseline', 'rollout'),
            'seed': config.get('seed', 1234),
            
            # Transfer Learning Info
            'training_method': results.get('training_method', 'direct'),
            'stage_number': results.get('stage_number', 1),
            'transfer_source_path': results.get('transfer_source_path', None),
            'transfer_source_graph_size': results.get('transfer_source_graph_size', None),
            'layers_transferred': results.get('layers_transferred', None),
            'layers_initialized': results.get('layers_initialized', None),
            
            # Performance Metrics
            'final_train_cost': results.get('final_train_cost', None),
            'final_val_cost': results.get('final_val_cost', None),
            'test_mean_cost': test_mean,
            'test_std_cost': test_std,
            'test_95th_percentile': test_95th,
            'test_min_cost': test_min,
            'test_max_cost': results.get('test_max_cost', None),
            'optimality_gap': results.get('optimality_gap', None),
            
            # Stage Performance
            'stage1_test_cost': results.get('stage1_test_cost', None),
            'stage2_test_cost': results.get('stage2_test_cost', None),
            'stage3_test_cost': results.get('stage3_test_cost', None),
            
            # Learning Dynamics
            'best_epoch': results.get('best_epoch', None),
            'convergence_epoch': results.get('convergence_epoch', None),
            'epoch10_val_cost': results.get('epoch10_val_cost', None),
            'epoch25_val_cost': results.get('epoch25_val_cost', None),
            'epoch50_val_cost': results.get('epoch50_val_cost', None),
            'convergence_speed': results.get('convergence_speed', None),
            'improvement_rate': improvement_rate,
            
            # Training Efficiency
            'total_training_time': results.get('total_training_time', None),
            'time_per_epoch': results.get('time_per_epoch', None),
            'total_samples_seen': results.get('total_samples_seen', None),
            'samples_per_improvement': samples_per_improvement,
            'final_grad_norm': results.get('final_grad_norm', None),
            
            # Robustness
            'coefficient_of_variation': cv,
            'performance_stability': stability,
            
            # Additional
            'notes': notes,
            'issues': issues,
            'output_path': config.get('save_dir', '')
        }
        
        # Append to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)
        
        print(f"\n{'='*60}")
        print(f"Logged experiment: {exp_id} - {record['experiment_name']}")
        print(f"Method: {record['training_method']} | Stage: {record['stage_number']}")
        print(f"Step Features: {record['step_features_used']}")  # NEW: Print features used
        print(f"Test Cost: {test_mean:.4f} ± {test_std:.4f}" if test_mean else "Test Cost: N/A")
        print(f"Training Time: {record['total_training_time']:.1f} min" if record['total_training_time'] else "")
        print(f"{'='*60}\n")
        return exp_id
    
    def load_experiments(self):
        """Load all experiments as a list of dictionaries"""
        experiments = []
        with open(self.log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                experiments.append(row)
        return experiments
    
    def get_experiments_by_method(self, method):
        """Filter experiments by training method"""
        all_exps = self.load_experiments()
        return [exp for exp in all_exps if exp.get('training_method') == method]
    
    def compare_methods(self, methods=['direct', '2-stage', '3-stage']):
        """Generate comparison table of different training methods"""
        print(f"\n{'='*80}")
        print("TRAINING METHOD COMPARISON")
        print(f"{'='*80}")
        print(f"{'Method':<15} {'Test Cost':<15} {'Std':<10} {'Time (min)':<12} {'Best Epoch':<12}")
        print(f"{'-'*80}")
        
        for method in methods:
            exps = self.get_experiments_by_method(method)
            if exps:
                # Get the most recent or final stage
                latest = exps[-1]
                cost = float(latest['test_mean_cost']) if latest['test_mean_cost'] else None
                std = float(latest['test_std_cost']) if latest['test_std_cost'] else None
                time = float(latest['total_training_time']) if latest['total_training_time'] else None
                epoch = latest['best_epoch'] if latest['best_epoch'] else 'N/A'
                
                cost_str = f"{cost:.4f}" if cost else "N/A"
                std_str = f"±{std:.4f}" if std else ""
                time_str = f"{time:.1f}" if time else "N/A"
                
                print(f"{method:<15} {cost_str:<15} {std_str:<10} {time_str:<12} {epoch:<12}")
            else:
                print(f"{method:<15} {'Not run yet':<45}")
        
        print(f"{'='*80}\n")