import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from experiment_tracker import ExperimentTracker

def analyze_time_slicing_experiments(csv_file='experiment_log.csv'):
    """Analyze time slicing experiments using the experiment tracker"""
    
    # Load experiments
    tracker = ExperimentTracker(csv_file)
    all_exps = tracker.load_experiments()
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_exps)
    
    # Filter time slicing experiments
    ts_experiments = df[df['use_time_slicing'] == 'True']
    baseline_experiments = df[df['use_time_slicing'] != 'True']
    
    print("\n" + "="*80)
    print("TIME SLICING EXPERIMENT ANALYSIS")
    print("="*80)
    
    # 1. Window Size Comparison
    if len(ts_experiments) > 0:
        print("\n--- Window Size Comparison ---")
        window_comparison = ts_experiments.groupby('window_size_W').agg({
            'test_mean_cost': 'mean',
            'test_std_cost': 'mean',
            'total_training_time': 'mean'
        })
        print(window_comparison)
        
        # 2. Refresh Strategy Comparison
        print("\n--- Refresh Strategy Comparison ---")
        strategy_comparison = ts_experiments.groupby('refresh_strategy').agg({
            'test_mean_cost': 'mean',
            'test_std_cost': 'mean',
            'total_training_time': 'mean'
        })
        print(strategy_comparison)
        
        # 3. Start Time Comparison
        print("\n--- Start Time Comparison ---")
        start_time_comparison = ts_experiments.groupby('start_time_bin').agg({
            'test_mean_cost': 'mean',
            'test_std_cost': 'mean'
        })
        print(start_time_comparison)
        
        # 4. Compare with Baseline
        if len(baseline_experiments) > 0:
            baseline_cost = float(baseline_experiments['test_mean_cost'].iloc[0]) if baseline_experiments['test_mean_cost'].notna().any() else None
            if baseline_cost:
                print("\n--- Comparison with Baseline (No Time Slicing) ---")
                print(f"Baseline cost: {baseline_cost:.4f}")
                best_ts = ts_experiments.loc[ts_experiments['test_mean_cost'].astype(float).idxmin()]
                best_cost = float(best_ts['test_mean_cost'])
                improvement = ((baseline_cost - best_cost) / baseline_cost) * 100
                print(f"Best time slicing cost: {best_cost:.4f}")
                print(f"Improvement: {improvement:+.2f}%")
                print(f"Best config: W={best_ts['window_size_W']}, Strategy={best_ts['refresh_strategy']}, Start={best_ts['start_time_bin']}")
    
    # Create visualizations
    create_visualizations(ts_experiments, baseline_experiments)
    
    return df, ts_experiments, baseline_experiments

def create_visualizations(ts_experiments, baseline_experiments):
    """Create comparison plots"""
    
    if len(ts_experiments) == 0:
        print("\nNo time slicing experiments found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Window Size vs Cost
    if 'window_size_W' in ts_experiments.columns:
        window_data = ts_experiments.groupby('window_size_W')['test_mean_cost'].mean()
        axes[0, 0].bar(window_data.index.astype(str), window_data.values)
        axes[0, 0].set_title('Mean Cost vs Window Size')
        axes[0, 0].set_xlabel('Window Size (W)')
        axes[0, 0].set_ylabel('Mean Cost (minutes)')
    
    # 2. Refresh Strategy vs Cost
    if 'refresh_strategy' in ts_experiments.columns:
        strategy_data = ts_experiments.groupby('refresh_strategy')['test_mean_cost'].mean()
        axes[0, 1].bar(strategy_data.index, strategy_data.values)
        axes[0, 1].set_title('Mean Cost vs Refresh Strategy')
        axes[0, 1].set_xlabel('Refresh Strategy')
        axes[0, 1].set_ylabel('Mean Cost (minutes)')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Training Time Comparison
    if len(baseline_experiments) > 0 and 'total_training_time' in ts_experiments.columns:
        baseline_time = float(baseline_experiments['total_training_time'].iloc[0]) if baseline_experiments['total_training_time'].notna().any() else None
        ts_time = ts_experiments['total_training_time'].mean()
        
        if baseline_time:
            axes[1, 0].bar(['Baseline', 'Time Slicing'], [baseline_time, ts_time])
            axes[1, 0].set_title('Training Time Comparison')
            axes[1, 0].set_ylabel('Time (minutes)')
    
    # 4. Start Time vs Cost
    if 'start_time_bin' in ts_experiments.columns:
        start_data = ts_experiments.groupby('start_time_bin')['test_mean_cost'].mean()
        axes[1, 1].bar(start_data.index.astype(str), start_data.values)
        axes[1, 1].set_title('Mean Cost vs Start Time')
        axes[1, 1].set_xlabel('Start Time Bin')
        axes[1, 1].set_ylabel('Mean Cost (minutes)')
    
    plt.tight_layout()
    plt.savefig('time_slicing_analysis.png', dpi=300)
    print("\nVisualizations saved to time_slicing_analysis.png")

if __name__ == '__main__':
    df, ts_exps, baseline = analyze_time_slicing_experiments()
    df.to_csv('experiment_analysis.csv', index=False)
    print("\nFull analysis saved to experiment_analysis.csv")