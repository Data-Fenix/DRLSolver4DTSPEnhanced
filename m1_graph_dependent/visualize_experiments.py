import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiments(log_file='experiment_log.csv'):
    """Load experiment log"""
    return pd.read_csv(log_file)

def plot_heuristic_comparison(df):
    """Compare performance across heuristics"""
    # Filter for cost-aware gating experiments
    gating_exps = df[df['use_cost_aware_gating'] == True]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Mean cost by heuristic
    gating_exps.groupby('heuristic_type')['test_mean_cost'].mean().plot(
        kind='bar', ax=axes[0], color='skyblue'
    )
    axes[0].set_title('Mean Test Cost by Heuristic')
    axes[0].set_ylabel('Cost')
    axes[0].set_xlabel('Heuristic Type')
    
    # Plot 2: Training time
    gating_exps.groupby('heuristic_type')['total_training_time'].mean().plot(
        kind='bar', ax=axes[1], color='lightcoral'
    )
    axes[1].set_title('Training Time by Heuristic')
    axes[1].set_ylabel('Time (minutes)')
    
    # Plot 3: 95th percentile (robustness)
    gating_exps.groupby('heuristic_type')['test_95th_percentile'].mean().plot(
        kind='bar', ax=axes[2], color='lightgreen'
    )
    axes[2].set_title('95th Percentile Cost by Heuristic')
    axes[2].set_ylabel('Cost')
    
    plt.tight_layout()
    plt.savefig('heuristic_comparison.png', dpi=300)
    plt.show()

def plot_lambda_sensitivity(df, heuristic='linear_time'):
    """Plot lambda sensitivity for a specific heuristic"""
    data = df[(df['heuristic_type'] == heuristic) & (df['use_cost_aware_gating'] == True)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['lambda_heuristic'], data['test_mean_cost'], marker='o', linewidth=2)
    plt.xlabel('Lambda (Heuristic Weight)')
    plt.ylabel('Test Mean Cost')
    plt.title(f'Lambda Sensitivity Analysis - {heuristic}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'lambda_sensitivity_{heuristic}.png', dpi=300)
    plt.show()

def create_summary_table(df):
    """Create summary comparison table"""
    summary = df.groupby(['use_step_mlp', 'use_cost_aware_gating', 'heuristic_type']).agg({
        'test_mean_cost': ['mean', 'std'],
        'test_95th_percentile': 'mean',
        'total_training_time': 'mean'
    }).round(4)
    
    print("\n=== Experiment Summary ===")
    print(summary)
    summary.to_csv('experiment_summary.csv')
    return summary

if __name__ == '__main__':
    df = load_experiments()
    
    # Generate visualizations
    plot_heuristic_comparison(df)
    plot_lambda_sensitivity(df, heuristic='linear_time')
    
    # Generate summary
    create_summary_table(df)
