import torch
import numpy as np
from m1.train import *
from m1.transformer import AttentionModel

def evaluate_comprehensive(mat, model, test_dataset, opts, model_name="Model"):
    """Comprehensive evaluation with all three metrics"""
    
    # Get costs for all test instances
    costs = rollout(mat, model, test_dataset, opts)
    costs_minutes = costs.numpy()  # Convert to minutes
    
    # 1. Mean Tour Time
    mean_tour_time = np.mean(costs_minutes)
    
    # 2. 95th Percentile Tour Time  
    percentile_95_time = np.percentile(costs_minutes, 95)
    
    return mean_tour_time, percentile_95_time, costs_minutes

def run_heuristic_baselines(mat, test_dataset):
    """Run heuristic baselines for optimality gap calculation"""
    
    heuristic_costs = {}
    
    # Nearest Neighbor (NN) heuristic
    nn_costs = run_nearest_neighbor_heuristic(test_dataset)
    heuristic_costs['NN'] = nn_costs
    
    # NN+1 (Nearest Neighbor + 1-step improvement)  
    nn1_costs = run_nn_plus_one_heuristic(test_dataset)
    heuristic_costs['NN1'] = nn1_costs
    
    # Rank-NN heuristic
    rank_nn_costs = run_ranking_nn_heuristic(test_dataset)
    heuristic_costs['Rank-NN'] = rank_nn_costs
    
    return heuristic_costs

def calculate_optimality_gap(model_costs, heuristic_costs):
    """Calculate optimality gap relative to best heuristic"""
    
    # Find best heuristic performance for each instance
    best_heuristic_costs = []
    
    for i in range(len(model_costs)):
        instance_heuristic_costs = [
            heuristic_costs['NN'][i],
            heuristic_costs['NN1'][i], 
            heuristic_costs['Rank-NN'][i]
        ]
        best_cost = min(instance_heuristic_costs)
        best_heuristic_costs.append(best_cost)
    
    best_heuristic_costs = np.array(best_heuristic_costs)
    
    # Calculate percentage gap for each instance
    gaps = ((model_costs - best_heuristic_costs) / best_heuristic_costs) * 100
    
    # Return mean gap (negative = better than heuristics)
    return np.mean(gaps)

def full_evaluation():
    """Complete evaluation including all three metrics"""
    
    # Setup
    ci = Cities()
    mat = DistanceMatrix(ci, load_dir='m1/data.csv', max_time_step=12)
    test_dataset = TSPDataset(ci, size=19, num_samples=1000)
    
    # Load models
    baseline_model = torch.load('path/to/baseline_m1.pt')
    step_mlp_model = torch.load('path/to/step_mlp_m1.pt') 
    
    # Evaluate models
    print("=== Model Evaluation ===")
    
    # Baseline M1
    baseline_mean, baseline_95th, baseline_costs = evaluate_comprehensive(
        mat, baseline_model, test_dataset, opts, "Baseline M1"
    )
    
    # Step-MLP Enhanced M1  
    step_mlp_mean, step_mlp_95th, step_mlp_costs = evaluate_comprehensive(
        mat, step_mlp_model, test_dataset, opts, "Step-MLP M1"
    )
    
    # Run heuristic baselines
    print("Running heuristic baselines...")
    heuristic_costs = run_heuristic_baselines(mat, test_dataset)
    
    # Calculate optimality gaps
    baseline_gap = calculate_optimality_gap(baseline_costs, heuristic_costs)
    step_mlp_gap = calculate_optimality_gap(step_mlp_costs, heuristic_costs)
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    print(f"{'Metric':<25} {'Baseline M1':<15} {'Step-MLP M1':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # 1. Mean Tour Time  
    print(f"{'Mean Tour Time (min)':<25} {baseline_mean:<15.2f} {step_mlp_mean:<15.2f} {((baseline_mean - step_mlp_mean)/baseline_mean*100):>+6.2f}%")
    
    # 2. 95th Percentile Tour Time
    print(f"{'95th Percentile (min)':<25} {baseline_95th:<15.2f} {step_mlp_95th:<15.2f} {((baseline_95th - step_mlp_95th)/baseline_95th*100):>+6.2f}%")
    
    # 3. Optimality Gap (vs best heuristic)
    print(f"{'Optimality Gap (%)':<25} {baseline_gap:<15.2f} {step_mlp_gap:<15.2f} {(step_mlp_gap - baseline_gap):>+6.2f}pp")
    
    print("\nNotes:")
    print("- Negative optimality gap = BETTER than best heuristic")  
    print("- 'pp' = percentage points difference")
    print("- '+' improvement = your method is better")

# Heuristic implementation stubs (you need to implement these)
def run_nearest_neighbor_heuristic(test_dataset):
    """Implement Nearest Neighbor heuristic for all test instances"""
    nn_costs = []
    for instance in test_dataset:
        cost = nearest_neighbor_tour(instance)
        nn_costs.append(cost)
    return nn_costs

def nearest_neighbor_tour(instance):
    """
    instance: one test case, must include:
        - node coordinates or indices
        - time-dependent cost matrix (e.g. 2D or 3D, [i, j, t])
    Returns: total tour time for NN heuristic
    """
    n = len(instance["nodes"])  # or however your structure works
    visited = [False] * n
    tour = [0]  # start from depot (index 0)
    current = 0
    current_time = 0.0
    visited[current] = True

    for step in range(1, n):
        # Find best next node given current time
        next_node = None
        min_time = float("inf")
        for i in range(n):
            if not visited[i]:
                time_cost = instance["cost_matrix"][current][i][int(current_time) % 12]  # shape as needed
                if time_cost < min_time:
                    min_time = time_cost
                    next_node = i
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        current_time += min_time  # update as needed for time-dependent constraints

    # Return to depot
    tour.append(0)
    final_time = instance["cost_matrix"][current][0][int(current_time) % 12]
    total_cost = current_time + final_time
    return total_cost

def run_nn_plus_one_heuristic(test_dataset):
    """Implement NN+1 (NN + one-step improvement) heuristic"""  
    # TODO: Implement NN1 heuristic
    pass

def run_ranking_nn_heuristic(test_dataset):
    """Implement Ranking-based NN heuristic"""
    # TODO: Implement Rank-NN heuristic
    pass

if __name__ == "__main__":
    full_evaluation()
