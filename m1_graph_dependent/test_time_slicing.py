import torch
from transformer import AttentionModel
from train import DistanceMatrix, Cities

# Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ci = Cities()
mat = DistanceMatrix(ci, load_dir='data.csv', max_time_step=12)

# Create model with time slicing
model_windowed = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=2,
    input_size=20,  # 19 customers + 1 depot
    max_t=12,
    use_time_slicing=True,
    window_size_W=3,  # 6-hour window
    n_cities=100,
    use_step_mlp=False,
    use_cost_aware_gating=False
).to(device)

# Create model without time slicing
model_full = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=2,
    input_size=20,  # 19 customers + 1 depot
    max_t=12,
    use_time_slicing=False,
    n_cities=100,
    use_step_mlp=False,
    use_cost_aware_gating=False
).to(device)

# ADD THIS: Set decode type before using models
model_windowed.set_decode_type("greedy")  # Use greedy decoding for testing
model_full.set_decode_type("greedy")      # Use greedy decoding for testing

# Create dummy input
batch_size = 4
graph_size = 20  # Include depot
node_dim = 100
dummy_input = torch.randn(batch_size, graph_size, node_dim).to(device)

print("="*60)
print("Testing Time Slicing Implementation")
print("="*60)

# Test windowed model
print("\n--- Testing Windowed Model (W=3) ---")
model_windowed.eval()
with torch.no_grad():
    cost_windowed, ll_windowed, _ = model_windowed(mat, dummy_input, start_time=0.25)
print(f"Cost shape: {cost_windowed.shape}, LL shape: {ll_windowed.shape}")
print(f"Sample costs: {cost_windowed[:3]}")

# Test full model
print("\n--- Testing Full Model (W=12) ---")
model_full.eval()
with torch.no_grad():
    cost_full, ll_full, _ = model_full(mat, dummy_input)
print(f"Cost shape: {cost_full.shape}, LL shape: {ll_full.shape}")
print(f"Sample costs: {cost_full[:3]}")

# Memory comparison
print("\n--- Memory Comparison ---")
windowed_params = sum(p.numel() for p in model_windowed.parameters())
full_params = sum(p.numel() for p in model_full.parameters())
print(f"Windowed model parameters: {windowed_params:,}")
print(f"Full model parameters: {full_params:,}")
print(f"Difference: {full_params - windowed_params:,} parameters")

print("\n="*60)
print("Test completed successfully!")
print("="*60)