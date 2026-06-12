# test_beam_search.py
import torch
from transformer import AttentionModel
from train import Cities, DistanceMatrix, TSPDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Small test
ci = Cities(n_cities=100)
mat = DistanceMatrix(ci, load_dir='data.csv', max_time_step=12)
model = AttentionModel(
    embedding_dim=128,
    hidden_dim=128,
    n_encode_layers=2,
    n_cities=100,
    max_t=12,
    use_step_mlp=False,  # Add this line to disable step MLP
    input_size=4  # Add this line (required parameter)
).to(device)

# Create small test batch
dataset = TSPDataset(ci, size=19, num_samples=4)
test_batch = dataset.data[:4].to(device)

# Test beam search
model.set_decode_type("beam", beam_width=3)
model.eval()

with torch.no_grad():
    cost, _, pi = model(mat, test_batch, return_pi=True)
    print(f"Beam search test passed!")
    print(f"Cost shape: {cost.shape}, Sequence shape: {pi.shape}")
    print(f"Mean cost: {cost.mean().item():.4f}")