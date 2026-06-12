"""
Generate node dataset files (e.g., node_49.txt) for DTSP evaluation.

Format matches node_19.txt:
  - 1000 rows, each row = depot(0) + graph_size unique city indices from 1-99
  - Space-separated integers

Usage:
  python generate_node_dataset.py --graph_size 49
  python generate_node_dataset.py --graph_size 49 --num_samples 1000 --seed 1234
"""

import numpy as np
import argparse
import os

def generate(graph_size, num_samples=1000, n_cities=100, seed=1234):
    rng = np.random.default_rng(seed)

    rows = []
    for _ in range(num_samples):
        # Sample graph_size unique city indices from 1..n_cities-1 (depot=0 excluded)
        cities = rng.choice(np.arange(1, n_cities), size=graph_size, replace=False)
        row = np.concatenate(([0], cities))  # depot first
        rows.append(row)

    data = np.array(rows, dtype=int)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_size',  type=int, default=49)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--n_cities',    type=int, default=100)
    parser.add_argument('--seed',        type=int, default=1234)
    parser.add_argument('--output_dir',  type=str, default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    data = generate(args.graph_size, args.num_samples, args.n_cities, args.seed)

    out_path = os.path.join(args.output_dir, f'node_{args.graph_size}.txt')
    np.savetxt(out_path, data, fmt='%d', delimiter=' ')

    print(f"Generated {args.num_samples} instances with graph_size={args.graph_size}")
    print(f"Shape: {data.shape}  (rows x cols = samples x graph_size+1)")
    print(f"City index range: {data[:, 1:].min()} – {data[:, 1:].max()}")
    print(f"Saved to: {out_path}")
