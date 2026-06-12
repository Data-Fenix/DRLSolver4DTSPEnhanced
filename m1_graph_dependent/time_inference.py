"""
time_inference.py  —  Measure inference wall-clock time for DTSP models.

Run from the m1_graph_dependent/ directory:
    python time_inference.py

Results printed:
    - Total inference time (mean ± std over N_RUNS) for all instances
    - Per-instance time in milliseconds
    - Detected graph_size for each model
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Force UTF-8 output so Unicode characters from transformer.py don't crash on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline
from transformer import AttentionModel

# ─────────────────────────────────────────────
#  Configuration – edit paths as needed
# ─────────────────────────────────────────────

MODELS = {
    # ── 19-city models (will be timed on valid_data_19.txt) ──────────────────
    "Baseline_19":         "../outputs/tsp_19/tsp19_baseline_no_features_seed1234_epoch99.pt",
    "CA_LT_lam0.25_19":   "../outputs/tsp_19/linear_time_lambda0.25_seed1234_epoch99.pt",
    "TimeSlicing_W9_19":  "../outputs/tsp_19/time_scling_phase3_W9_periodic_0.25_s1234_epoch49.pt",
    "TempMLP_19":          "../outputs/tsp_19/temp_mlp_only_seed1234_epoch49.pt",
    # ── 49-city models (will be timed on valid_data_49.txt) ──────────────────
    "Baseline_49":         "../outputs/tsp_49/tsp49_baseline_no_features_seed1234_epoch99.pt",
    "CA_NN_49":            "../outputs/tsp_49/tsp49_costaware_only_nn_s1234_epoch99.pt",
    "TimeSlicing_W3_49":   "../outputs/tsp_49/tsp49_phase5_W3_periodic0p25_s1234_epoch99.pt",
    "TempMLP_49":          "../outputs/tsp_49/tsp49_temp_mlp_v2light_seed1234_epoch99.pt",
}

# Dataset to evaluate on. The script auto-selects the matching dataset
# based on graph_size detected from each model.
DATASET_MAP = {
    19: "../valid_data_19.txt",
    49: "../valid_data_49.txt",
}
# Fallback if detected size is not in DATASET_MAP above:
DEFAULT_DATASET = "../valid_data_49.txt"

DATA_CSV   = "data.csv"        # travel-time CSV (in m1_graph_dependent/)
N_CITIES   = 100               # city pool size
N_SAMPLES  = 1000                # set to 10 for testing, 1000 for full run
BATCH_SIZE = 256              # instances per forward pass
N_WARMUP   = 1                 # warmup passes (not timed)
N_RUNS     = 5                 # timed runs to average

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  Minimal helpers (mirrors test.py / train.py)
# ─────────────────────────────────────────────

class Cities:
    def __init__(self, n_cities=100):
        self.n_cities = n_cities
        self.cities   = torch.rand((n_cities, 2))


class DistanceMatrix:
    def __init__(self, ci, load_dir, max_time_step=12):
        n  = ci.n_cities
        mt = max_time_step
        self.n_c           = n
        self.max_time_step = mt
        self.mat = torch.zeros(n * n * mt, device=DEVICE)
        self.m2  = torch.zeros(n * n * mt, device=DEVICE)
        self.m3  = torch.zeros(n * n * mt, device=DEVICE)
        self.m4  = torch.zeros(n * n * mt, device=DEVICE)
        self.var = torch.full((n * n,), 0.03, device=DEVICE)
        tmp = np.loadtxt(load_dir, delimiter=",")
        x   = np.arange(mt + 1)
        for k in range(n):
            self.var[k * n + k] = 0
            for j in range(n):
                i  = k * n + j
                cs = CubicSpline(
                    x, np.concatenate((tmp[i], [tmp[i, 0]])), bc_type="periodic"
                )
                sl = slice(i * mt, i * mt + mt)
                self.mat[sl] = torch.tensor(cs.c[3], device=DEVICE)
                self.m2[sl]  = torch.tensor(cs.c[2], device=DEVICE)
                self.m3[sl]  = torch.tensor(cs.c[1], device=DEVICE)
                self.m4[sl]  = torch.tensor(cs.c[0], device=DEVICE)

    def __getd__(self, st, a, b, t):
        a = torch.gather(st, 1, a)
        b = torch.gather(st, 1, b)
        tt = torch.floor(t * self.max_time_step) % self.max_time_step
        zz = (torch.floor(t * self.max_time_step) + 1) % self.max_time_step
        c = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + tt.squeeze().long()
        d = a.squeeze() * self.n_c * self.max_time_step + b.squeeze() * self.max_time_step + zz.squeeze().long()
        a0 = torch.gather(self.mat, 0, c)
        a1 = torch.gather(self.m2,  0, c)
        a2 = torch.gather(self.m3,  0, c)
        a3 = torch.gather(self.m4,  0, c)
        b0 = torch.gather(self.mat, 0, d)
        z  = (t.squeeze() * self.max_time_step - torch.floor(t.squeeze() * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res, _ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim=-1), dim=-1)
        res, _ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim=-1), dim=-1)
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
        a1 = torch.gather(self.m2,  0, c)
        a2 = torch.gather(self.m3,  0, c)
        a3 = torch.gather(self.m4,  0, c)
        b0 = torch.gather(self.mat, 0, d)
        tt  = tt.view(-1)
        ttt = t.expand(s0, s1).contiguous().view(-1)
        z  = (ttt * self.max_time_step - torch.floor(ttt * self.max_time_step)) / self.max_time_step
        z2 = z * z
        z3 = z2 * z
        res = a0 + a1 * z + a2 * z2 + a3 * z3
        minres = (a0 + b0) * 0.05
        maxres = (a0 + b0) * 5
        res, _ = torch.max(torch.cat((res.unsqueeze(-1), minres.unsqueeze(-1)), dim=-1), dim=-1)
        res, _ = torch.min(torch.cat((res.unsqueeze(-1), maxres.unsqueeze(-1)), dim=-1), dim=-1)
        return res.view(s0, s1)


class TSPDataset(Dataset):
    """Load pre-generated city-index sequences from a text file."""

    def __init__(self, filename, graph_size, n_cities):
        raw = np.loadtxt(filename, delimiter=" ", max_rows=N_SAMPLES)  # limit rows
        ind = torch.tensor(raw, dtype=torch.long)          # [N, graph_size+1]
        # Truncate or keep only graph_size+1 steps
        ind = ind[:, : graph_size + 1].unsqueeze(2)        # [N, gs+1, 1]
        n   = ind.size(0)
        self.data = torch.zeros(n, graph_size + 1, n_cities).scatter_(2, ind, 1.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
#  Model utilities
# ─────────────────────────────────────────────

def detect_graph_size(state_dict):
    """
    Infer graph_size from the shape of `project_visit.weight`.
    That layer has shape [embedding_dim, input_size] where input_size = graph_size + 1.
    Returns None if the key is absent.
    """
    key = "project_visit.weight"
    if key in state_dict:
        input_size = state_dict[key].shape[1]
        return input_size - 1
    return None


def build_model(state_dict, graph_size):
    """Instantiate AttentionModel with architecture auto-detected from state_dict."""
    has_step_mlp     = any("step_mlp"         in k for k in state_dict)
    has_cost_aware   = any("lambda_heuristic" in k or "heuristic_computer" in k for k in state_dict)
    has_temp_mlp     = any("temp_mlp"         in k for k in state_dict)
    has_time_slicing = any("embed_windowed_traffic" in k for k in state_dict)

    # Detect step feature flags from the mlp input dimension
    # temp_mlp.0.weight shape: [hidden_dim, step_input_dim]
    use_step_ratio = use_linear_time = use_depot_distance = False
    use_tour_length = use_mean_dist_unvisited = False
    use_sin_cos_time = use_visited_mean = use_unvisited_mean = use_last_3_nodes = False

    if has_step_mlp or has_temp_mlp:
        mlp_w_key = next((k for k in state_dict if k in ("temp_mlp.0.weight", "step_mlp.0.weight")), None)
        if mlp_w_key:
            input_dim = state_dict[mlp_w_key].shape[1]
            # Map known input_dim values to the feature flags that produced them.
            # dim=5: step_ratio(1)+linear_time(1)+depot_distance(1)+tour_length(1)+mean_dist_unvisited(1)
            # dim=6: step_ratio(1)+sin_cos_time(2)+depot_distance(1)+tour_length(1)+mean_dist_unvisited(1)
            if input_dim == 5:
                use_step_ratio = use_linear_time = use_depot_distance = True
                use_tour_length = use_mean_dist_unvisited = True
            elif input_dim == 6:
                use_step_ratio = use_sin_cos_time = use_depot_distance = True
                use_tour_length = use_mean_dist_unvisited = True
            else:
                # Fallback: enable step_ratio only to satisfy the safety check.
                # load_state_dict will report a shape mismatch below.
                use_step_ratio = True

    model = AttentionModel(
        embedding_dim        = 128,
        hidden_dim           = 128,
        n_encode_layers      = 2,
        mask_inner           = True,
        mask_logits          = True,
        normalization        = "batch",
        tanh_clipping        = 10.0,
        checkpoint_encoder   = False,
        shrink_size          = None,
        step_mlp_dim         = 64,
        use_step_mlp         = has_step_mlp,
        use_temp_mlp         = has_temp_mlp,
        use_step_ratio       = use_step_ratio,
        use_linear_time      = use_linear_time,
        use_sin_cos_time     = use_sin_cos_time,
        use_depot_distance   = use_depot_distance,
        use_tour_length      = use_tour_length,
        use_mean_dist_unvisited = use_mean_dist_unvisited,
        use_visited_mean     = use_visited_mean,
        use_unvisited_mean   = use_unvisited_mean,
        use_last_3_nodes     = use_last_3_nodes,
        input_size           = graph_size + 1,
        max_t                = 12,
        use_cost_aware_gating= has_cost_aware,
        heuristic_type       = "linear_time",
        lambda_heuristic     = 1.0,
        use_nonlinear_transform = False,
        transform_type       = "piecewise",
        use_time_slicing     = has_time_slicing,   # architectural layer — must match checkpoint
        window_size_W        = 12,                 # fixed: embed_windowed_traffic always uses max_t=12
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()
    model.set_decode_type("greedy")
    return model


def run_inference(model, mat, dataset):
    """Single inference pass; returns costs tensor."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    costs  = []
    with torch.no_grad():
        for batch in loader:
            cost, _, _ = model(mat, batch.to(DEVICE))
            costs.append(cost.cpu())
    return torch.cat(costs)


def measure_time(model, mat, dataset):
    """
    Warm up, then time N_RUNS inference passes.
    Returns (mean_seconds, std_seconds).
    """
    # Warmup (fills CUDA caches, JIT paths, etc.)
    for _ in range(N_WARMUP):
        run_inference(model, mat, dataset)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        run_inference(model, mat, dataset)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)

    return float(np.mean(timings)), float(np.std(timings))


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")
    print(f"Runs   : {N_WARMUP} warmup + {N_RUNS} timed\n")

    print("Loading distance matrix …")
    ci  = Cities(N_CITIES)
    mat = DistanceMatrix(ci, DATA_CSV, max_time_step=12)
    print("Distance matrix ready.\n")

    header = f"{'Model':<22}  {'Size':>5}  {'Mean(s)':>9}  {'Std(s)':>8}  {'ms/inst':>9}  {'N':>6}"
    print(header)
    print("-" * len(header))

    results = []   # collect rows for Excel export

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"{name:<22}  [FILE NOT FOUND: {path}]")
            continue

        # Load checkpoint
        ckpt = torch.load(path, map_location="cpu")
        sd   = ckpt.get("model", ckpt)   # handle both dict formats

        # Auto-detect graph_size
        gs = detect_graph_size(sd)
        if gs is None:
            print(f"{name:<22}  [Cannot detect graph_size — 'project_visit.weight' missing]")
            continue

        # Select dataset
        dataset_file = DATASET_MAP.get(gs, DEFAULT_DATASET)
        if not os.path.exists(dataset_file):
            print(f"{name:<22}  gs={gs}  [Dataset not found: {dataset_file}]")
            continue

        # Check column count in dataset vs graph_size
        first_row = np.loadtxt(dataset_file, max_rows=1)
        dataset_gs = int(first_row.shape[0]) - 1   # columns = graph_size + 1 (depot)
        if dataset_gs != gs:
            print(
                f"{name:<22}  gs={gs}  "
                f"[WARNING: dataset has {dataset_gs} cities but model expects {gs}. "
                f"Using first {gs+1} columns — results may be meaningless for quality, "
                f"but timing is still valid.]"
            )

        # Build model
        try:
            model = build_model(sd, gs)
        except Exception as e:
            print(f"{name:<22}  gs={gs}  [Model load error: {e}]")
            continue

        # Load dataset
        dataset = TSPDataset(dataset_file, gs, N_CITIES)
        n_inst  = len(dataset)

        # Time inference
        mean_t, std_t = measure_time(model, mat, dataset)
        ms_per = mean_t / n_inst * 1000.0

        print(f"{name:<22}  {gs:>5}  {mean_t:>9.4f}  {std_t:>8.4f}  {ms_per:>9.4f}  {n_inst:>6}")

        results.append({
            "Model":          name,
            "Graph Size":     gs,
            "N Instances":    n_inst,
            "N Warmup":       N_WARMUP,
            "N Runs":         N_RUNS,
            "Mean Time (s)":  round(mean_t, 6),
            "Std Time (s)":   round(std_t,  6),
            "ms per Instance":round(ms_per, 4),
            "Device":         str(DEVICE),
            "Checkpoint":     path,
        })

    print("\nNote: 19-city models cannot produce valid results on 49-city data.")
    print("      For fair comparison, use matching datasets (valid_data_19 / valid_data_49).")
    print("      49-city best models are in: ../outputs/tsp_49/")

    # ── Save to Excel ────────────────────────────────────────────────────────
    if results:
        df        = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = f"../inference_timing_{timestamp}.xlsx"
        df.to_excel(out_path, index=False)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
