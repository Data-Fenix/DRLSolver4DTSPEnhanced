#!/bin/bash
# Comparison experiment: Direct vs 2-Stage vs 3-Stage training on 20-node TSP
# This script runs all three training methods for statistical comparison

#SBATCH --job-name=tsp20_comparison
#SBATCH --output=logs/tsp20_comparison_%j.out
#SBATCH --error=logs/tsp20_comparison_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Activate virtual environment
source ~/anuradha/DRLSolver4DTSP-main/venv/bin/activate

# Move to project folder
cd ~/anuradha/DRLSolver4DTSP-main

# ==========================================
# Configuration
# ==========================================
SEEDS=(1234 5678 9012)  # Use 3-5 seeds for statistical robustness
TARGET_SIZE=20
BASE_LR=1e-4

echo "=========================================="
echo "TSP 20-Node Training Method Comparison"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# ==========================================
# Method 1: Direct Training (100 epochs on 20-node)
# ==========================================
echo ""
echo "=========================================="
echo "METHOD 1: Direct Training (100 epochs on 20-node)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Running Direct Training with seed=$SEED ---"
    python m1/train.py \
        --graph_size=$TARGET_SIZE \
        --n_epochs=100 \
        --seed=$SEED \
        --baseline=rollout \
        --run_name=direct_20node_100ep_seed${SEED} \
        --training_method=direct \
        --stage_number=1 \
        --lr_model=$BASE_LR
done

# ==========================================
# Method 2: 2-Stage Progressive Training
# ==========================================
echo ""
echo "=========================================="
echo "METHOD 2: 2-Stage Progressive Training"
echo "Stage 1: 10-node (50 epochs) -> Stage 2: 20-node (50 epochs)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Running 2-Stage Training with seed=$SEED ---"
    
    # Stage 1: Train on 10-node for 50 epochs
    echo "  Stage 1: Training on graph_size=10 (50 epochs)"
    python m1/train.py \
        --graph_size=10 \
        --n_epochs=50 \
        --seed=$SEED \
        --baseline=rollout \
        --run_name=2stage_10node_50ep_seed${SEED} \
        --training_method=2-stage \
        --stage_number=1 \
        --lr_model=$BASE_LR
    
    # Find the latest checkpoint from stage 1
    echo "  Finding latest checkpoint from stage 1..."
    STAGE1_DIR=$(find outputs/tsp_10 -type d -name "2stage_10node_50ep_seed${SEED}_*" 2>/dev/null | sort -r | head -n 1)
    
    if [ -z "$STAGE1_DIR" ]; then
        echo "  ERROR: Could not find stage1 checkpoint directory for seed=$SEED"
        continue
    fi
    
    echo "  Found directory: $STAGE1_DIR"
    
    # Get epoch-49.pt (last epoch of stage 1)
    STAGE1_CHECKPOINT=$(find "$STAGE1_DIR" -name "epoch-49.pt" 2>/dev/null)
    if [ -z "$STAGE1_CHECKPOINT" ] || [ ! -f "$STAGE1_CHECKPOINT" ]; then
        # Fallback to latest checkpoint
        STAGE1_CHECKPOINT=$(find "$STAGE1_DIR" -name "epoch-*.pt" 2>/dev/null | sort -V | tail -n 1)
    fi
    
    if [ -z "$STAGE1_CHECKPOINT" ] || [ ! -f "$STAGE1_CHECKPOINT" ]; then
        echo "  ERROR: Checkpoint not found in $STAGE1_DIR"
        continue
    fi
    
    echo "  Using checkpoint: $STAGE1_CHECKPOINT"
    
    # Stage 2: Transfer to 20-node for 50 epochs
    echo "  Stage 2: Training on graph_size=20 (50 epochs, transferring from stage 1)"
    python m1/train.py \
        --graph_size=$TARGET_SIZE \
        --n_epochs=50 \
        --seed=$SEED \
        --load_path="$STAGE1_CHECKPOINT" \
        --baseline=rollout \
        --run_name=2stage_20node_50ep_seed${SEED} \
        --training_method=2-stage \
        --stage_number=2 \
        --lr_model=$BASE_LR
    
    echo "  2-Stage training completed for seed=$SEED"
done

# ==========================================
# Method 3: 3-Stage Progressive Training
# ==========================================
echo ""
echo "=========================================="
echo "METHOD 3: 3-Stage Progressive Training"
echo "Stage 1: 10-node (40 epochs) -> Stage 2: 15-node (30 epochs) -> Stage 3: 20-node (30 epochs)"
echo "=========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "--- Running 3-Stage Training with seed=$SEED ---"
    
    # Stage 1: Train on 10-node for 40 epochs
    echo "  Stage 1: Training on graph_size=10 (40 epochs)"
    python m1/train.py \
        --graph_size=10 \
        --n_epochs=40 \
        --seed=$SEED \
        --baseline=rollout \
        --run_name=3stage_10node_40ep_seed${SEED} \
        --training_method=3-stage \
        --stage_number=1 \
        --lr_model=$BASE_LR
    
    # Find the latest checkpoint from stage 1
    echo "  Finding latest checkpoint from stage 1..."
    STAGE1_DIR=$(find outputs/tsp_10 -type d -name "3stage_10node_40ep_seed${SEED}_*" 2>/dev/null | sort -r | head -n 1)
    
    if [ -z "$STAGE1_DIR" ]; then
        echo "  ERROR: Could not find stage1 checkpoint directory for seed=$SEED"
        continue
    fi
    
    echo "  Found directory: $STAGE1_DIR"
    
    # Get epoch-39.pt (last epoch of stage 1)
    STAGE1_CHECKPOINT=$(find "$STAGE1_DIR" -name "epoch-39.pt" 2>/dev/null)
    if [ -z "$STAGE1_CHECKPOINT" ] || [ ! -f "$STAGE1_CHECKPOINT" ]; then
        # Fallback to latest checkpoint
        STAGE1_CHECKPOINT=$(find "$STAGE1_DIR" -name "epoch-*.pt" 2>/dev/null | sort -V | tail -n 1)
    fi
    
    if [ -z "$STAGE1_CHECKPOINT" ] || [ ! -f "$STAGE1_CHECKPOINT" ]; then
        echo "  ERROR: Checkpoint not found in $STAGE1_DIR"
        continue
    fi
    
    echo "  Using checkpoint: $STAGE1_CHECKPOINT"
    
    # Stage 2: Transfer to 15-node for 30 epochs
    echo "  Stage 2: Training on graph_size=15 (30 epochs, transferring from stage 1)"
    python m1/train.py \
        --graph_size=15 \
        --n_epochs=30 \
        --seed=$SEED \
        --load_path="$STAGE1_CHECKPOINT" \
        --baseline=rollout \
        --run_name=3stage_15node_30ep_seed${SEED} \
        --training_method=3-stage \
        --stage_number=2 \
        --lr_model=$BASE_LR
    
    # Find the latest checkpoint from stage 2
    echo "  Finding latest checkpoint from stage 2..."
    STAGE2_DIR=$(find outputs/tsp_15 -type d -name "3stage_15node_30ep_seed${SEED}_*" 2>/dev/null | sort -r | head -n 1)
    
    if [ -z "$STAGE2_DIR" ]; then
        echo "  ERROR: Could not find stage2 checkpoint directory for seed=$SEED"
        continue
    fi
    
    echo "  Found directory: $STAGE2_DIR"
    
    # Get epoch-29.pt (last epoch of stage 2)
    STAGE2_CHECKPOINT=$(find "$STAGE2_DIR" -name "epoch-29.pt" 2>/dev/null)
    if [ -z "$STAGE2_CHECKPOINT" ] || [ ! -f "$STAGE2_CHECKPOINT" ]; then
        # Fallback to latest checkpoint
        STAGE2_CHECKPOINT=$(find "$STAGE2_DIR" -name "epoch-*.pt" 2>/dev/null | sort -V | tail -n 1)
    fi
    
    if [ -z "$STAGE2_CHECKPOINT" ] || [ ! -f "$STAGE2_CHECKPOINT" ]; then
        echo "  ERROR: Checkpoint not found in $STAGE2_DIR"
        continue
    fi
    
    echo "  Using checkpoint: $STAGE2_CHECKPOINT"
    
    # Stage 3: Transfer to 20-node for 30 epochs
    echo "  Stage 3: Training on graph_size=20 (30 epochs, transferring from stage 2)"
    python m1/train.py \
        --graph_size=$TARGET_SIZE \
        --n_epochs=30 \
        --seed=$SEED \
        --load_path="$STAGE2_CHECKPOINT" \
        --baseline=rollout \
        --run_name=3stage_20node_30ep_seed${SEED} \
        --training_method=3-stage \
        --stage_number=3 \
        --lr_model=$BASE_LR
    
    echo "  3-Stage training completed for seed=$SEED"
done

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Generating comparison summary..."
python -c "
from experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()
tracker.compare_methods(['direct', '2-stage', '3-stage'])
"
echo ""
echo "Results saved to: experiment_log.csv"
echo "Check individual experiment outputs in: outputs/tsp_*/"
