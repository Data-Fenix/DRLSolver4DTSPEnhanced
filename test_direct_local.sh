#!/bin/bash
# Quick local test for direct training method

echo "=========================================="
echo "Testing Direct Training Method Locally"
echo "=========================================="

python m1/train.py \
    --graph_size=10 \
    --n_epochs=2 \
    --seed=1234 \
    --baseline=rollout \
    --run_name=test_direct_quick \
    --training_method=direct \
    --stage_number=1 \
    --lr_model=1e-4 \
    --batch_size=64 \
    --epoch_size=640 \
    --val_size=50

echo ""
echo "=========================================="
echo "Test completed! Check:"
echo "1. outputs/tsp_10/test_direct_local_*/ for checkpoints"
echo "2. experiment_log.csv for the logged entry"
echo "=========================================="