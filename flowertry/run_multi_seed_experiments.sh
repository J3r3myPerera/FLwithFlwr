#!/bin/bash

# Multi-seed experiment runner for stratified client selection
# This script runs experiments with multiple random seeds to establish statistical significance

echo "=========================================="
echo "Multi-Seed Experiment Runner"
echo "=========================================="
echo "Running stratified vs random comparison with multiple seeds"
echo "This will take significant time (20-30 minutes per seed)"
echo ""

# Array of seeds for statistical analysis
SEEDS=(42 43 44 45 46)

# Create output directory for multi-seed results
MULTI_SEED_DIR="outputs/multi_seed_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MULTI_SEED_DIR"

echo "Results will be saved to: $MULTI_SEED_DIR"
echo ""

# Run experiment for each seed
for seed in "${SEEDS[@]}"
do
    echo "=========================================="
    echo "Running experiment with seed: $seed"
    echo "=========================================="
    
    # Run the experiment
    python main_stratified.py seed=$seed
    
    # Copy results to multi-seed directory
    LATEST_OUTPUT=$(ls -td outputs/2026-*/[0-9][0-9]-[0-9][0-9]-[0-9][0-9] 2>/dev/null | head -1)
    if [ -n "$LATEST_OUTPUT" ]; then
        cp -r "$LATEST_OUTPUT" "$MULTI_SEED_DIR/seed_$seed"
        echo "✓ Results saved to: $MULTI_SEED_DIR/seed_$seed"
    fi
    
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results directory: $MULTI_SEED_DIR"
echo ""
echo "To analyze results, run:"
echo "  python analyze_multi_seed_results.py --results_dir=$MULTI_SEED_DIR"
echo ""
