#!/bin/bash
# Quick Test Script for Hybrid Strategy Improvements
# Run this to compare all strategies and verify Hybrid performs best

echo "=================================================="
echo "Testing Enhanced Hybrid FL Strategy"
echo "=================================================="
echo ""
echo "This will run FedAvg, FedProx, SCAFFOLD, and Hybrid"
echo "Expected: Hybrid should show best performance"
echo ""
echo "Key Metrics to Watch:"
echo "  - Lowest RMSE (target: <3000)"
echo "  - Highest R² (target: >0.85)"
echo "  - Lowest MAPE (target: <15%)"
echo "  - Fastest convergence"
echo ""
echo "=================================================="
echo ""

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate conda environment
echo "Activating flower_tutorial environment..."
eval "$(conda shell.bash hook)"
conda activate flower_tutorial

# Run comparison
echo ""
echo "Starting strategy comparison..."
echo "This will take 15-30 minutes depending on your hardware"
echo ""

python main.py compare_all=true num_rounds=45

echo ""
echo "=================================================="
echo "Testing Complete!"
echo "=================================================="
echo ""
echo "Results saved in: outputs/$(date +%Y-%m-%d)/"
echo ""
echo "To view just Hybrid performance:"
echo "  python main.py strategy=hybrid"
echo ""
echo "Check HYBRID_IMPROVEMENTS.md for detailed explanation"
echo "=================================================="
