# Round-by-Round Training Visualization - Complete Solution

## Problem

Your federated learning simulations were not storing round-by-round metrics properly, making it impossible to create training progression plots like the one you showed me (Training Loss, Validation Loss, MAPE, and Final Metrics comparison).

## Root Cause

The history extraction code in `main.py` (lines 286-301) wasn't handling Flower's `History` object correctly. The `metrics_centralized` dictionary returns tuples `(num_examples, metrics_dict)`, not just the metrics dict directly.

## Solution Implemented

### 1. Fixed History Extraction in `main.py`

**Location**: Lines 286-318

**What Changed**:

```python
# OLD (wasn't working):
if result.metrics_centralized:
    for round_num, metrics in result.metrics_centralized.items():
        if isinstance(metrics, dict):
            # ... collect metrics

# NEW (fixed):
if hasattr(result, 'metrics_centralized') and result.metrics_centralized:
    sorted_rounds = sorted(result.metrics_centralized.items(), key=lambda x: x[0])
    for round_num, metrics_tuple in sorted_rounds:
        # Handle tuple format: (num_examples, metrics_dict)
        if isinstance(metrics_tuple, tuple) and len(metrics_tuple) == 2:
            _, metrics = metrics_tuple
        # ... collect metrics
```

**Why This Fixes It**:

- Properly extracts metrics from the tuple format
- Sorts rounds to ensure chronological order
- Handles both tuple and dict formats for robustness
- Converts all values to float for JSON serialization

### 2. Created `plot_round_progression.py`

**Purpose**: Read `comparison_results.json` and create visualizations

**Features**:

- 2×2 grid plot (like your attached image):
  - Top-left: Training Loss progression
  - Top-right: R² Score progression (validation performance)
  - Bottom-left: MAPE progression with 15% target line
  - Bottom-right: Final metrics bar chart (MAPE, Acc@10%, Acc@20%)
- Optional detailed 3×2 grid with all metrics
- Optional round-by-round summary tables
- Automatic latest simulation detection
- Color-coded strategies

### 3. Created `test_plotting.py`

**Purpose**: Generate mock data to test the plotting without running a full simulation

**What It Does**:

- Creates synthetic 10-round training data
- Simulates improving metrics over rounds
- Saves to `outputs/test_plotting/comparison_results.json`
- Perfect for testing the plotting script

### 4. Documentation Files

- `README_PLOTTING.md` - Comprehensive guide with troubleshooting
- `USAGE.py` - Interactive quick start guide
- `SOLUTION.md` - This file

## How to Use

### Quick Test (No simulation needed)

```bash
# 1. Generate test data
python test_plotting.py

# 2. Plot it
python plot_round_progression.py --output-dir outputs/test_plotting --summary

# 3. Check the output
open outputs/test_plotting/training_progression.png
```

### With Real Simulations

```bash
# 1. Run your FL simulation
python main.py --config-name "base copy 2"

# 2. Plot the results (automatically finds latest)
python plot_round_progression.py

# Or plot a specific simulation
python plot_round_progression.py --output-dir outputs/2026-01-19/21-52-59

# With all options
python plot_round_progression.py --detailed --summary
```

## Verification

### Check if History is Being Collected

```bash
cd outputs/2026-01-19/21-52-59
cat comparison_results.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for strategy, data in d.items():
    if 'history' in data:
        num_rounds = len(data['history'].get('rounds', []))
        print(f'{strategy}: {num_rounds} rounds')
"
```

**Expected Output (after fix)**:

```
fedavg: 30 rounds
fedprox: 30 rounds
scaffold: 30 rounds
hybrid: 30 rounds
```

**Before Fix**:

```
fedavg: 0 rounds
fedprox: 0 rounds
scaffold: 0 rounds
hybrid: 0 rounds
```

## Output Files

When you run `plot_round_progression.py`, it generates:

1. **`training_progression.png`** (always created)
   - 2×2 grid plot matching your example image
   - Shows all strategies on same axes for comparison
   - 300 DPI high quality for presentations

2. **`detailed_metrics_progression.png`** (with `--detailed` flag)
   - 3×2 grid with individual plots for each metric
   - Loss, RMSE, MAE, R², MAPE, Accuracy@10%

## Important Notes

⚠️ **Simulations run BEFORE the fix will not have history data**

- The fix only affects NEW simulations
- Old `comparison_results.json` files have empty history['rounds']
- You must run a new simulation to test the plotting

✅ **Config Requirements**

- Must have `compare_all: true` to generate `comparison_results.json`
- If running single strategy mode, only that strategy will be plotted

## What Gets Plotted

For each strategy, the script plots these metrics over rounds:

1. **Loss**: MSE loss (should decrease)
2. **RMSE**: Root Mean Square Error in $ (should decrease)
3. **MAE**: Mean Absolute Error in $ (should decrease)
4. **R²**: R-squared score (should increase, max 1.0)
5. **MAPE**: Mean Absolute Percentage Error % (should decrease)
6. **Accuracy@10%**: % predictions within ±10% (should increase)
7. **Accuracy@20%**: % predictions within ±20% (should increase)

## Example Output

After running the test:

```
📂 Using output directory: outputs/test_plotting

✅ Found history data for 4 strategies:
   - fedavg: 10 rounds
   - fedprox: 10 rounds
   - scaffold: 10 rounds
   - hybrid: 10 rounds

✅ Plot saved to: outputs/test_plotting/training_progression.png
✅ Done!
```

With `--summary` flag, you also get:

```
====================================================================================================
ROUND-BY-ROUND METRICS SUMMARY
====================================================================================================

📊 HYBRID
----------------------------------------------------------------------------------------------------
 Round |       Loss |       RMSE |       R² |    MAPE |  Acc@10% |  Acc@20%
----------------------------------------------------------------------------------------------------
     1 |   0.035862 | $    1,434 |   0.6414 |   8.18% |   57.32% |   59.77%
     2 |   0.030530 | $    1,221 |   0.6947 |   7.59% |   62.74% |   63.96%
     ...
    10 |   0.008119 | $      325 |   0.9188 |   7.06% |   83.65% |   88.08%
```

## Troubleshooting

### "No history data found"

**Cause**: Simulation was run before the fix  
**Solution**: Run a new simulation

### "No comparison_results.json found"

**Cause**: Wrong directory or single strategy mode  
**Solution**:

- Check you're in the right directory
- Ensure `compare_all: true` in config

### Empty/Flat Lines in Plots

**Cause**: History collected but all values are zero  
**Solution**:

- Check `server.py:compute_regression_metrics()` is working
- Verify `strategy.evaluate()` returns metrics

### "No module named matplotlib"

**Cause**: Missing dependencies  
**Solution**: `pip install matplotlib numpy`

## Files Modified/Created

### Modified:

- `main.py` (lines 286-318) - Fixed history extraction

### Created:

- `plot_round_progression.py` - Main plotting script ⭐
- `test_plotting.py` - Test data generator
- `README_PLOTTING.md` - Detailed documentation
- `USAGE.py` - Interactive quick start
- `SOLUTION.md` - This file

## Summary

You now have:

1. ✅ Fixed history collection in main.py
2. ✅ Plotting script that works like your example image
3. ✅ Test script to verify without running full simulation
4. ✅ Complete documentation

The plotting script will generate professional-quality graphs showing:

- How each strategy improves over rounds
- Comparative performance across strategies
- Final metrics bar chart
- MAPE progression with target line (like your example)

All future simulations will automatically collect round-by-round metrics that can be plotted with a simple command: `python plot_round_progression.py`
