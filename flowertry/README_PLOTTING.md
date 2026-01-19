# Round-by-Round Plotting Guide

## Problem Identified

The current implementation in `main.py` (lines 286-301) collects round-by-round metrics during training, but the history extraction wasn't working correctly. The metrics are supposed to be stored in `comparison_results.json`, but the `rounds` list was empty.

## Fix Applied

### 1. Fixed History Extraction in `main.py`

**Issue**: The `result.metrics_centralized` returns tuples `(num_examples, metrics_dict)`, not just the metrics dict directly.

**Fixed Code** (lines 286-318):

```python
# Extract history from result
# metrics_centralized is a dict where keys are round numbers and values are tuples (num_examples, metrics_dict)
if hasattr(result, 'metrics_centralized') and result.metrics_centralized:
    # Sort by round number to ensure proper ordering
    sorted_rounds = sorted(result.metrics_centralized.items(), key=lambda x: x[0])

    for round_num, metrics_tuple in sorted_rounds:
        # metrics_tuple is typically (num_examples, metrics_dict)
        if isinstance(metrics_tuple, tuple) and len(metrics_tuple) == 2:
            _, metrics = metrics_tuple
        elif isinstance(metrics_tuple, dict):
            metrics = metrics_tuple
        else:
            continue

        if metrics:
            # Get corresponding loss
            loss_value = result.losses_centralized.get(round_num, (0, 0))
            loss = loss_value[1] if isinstance(loss_value, tuple) else loss_value

            history["rounds"].append(round_num)
            history["loss"].append(float(loss))
            history["rmse"].append(float(metrics.get("rmse", 0)))
            # ... etc
```

### 2. Created New Plotting Script: `plot_round_progression.py`

This script reads the `comparison_results.json` file and creates visualizations like your attached image.

**Features**:

- 2×2 grid plot with:
  - Training Loss progression
  - Validation Performance (R² Score)
  - MAPE Over Training (with 15% target line)
  - Final Metrics Comparison (bar chart)
- Detailed 3×2 grid with all metrics
- Summary tables

## Usage

### After Running a Simulation

1. **Run your FL simulation as usual**:

   ```bash
   python main.py --config-name "base copy 2"
   ```

2. **Plot the results from latest simulation**:

   ```bash
   python plot_round_progression.py
   ```

3. **Or specify a specific output directory**:

   ```bash
   python plot_round_progression.py --output-dir outputs/2026-01-19/21-52-59
   ```

4. **With detailed metrics plots**:

   ```bash
   python plot_round_progression.py --detailed
   ```

5. **With summary table**:

   ```bash
   python plot_round_progression.py --summary
   ```

6. **All options**:
   ```bash
   python plot_round_progression.py --detailed --summary
   ```

## Output Files

The script generates:

- `training_progression.png` - Main 2×2 plot (like your attached image)
- `detailed_metrics_progression.png` - Detailed 3×2 plot (if --detailed flag used)

## Testing the Fix

Since your most recent simulation (21-52-59) was run before the fix, you'll need to run a new simulation to test:

```bash
# Run a short test simulation (5 rounds)
python main.py --config-name "base copy 2" num_rounds=5

# Then plot the results
python plot_round_progression.py

# If it works, you'll see training_progression.png with all rounds plotted
```

## Expected Output

After running the plotting script, you should see:

```
📂 Using output directory: outputs/2026-01-19/XX-XX-XX

✅ Found history data for 4 strategies:
   - fedavg: 30 rounds
   - fedprox: 30 rounds
   - scaffold: 30 rounds
   - hybrid: 30 rounds

✅ Plot saved to: outputs/2026-01-19/XX-XX-XX/training_progression.png
✅ Done!
```

## Verifying History Collection

To check if history is being collected in an existing simulation:

```bash
cd outputs/2026-01-19/21-52-59
cat comparison_results.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
for strategy, data in d.items():
    if 'history' in data:
        num_rounds = len(data['history'].get('rounds', []))
        print(f'{strategy}: {num_rounds} rounds in history')
"
```

**Expected output after fix**:

```
fedavg: 30 rounds in history
fedprox: 30 rounds in history
scaffold: 30 rounds in history
hybrid: 30 rounds in history
```

**Before fix** (what you currently see):

```
fedavg: 0 rounds in history
fedprox: 0 rounds in history
scaffold: 0 rounds in history
hybrid: 0 rounds in history
```

## Troubleshooting

### Issue: "No history data found"

- **Cause**: Simulation was run before the fix
- **Solution**: Run a new simulation

### Issue: "No comparison_results.json found"

- **Cause**: Running in single strategy mode instead of comparison mode
- **Solution**: Make sure `compare_all: true` in your config

### Issue: Empty plots

- **Cause**: History was collected but metrics are missing
- **Solution**: Check that all metrics are being computed in `server.py:compute_regression_metrics()`

## Next Steps

After confirming the fix works:

1. Run your full 30-round simulations
2. Use `plot_round_progression.py` to generate visualizations
3. Compare plots across different configurations
4. Save plots for your reports/presentations
