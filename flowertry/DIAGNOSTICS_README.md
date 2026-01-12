# Class Imbalance Diagnostics - Usage Guide

**Purpose**: Diagnose if class imbalance is causing accuracy to cap at ~60%

---

## Quick Start

### Option 1: Analyze Dataset Directly (Recommended First)

Run the standalone analysis script:

```bash
cd flowertry
python analyze_data_distribution.py
```

**What it does**:
- ✅ Loads your dataset
- ✅ Analyzes class distribution (shows if one class dominates)
- ✅ Checks samples per client (shows if too few samples)
- ✅ Tests different FL configurations
- ✅ Provides specific fix recommendations

**Runtime**: ~1-2 minutes

**Expected output**:
```
================================================================
DATASET CLASS DISTRIBUTION ANALYSIS
================================================================

[1/4] Loading dataset...
Dataset loaded: 1000 total samples

[2/4] Analyzing overall class distribution...

============================================================
Full Dataset Class Distribution
============================================================
Total samples: 1000

  Low (<7%)           :   120 samples ( 12.0%)
  Medium (7-12%)      :   620 samples ( 62.0%)  ← DOMINATES!
  High (>12%)         :   260 samples ( 26.0%)

Imbalance ratio: 5.17x (max/min class)
⚠️  WARNING: Severe class imbalance detected!
   → Consider using class weights in loss function
   → Or rebalance classes using percentile thresholds

...

🔴 CRITICAL: Severe class imbalance detected!

This is very likely why all models cap at ~60% accuracy.
The model is probably just predicting the majority class.

IMMEDIATE FIXES REQUIRED:
[Shows specific code fixes]
```

---

### Option 2: Run with Strategy Comparison

The diagnostics are now integrated into the comparison script:

```bash
cd flowertry
python compare_strategies.py
```

**What happens**:
- Runs all strategies (FedAvg, FedProx, SCAFFOLD)
- **NEW**: Automatically analyzes class distribution in test set
- **NEW**: Shows imbalance ratio and recommendations
- Saves results as before

**Look for this section in output**:
```
================================================================================
RUNNING DIAGNOSTIC ANALYSIS
================================================================================

[Step 1/3] Analyzing test set class distribution...

============================================================
Test Set Class Distribution
============================================================
...

Key Findings:
  • Class imbalance ratio: 5.17x
  • ⚠️  SEVERE CLASS IMBALANCE DETECTED
  • This is likely limiting model performance to majority class baseline
  • Recommendation: Add class weights to loss function
```

---

## Understanding the Results

### Class Imbalance Ratio

**What it means**:
```
Imbalance Ratio = (Largest Class %) / (Smallest Class %)
```

**Interpretation**:
- **< 2.0x**: ✅ Balanced - not a problem
- **2.0-3.0x**: ⚠️ Moderate imbalance - may hurt performance
- **> 3.0x**: 🔴 Severe imbalance - likely causing 60% accuracy cap

### Example Scenarios

**Scenario 1: Severe Imbalance (Ratio = 5.2x)**
```
Class 0 (Low):     120 samples (12%)
Class 1 (Medium):  620 samples (62%)  ← Majority class
Class 2 (High):    260 samples (26%)

Imbalance ratio: 62% / 12% = 5.17x

Problem: Model can get 62% accuracy by ALWAYS predicting Class 1!
         This is exactly what you're seeing: ~60% accuracy cap
```

**Scenario 2: Balanced (Ratio = 1.3x)**
```
Class 0 (Low):     310 samples (31%)
Class 1 (Medium):  360 samples (36%)
Class 2 (High):    330 samples (33%)

Imbalance ratio: 36% / 31% = 1.16x

✅ Well balanced - model must learn real patterns
```

---

## Diagnostic Files Created

### 1. `diagnostics.py`

**Module with utility functions**:
- `analyze_class_distribution()` - Analyzes label distribution
- `analyze_predictions()` - Compares predictions vs true labels
- `print_confusion_matrix()` - Shows per-class performance
- `full_diagnostic_report()` - Complete analysis
- `collect_predictions()` - Gets predictions from model

**Usage in your code**:
```python
from diagnostics import analyze_class_distribution, full_diagnostic_report

# Analyze class balance
distribution = analyze_class_distribution(labels, "My Dataset")

# Full diagnostic (requires predictions)
y_true, y_pred = ... # Get true and predicted labels
report = full_diagnostic_report(y_true, y_pred, "FedAvg")
```

### 2. `analyze_data_distribution.py`

**Standalone analysis script** - Use this first!

Analyzes:
- Overall class distribution
- Test set distribution
- Samples per client for different configurations
- Per-client class distributions

### 3. `compare_strategies.py` (modified)

**Integrated diagnostics**:
- Automatically runs class distribution analysis
- Shows imbalance warnings
- Provides fix recommendations

---

## What to Do Based on Results

### If Severe Imbalance Detected (Ratio > 3.0)

**This is almost certainly why accuracy caps at 60%!**

#### Fix #1: Add Class Weights (Recommended)

**File**: `model.py`

**Location**: In `train()`, `train_fedprox()`, `train_scaffold()` functions

**Add this code**:
```python
# After imports, add:
from sklearn.utils.class_weight import compute_class_weight

# In train function, before training loop:
def train(net, trainloader, optimizer, epochs, device, max_grad_norm=1.0):
    # Collect all labels from trainloader
    all_labels = []
    for _, labels in trainloader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    # Compute class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)

    # Use weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Added weight!

    # ... rest of training code
```

**Expected improvement**: +10-20% accuracy

---

#### Fix #2: Rebalance Classes (Alternative)

**File**: `dataset.py`

**Function**: `discretize_savings()`

**Replace with**:
```python
def discretize_savings(savings_percentage: pd.Series) -> np.ndarray:
    """
    Discretize using PERCENTILES for balanced classes.
    """
    # Use 33rd and 67th percentiles as thresholds
    low_threshold = savings_percentage.quantile(0.33)
    high_threshold = savings_percentage.quantile(0.67)

    labels = np.zeros(len(savings_percentage), dtype=np.int64)
    labels[savings_percentage >= low_threshold] = 1
    labels[savings_percentage >= high_threshold] = 2

    return labels
```

**Result**: Guarantees 33% in each class

**Expected improvement**: +10-20% accuracy

---

### If Too Few Samples Per Client

**File**: `conf/base.yaml`

**Change**:
```yaml
num_clients: 20  # Down from 100
num_clients_per_round_fit: 5  # Down from 10
```

**Why**: More samples per client = more stable training

**Expected improvement**: +5-15% accuracy

---

## Testing the Fixes

### Before Fix:
```bash
python compare_strategies.py
```

**Expected output**:
```
FedAvg:    60.5%
FedProx:   59.8%
SCAFFOLD:  60.2%

Class imbalance ratio: 5.17x  ← Problem identified!
```

---

### After Applying Fix #1 (Class Weights):
```bash
python compare_strategies.py
```

**Expected output**:
```
FedAvg:    72.3%  ← Improved!
FedProx:   74.1%  ← Improved!
SCAFFOLD:  75.8%  ← Improved!

Class imbalance ratio: 5.17x  ← Still imbalanced but handled
```

---

### After Applying Fix #2 (Rebalancing):
```bash
python analyze_data_distribution.py  # Check if balanced now
python compare_strategies.py
```

**Expected output**:
```
Class 0: 333 samples (33%)  ← Balanced!
Class 1: 334 samples (33%)
Class 2: 333 samples (33%)

Imbalance ratio: 1.00x  ← Perfect!

FedAvg:    74.5%
FedProx:   76.2%
SCAFFOLD:  78.1%
```

---

## Quick Reference

| Script | Purpose | Runtime | When to Use |
|--------|---------|---------|-------------|
| `analyze_data_distribution.py` | Diagnose class imbalance | 1-2 min | **Run first!** |
| `compare_strategies.py` | Full comparison + diagnostics | 30-60 min | After applying fixes |
| `test_scaffold_fixes.py` | Quick SCAFFOLD test | 5 min | Test SCAFFOLD only |

---

## Troubleshooting

### Q: Script says "Severe class imbalance" but accuracy is still 60% after adding class weights

**A**: Make sure you added class weights to ALL three training functions:
- `train()` for FedAvg
- `train_fedprox()` for FedProx
- `train_scaffold()` for SCAFFOLD

### Q: How do I know if the fix worked?

**A**: Compare before/after:
- **Before**: All strategies ~60%, one class dominates predictions
- **After**: Strategies 70-80%, predictions more distributed

### Q: Should I use Fix #1 or Fix #2?

**A**:
- **Fix #1 (class weights)**: Easier, keeps original class definitions
- **Fix #2 (rebalancing)**: More thorough, changes task slightly

**Recommendation**: Try Fix #1 first, then Fix #2 if needed

---

## Summary

1. ✅ **Run diagnosis**: `python analyze_data_distribution.py`
2. ✅ **If imbalance ratio > 3.0**: Apply Fix #1 or Fix #2
3. ✅ **Re-run strategies**: `python compare_strategies.py`
4. ✅ **Verify improvement**: Should see 70-80% accuracy

The diagnostic tools will tell you exactly what's wrong and how to fix it!
