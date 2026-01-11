# Critical Fixes for FedAvg, FedProx, and SCAFFOLD Implementation

**Date**: 2026-01-11
**Issue**: FedAvg and FedProx failing with error: `'list' object has no attribute 'tensors'`

---

## Problem Diagnosis

### Root Cause
The error occurs because FedAvg and FedProx strategies don't have initial parameters set, causing Flower to request initial parameters from a client. When the evaluation function receives these parameters, they may not be in the expected `Parameters` format.

Additionally, **CRITICAL DATA ISSUE**: The `alpha` parameter in `conf/base.yaml` is STILL formatted incorrectly as a dictionary instead of a float, which means your data is likely IID (homogeneous) instead of non-IID, defeating the purpose of FedProx and SCAFFOLD.

---

## Fix #1: YAML Configuration - Alpha Parameter (CRITICAL)

### Current Code (WRONG):
```yaml
# Lines 16-18 in conf/base.yaml
alpha:
  0.1 # Dirichlet parameter for non-IID (lower = more heterogeneous)
  # 0.1 = very heterogeneous, 0.5 = moderate (better for seeing FedProx benefits), 1.0 = mild
```

### Fixed Code:
```yaml
# Correct format - alpha must be a float, not a dict!
alpha: 0.1  # Dirichlet parameter for non-IID (lower = more heterogeneous)
# 0.1 = very heterogeneous, 0.5 = moderate (better for seeing FedProx benefits), 1.0 = mild
```

**Impact**: This is causing your data to be IID instead of non-IID, which is why FedAvg performs as well as or better than FedProx/SCAFFOLD. **FIX THIS FIRST!**

---

## Fix #2: Initialize Parameters for All Strategies

The issue is that none of the strategies have initial parameters. We need to create a helper function to initialize parameters consistently.

### Create Initial Parameters Helper

Add this function to `server.py`:

```python
def get_initial_parameters(num_classes: int = 3):
    """
    Create initial parameters for the model.

    Args:
        num_classes: Number of output classes (default: 3)

    Returns:
        Parameters: Initial model parameters as Flower Parameters object
    """
    from flwr.common import ndarrays_to_parameters
    from model import Net

    model = Net(num_classes)
    # Get model parameters as numpy arrays
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(params)
```

### Update compare_strategies.py

Modify the strategy initialization to include initial parameters:

```python
def run_fedavg(cfg: DictConfig, trainloaders, validationloaders, testloader, client_fn):
    """Run FedAvg strategy."""
    print("\n" + "=" * 60)
    print("RUNNING FedAvg")
    print("=" * 60)

    # Import initial parameters helper
    from server import get_initial_parameters

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        initial_parameters=get_initial_parameters(cfg.num_classes)  # ADD THIS
    )

    # ... rest of function
```

Do the same for `run_fedprox()` - add `initial_parameters=get_initial_parameters(cfg.num_classes)` to all three strategy initializations (FedProx, AdaptiveFedProx, MultiSignalAdaptiveFedProx).

---

## Fix #3: SCAFFOLD Strategy - Multiple Issues

The modified `scaffold_strategy.py` has several issues that need fixing.

### Issue 3.1: Initialize Parameters Method

**Current Code (Line 64-67):**
```python
def initialize_parameters(
    self, client_manager: fl.server.client_manager.ClientManager
) -> Optional[Parameters]:
    return None  # WRONG - returns None
```

**Fixed Code:**
```python
def initialize_parameters(
    self, client_manager: fl.server.client_manager.ClientManager
) -> Optional[Parameters]:
    # Return initial parameters if provided during initialization
    if self.initial_parameters is not None:
        return self.initial_parameters

    # Otherwise, create default parameters
    from model import Net
    from flwr.common import ndarrays_to_parameters

    model = Net(num_classes=3)  # Hardcoded for this project
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    self.initial_parameters = ndarrays_to_parameters(params)
    return self.initial_parameters
```

### Issue 3.2: Add initial_parameters to __init__

**Current __init__ (Line 26-58)** is missing `initial_parameters` parameter.

**Add this parameter:**
```python
def __init__(
    self,
    min_fit_clients: int,
    min_available_clients: int,
    min_evaluate_clients: int,
    on_fit_config_fn=None,
    evaluate_fn=None,
    total_clients: Optional[int] = None,
    model: Optional[torch.nn.Module] = None,
    test_loader=None,
    criterion=None,
    config: Optional[dict] = None,
    initial_parameters: Optional[Parameters] = None,  # ADD THIS LINE
):
    # ... existing code ...

    self.current_parameters: Optional[List] = None
    self.c_global: Optional[List[np.ndarray]] = None
    self.client_control_variates: Dict[str, List[np.ndarray]] = {}

    # PyTorch evaluation setup
    self.model = model
    self.test_loader = test_loader
    self.criterion = criterion
    self.config = config or {}
    self.initial_parameters = initial_parameters  # ADD THIS LINE
```

### Issue 3.3: Update run_fedscaffold in compare_strategies.py

```python
def run_fedscaffold(cfg: DictConfig, trainloaders, validationloaders, testloader):
    """Run FedSCAFFOLD strategy with improved implementation."""
    print("\n" + "=" * 60)
    print("RUNNING FedSCAFFOLD (Improved)")
    print("=" * 60)

    from scaffold_strategy import FedScaffoldStrategy
    from cleint import generate_client_fn
    from server import get_initial_parameters  # ADD THIS

    server_lr = cfg.get('scaffold_server_lr', 1.0)
    print(f"  Server learning rate: {server_lr}")

    strategy = FedScaffoldStrategy(
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        total_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        initial_parameters=get_initial_parameters(cfg.num_classes)  # ADD THIS
    )

    # ... rest of function
```

---

## Fix #4: SCAFFOLD Gradient Correction (In model.py)

The current gradient correction in `train_scaffold()` at line 174 is:
```python
param.grad.data += (cg - ci)  # Current implementation
```

Based on the SCAFFOLD paper (Karimireddy et al., ICML 2020), the correct gradient correction should be:
```python
param.grad.data += (ci - cg)  # Revert to original
```

**Explanation**: The SCAFFOLD algorithm corrects the local gradient by adding `(c_i - c)` where `c` is the server control variate. This correction adjusts the local gradient to account for client drift.

However, since you reported that BOTH directions gave poor performance, the issue might be elsewhere (e.g., the data being IID due to the alpha formatting bug).

**Recommendation**:
1. First fix the alpha parameter formatting (Fix #1)
2. Then fix the initialization issues (Fixes #2 and #3)
3. Test with BOTH gradient correction directions to see which works better with proper non-IID data

---

## Fix #5: FedProx Performance Issues

### Issue: mu Values and Adaptive Adaptation

Looking at your current config:

```yaml
proximal_mu: 0.01  # Fixed mu
multi_signal_mu:
  base_mu: 0.001
  mu_max: 0.3
```

These values are reasonable, but FedProx won't show benefits if:
1. **Data is IID** (due to alpha formatting bug - FIX THIS FIRST!)
2. **Local epochs too low** - Try increasing to 15-20 to see more client drift
3. **mu too conservative** - Adaptive mu might not be adapting up enough

### Recommended Configuration Changes

```yaml
# For better FedProx/SCAFFOLD comparison
alpha: 0.1  # MUST be float, not dict!
config_fit:
  lr: 0.01
  momentum: 0.9
  local_epochs: 15  # Increase from 10 to create more drift
  max_grad_norm: 1.0

proximal_mu: 0.01  # Good starting point

multi_signal_mu:
  base_mu: 0.005  # Increase from 0.001 to allow faster adaptation
  mu_min: 0.0001
  mu_max: 0.2  # Reduce from 0.3 to prevent over-regularization
  smoothing_factor: 0.75  # Increase for more stability
  warmup_rounds: 5  # Give more time to stabilize
```

---

## Summary of All Required Changes

### Priority 1 (CRITICAL):
1. **Fix alpha parameter in base.yaml** - Change from dict to float (lines 16-18)
2. **Add get_initial_parameters() to server.py**
3. **Add initial_parameters to all strategies in compare_strategies.py**

### Priority 2 (Important):
4. **Fix SCAFFOLD __init__ to accept initial_parameters**
5. **Fix SCAFFOLD initialize_parameters() method**
6. **Test both gradient correction directions** in model.py after fixing data

### Priority 3 (Optimization):
7. **Increase local_epochs** to 15 in config
8. **Increase base_mu** to 0.005 for faster adaptation
9. **Run diagnostic script** to verify data heterogeneity

---

## Testing Procedure

After applying all fixes:

1. **Verify alpha is parsed correctly**:
   ```bash
   python diagnose_data.py --config-name=base
   ```
   Check that alpha shows as `0.1` (float) not a dict, and heterogeneity score is > 20%.

2. **Run comparison**:
   ```bash
   python compare_strategies.py
   ```

3. **Expected Results** (with alpha=0.1, local_epochs=15):
   - **FedAvg**: 55-65% accuracy, slow/unstable convergence
   - **FedProx**: 65-75% accuracy, steady improvement
   - **SCAFFOLD**: 70-80% accuracy, fastest convergence

If you still see FedAvg outperforming others after these fixes, the issue is likely:
- Control variate update formula in SCAFFOLD
- Gradient correction direction
- Adaptive mu not adapting properly

---

## References

- SCAFFOLD Paper: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020
- Flower Documentation: https://flower.dev/docs/
- FedProx Paper: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020

---

## Change Log

**2026-01-11**: Initial documentation of critical fixes for FedAvg, FedProx, and SCAFFOLD implementation issues.
