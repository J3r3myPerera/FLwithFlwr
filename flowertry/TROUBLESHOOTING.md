# Troubleshooting Guide

## Common Errors and Solutions

### 1. YAML Parsing Errors

**Error**: `yaml.scanner.ScannerError` or `yaml.parser.ParserError`

**Cause**: Invalid YAML syntax (e.g., commented lines with incorrect indentation)

**Solution**: 
- Ensure all comments start with `#` at the beginning of the line
- Don't mix commented and active config lines with different indentation
- Fixed in `conf/base.yaml` - removed problematic commented lines

### 2. Import Errors for `adaptive_fedprox`

**Error**: `ModuleNotFoundError: No module named 'adaptive_fedprox'`

**Cause**: Adaptive FedProx is enabled but module doesn't exist or has errors

**Solution**: 
- Set `adaptive_mu.enabled: false` in `conf/base.yaml` (already fixed)
- Or ensure `adaptive_fedprox.py` exists and is error-free

### 3. AttributeError: 'dict' object has no attribute 'parameters'

**Error**: `AttributeError: 'dict' object has no attribute 'parameters'`

**Cause**: `configure_fit()` returning `(client, config_dict)` instead of `(client, FitIns)`

**Solution**: Already fixed in `fedprox_scaffold_strategy.py` and `scaffold_strategy.py`
- Changed to return `FitIns(parameters, config)` instead of just `config`

### 4. TypeError: list indices must be integers or slices, not list

**Error**: `TypeError: list indices must be integers or slices, not list`

**Cause**: Trying to use list indexing on numpy arrays incorrectly

**Solution**: Already fixed in `dataset.py`
- Converted `trainval_indices` and `client_subset_indices` to numpy arrays

### 5. Control Variate Shape Mismatch

**Error**: `RuntimeError: shape mismatch` or `ValueError: operands could not be broadcast`

**Cause**: Control variates (c_global, c_client) have wrong shape compared to model parameters

**Solution**: 
- Ensure control variates are initialized with same shape as parameters
- Check that parameter order matches between client and server

### 6. Proximal Term Not Working

**Symptoms**: FedProx performs same as FedAvg

**Causes**:
- `proximal_mu` too small (try 0.2-0.5)
- Too many local epochs (use 3-5)
- Adaptive mode enabled with low base_mu (disable or increase)

**Solution**: 
- Set `adaptive_mu.enabled: false`
- Use `proximal_mu: 0.3` or higher
- Reduce `local_epochs` to 3-5

### 7. SCAFFOLD Control Variates Not Updating

**Symptoms**: Hybrid/SCAFFOLD not improving over FedAvg

**Causes**:
- Control variates not being passed correctly
- `scaffold_lr` too small
- Control variate update formula incorrect

**Solution**: 
- Check that `c_global` and `c_client` are in config
- Use `scaffold_lr: 1.0`
- Verify control variate updates in `_fit_hybrid()`

### 8. Random Import Error

**Error**: `NameError: name 'random' is not defined`

**Cause**: `import random` inside function instead of at module level

**Solution**: Already fixed - moved `import random` to top of `cleint.py`

### 9. Hydra Configuration Errors

**Error**: `omegaconf.errors.ConfigKeyError` or `hydra.errors.ConfigCompositionException`

**Causes**:
- Invalid YAML syntax
- Missing required config keys
- Type mismatches

**Solution**: 
- Validate YAML syntax
- Check all required keys are present
- Ensure types match (bool, int, float, str)

### 10. Flower Simulation Errors

**Error**: `flwr.common.errors.FlwrException` or simulation hangs

**Causes**:
- Client function returning wrong format
- Strategy not compatible with client
- Resource constraints

**Solution**:
- Verify client returns `(parameters, num_examples, metrics)`
- Check strategy configuration matches client implementation
- Reduce `num_clients` or `num_clients_per_round_fit` if resource-limited

## Verification Steps

### 1. Check Configuration

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('conf/base.yaml'))"
```

### 2. Test Imports

```python
# Test all imports
from dataset import prepare_dataset
from cleint import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
from fedprox_scaffold_strategy import FedProxScaffoldStrategy
from scaffold_strategy import FedScaffoldStrategy
```

### 3. Verify FedProx is Active

Look for debug prints:
- `[FedProx] Training with proximal_mu=0.3000`
- `[Hybrid] Training with proximal_mu=...`

### 4. Check Data Distribution

Verify non-IID is working:
- Should see "Data Distribution: Non-IID"
- Should see class distribution for first 5 clients

## Current Configuration Status

✅ **Fixed Issues**:
- YAML syntax errors (removed problematic comments)
- `random` import moved to top level
- `FitIns` return type fixed
- Adaptive mode disabled (using fixed mu)
- Proximal mu increased to 0.3
- Local epochs reduced to 5

✅ **Expected Behavior**:
- FedProx uses `proximal_mu: 0.3`
- Hybrid uses `proximal_mu: 0.25` + SCAFFOLD
- Data is very heterogeneous (`alpha: 0.1`)

## If Errors Persist

1. **Check terminal output** - Look for full traceback
2. **Run with verbose logging**:
   ```bash
   python compare_strategies.py --verbose
   ```
3. **Test individual components**:
   ```python
   # Test dataset preparation
   python -c "from dataset import prepare_dataset; prepare_dataset(10, 32, iid=False, alpha=0.1)"
   
   # Test client creation
   python -c "from cleint import FlowerClient; print('OK')"
   ```
4. **Check Flower version compatibility**:
   ```bash
   pip show flwr
   # Should be >= 1.0.0
   ```

## Debug Mode

To enable more verbose output, add to your code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set in config:
```yaml
hydra:
  verbose: true
```
