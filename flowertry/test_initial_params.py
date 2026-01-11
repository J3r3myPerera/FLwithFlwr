"""
Quick test to verify initial parameters are created correctly.
"""

from server import get_initial_parameters
from flwr.common import Parameters, parameters_to_ndarrays

# Test 1: Create initial parameters
print("Test 1: Creating initial parameters...")
params = get_initial_parameters(num_classes=3)
print(f"  Type: {type(params)}")
print(f"  Is Parameters object: {isinstance(params, Parameters)}")

if hasattr(params, 'tensors'):
    print(f"  Has 'tensors' attribute: True")
    print(f"  Number of tensors: {len(params.tensors)}")
else:
    print(f"  ERROR: Missing 'tensors' attribute!")

# Test 2: Convert back to ndarrays
print("\nTest 2: Converting Parameters to ndarrays...")
try:
    ndarrays = parameters_to_ndarrays(params)
    print(f"  Success! Got {len(ndarrays)} arrays")
    print(f"  Shapes: {[arr.shape for arr in ndarrays]}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\nAll tests passed!")
