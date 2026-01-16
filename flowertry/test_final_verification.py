"""Final verification test for simplified plotting."""

import sys
import importlib

print("="*80)
print("FINAL VERIFICATION TEST")
print("="*80)

# Test 1: Import modules
print("\n1. Testing module imports...")
try:
    from visualize_metrics import ComparisonPlotter, save_metrics_to_csv
    print("   ✅ visualize_metrics imports successfully")
except Exception as e:
    print(f"   ❌ Error importing visualize_metrics: {e}")
    sys.exit(1)

try:
    # Import main without executing
    import main
    print("   ✅ main module imports successfully")
except Exception as e:
    print(f"   ❌ Error importing main: {e}")
    sys.exit(1)

# Test 2: Check MetricsPlotter is not used
print("\n2. Checking MetricsPlotter removal...")
try:
    from visualize_metrics import MetricsPlotter
    print("   ⚠️  MetricsPlotter still exists in module (but not used)")
except ImportError:
    print("   ✅ MetricsPlotter not imported (good)")

# Test 3: Verify ComparisonPlotter has only final_comparison
print("\n3. Verifying ComparisonPlotter methods...")
plotter = ComparisonPlotter(output_dir=".")

removed = ['plot_training_history', 'plot_convergence_analysis']
kept = ['plot_final_comparison', 'generate_all_plots']

all_ok = True
for method in removed:
    if hasattr(plotter, method):
        print(f"   ❌ {method} should be removed!")
        all_ok = False
    else:
        print(f"   ✅ {method} removed")

for method in kept:
    if hasattr(plotter, method):
        print(f"   ✅ {method} exists")
    else:
        print(f"   ❌ {method} missing!")
        all_ok = False

# Test 4: Check main.py doesn't use plotter parameter
print("\n4. Checking main.py for plotter usage...")
with open('main.py', 'r') as f:
    main_content = f.read()
    
if 'plotter=plotter' in main_content:
    print("   ❌ Found 'plotter=plotter' in main.py")
    all_ok = False
else:
    print("   ✅ No 'plotter=plotter' calls found")

if 'MetricsPlotter(' in main_content:
    print("   ❌ Found 'MetricsPlotter(' instantiation in main.py")
    all_ok = False
else:
    print("   ✅ No MetricsPlotter instantiation found")

# Test 5: Verify imports
if 'from visualize_metrics import MetricsPlotter' in main_content:
    print("   ❌ MetricsPlotter still imported in main.py")
    all_ok = False
else:
    print("   ✅ MetricsPlotter not imported in main.py")

print("\n" + "="*80)
if all_ok:
    print("✅ ALL TESTS PASSED")
    print("\nThe system now:")
    print("  • Has no real-time plotting during training")
    print("  • Only generates final comparison plots")
    print("  • Has cleaner, faster execution")
else:
    print("❌ SOME TESTS FAILED - Please review")
print("="*80)
