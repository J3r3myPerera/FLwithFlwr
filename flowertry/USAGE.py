#!/usr/bin/env python3
"""
Quick Start: Plot Round-by-Round Training Progression

This script demonstrates how to use the plotting functionality
for your Federated Learning simulations.
"""

import os
import sys

def print_usage():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Round-by-Round FL Training Visualization                     ║
║                                                                       ║
║  Plots training progression graphs like the one in your example     ║
╚══════════════════════════════════════════════════════════════════════╝

📋 WHAT WAS FIXED:
------------------
1. Fixed history collection in main.py (lines 286-318)
   - Now properly extracts round-by-round metrics from Flower's History object
   - Handles tuple format: (num_examples, metrics_dict)
   - Sorts rounds to ensure proper ordering

2. Created plot_round_progression.py
   - Reads comparison_results.json
   - Generates 2×2 grid plot (Training Loss, R², MAPE, Final Metrics)
   - Optional detailed 3×2 plot with all metrics
   - Optional summary tables

📊 USAGE:
---------

1️⃣  Run a simulation (with the fix):
    python main.py --config-name "base copy 2"

2️⃣  Plot the results:
    python plot_round_progression.py

    This will automatically find and plot the latest simulation.

3️⃣  Plot a specific simulation:
    python plot_round_progression.py --output-dir outputs/2026-01-19/21-52-59

4️⃣  With detailed metrics and summary:
    python plot_round_progression.py --detailed --summary

5️⃣  Test with mock data:
    python test_plotting.py
    python plot_round_progression.py --output-dir outputs/test_plotting

🎨 OUTPUT:
----------
Creates a plot similar to your attached image:

    ┌──────────────────┬──────────────────┐
    │  Training Loss   │  R² Score        │
    │  (decreasing)    │  (increasing)    │
    ├──────────────────┼──────────────────┤
    │  MAPE Progress   │  Final Metrics   │
    │  (with target)   │  (bar chart)     │
    └──────────────────┴──────────────────┘

Files created:
• training_progression.png          (main plot)
• detailed_metrics_progression.png  (if --detailed)

📁 VERIFY HISTORY IS COLLECTED:
--------------------------------
Check if a simulation has history data:

    cd outputs/2026-01-19/21-52-59
    cat comparison_results.json | python3 -c "
    import sys, json
    d = json.load(sys.stdin)
    for s, data in d.items():
        rounds = len(data.get('history', {}).get('rounds', []))
        print(f'{s}: {rounds} rounds')
    "

Expected (after fix):
    fedavg: 30 rounds
    fedprox: 30 rounds
    scaffold: 30 rounds
    hybrid: 30 rounds

⚠️  IMPORTANT NOTES:
--------------------
• Simulations run BEFORE the fix won't have history data
• You need to run a NEW simulation to test the plotting
• The config must have compare_all: true to compare strategies
• History is stored in comparison_results.json under each strategy

🔧 COMMAND LINE OPTIONS:
------------------------
    --output-dir, -o    Path to output directory (default: latest)
    --detailed, -d      Generate detailed 3×2 metrics plot
    --summary, -s       Print round-by-round summary table

📖 EXAMPLES:
------------
# Plot latest simulation with all options
python plot_round_progression.py --detailed --summary

# Plot a specific simulation
python plot_round_progression.py -o outputs/2026-01-19/21-52-59 -d -s

# Quick test with mock data
python test_plotting.py && python plot_round_progression.py -o outputs/test_plotting

🐛 TROUBLESHOOTING:
-------------------
Problem: "No history data found"
→ Solution: Run a new simulation (old ones don't have history)

Problem: "No comparison_results.json found"
→ Solution: Ensure compare_all: true in your config

Problem: Empty plots
→ Solution: Check that evaluate() returns metrics in server.py

📚 FILES INVOLVED:
------------------
Modified:
• main.py (lines 286-318) - Fixed history extraction

Created:
• plot_round_progression.py - Main plotting script
• test_plotting.py - Test data generator
• README_PLOTTING.md - Detailed documentation
• USAGE.py (this file) - Quick reference

✅ NEXT STEPS:
--------------
1. Run: python test_plotting.py
2. Run: python plot_round_progression.py -o outputs/test_plotting -d -s
3. Verify the plot looks like your example image
4. Run a real simulation: python main.py --config-name "base copy 2"
5. Plot results: python plot_round_progression.py --detailed --summary
6. Use the plots in your reports! 🎉
""")

if __name__ == "__main__":
    print_usage()
    
    print("\n" + "="*70)
    response = input("\n🚀 Want to run a test now? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\n1. Creating test data...")
        os.system("python test_plotting.py")
        
        print("\n2. Generating plots...")
        os.system("python plot_round_progression.py --output-dir outputs/test_plotting --summary")
        
        print("\n✅ Check outputs/test_plotting/training_progression.png")
        print("   This is what your real simulation plots will look like!")
    else:
        print("\n👋 Run 'python USAGE.py' anytime to see this help again!")
