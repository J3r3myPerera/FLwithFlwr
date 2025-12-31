#!/usr/bin/env python3
"""
Run all three FL strategies (FedAvg, FedProx, SCAFFOLD) and create comparison.

Usage:
    python run_comparison.py                    # Run all experiments
    python run_comparison.py --visualize-only   # Only visualize existing results
    python run_comparison.py --iid              # Run with IID data
    python run_comparison.py --extreme          # Run with extreme non-IID data
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import glob
import argparse

from visualize import create_comparison_report, plot_full_comparison


def find_latest_results(strategy: str, partition: str = "dirichlet") -> str:
    """Find the most recent results.pkl for a given strategy."""
    outputs_dir = Path("outputs")
    
    # Search for results files
    pattern = str(outputs_dir / "*" / "*" / "results.pkl")
    result_files = glob.glob(pattern)
    
    # Filter by strategy in config
    matching = []
    for f in result_files:
        try:
            import pickle
            with open(f, 'rb') as fp:
                results = pickle.load(fp)
            if results['config'].get('strategy') == strategy:
                if results['config'].get('partition_type') == partition:
                    matching.append((f, os.path.getmtime(f)))
        except:
            pass
    
    if matching:
        # Return most recent
        matching.sort(key=lambda x: -x[1])
        return matching[0][0]
    
    return None


def run_experiment(config_name: str) -> str:
    """Run a single FL experiment and return the results path."""
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {config_name}")
    print(f"{'='*70}\n")
    
    cmd = ["python", "main.py", f"--config-name={config_name}"]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error running {config_name}")
        return None
    
    # Find the most recent output directory
    outputs_dir = Path("outputs")
    subdirs = list(outputs_dir.glob("*/*"))
    if subdirs:
        latest = max(subdirs, key=os.path.getmtime)
        results_path = latest / "results.pkl"
        if results_path.exists():
            return str(results_path)
    
    return None


def run_all_experiments(partition_type: str = "noniid") -> dict:
    """Run FedAvg, FedProx, and SCAFFOLD experiments."""
    
    if partition_type == "iid":
        configs = {
            'FedAvg (IID)': 'iid_fedavg',
        }
        # For IID, just run FedAvg as baseline
        print("\nRunning IID baseline experiment...")
    elif partition_type == "extreme":
        configs = {
            'FedAvg (Extreme Non-IID)': 'noniid_fedavg',  # Will modify alpha
            'FedProx (Extreme Non-IID)': 'extreme_noniid_fedprox',
            'SCAFFOLD (Extreme Non-IID)': 'extreme_noniid_scaffold',
        }
    else:
        configs = {
            'FedAvg (Non-IID)': 'noniid_fedavg',
            'FedProx (Non-IID)': 'noniid_fedprox',
            'SCAFFOLD (Non-IID)': 'noniid_scaffold',
        }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\n{'#'*70}")
        print(f"# Running: {name}")
        print(f"{'#'*70}")
        
        results_path = run_experiment(config)
        if results_path:
            results[name] = results_path
            print(f"✓ {name} completed: {results_path}")
        else:
            print(f"✗ {name} failed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run FL strategy comparison')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only create visualization from existing results')
    parser.add_argument('--iid', action='store_true',
                       help='Run with IID data partitioning')
    parser.add_argument('--extreme', action='store_true',
                       help='Run with extreme non-IID (alpha=0.1)')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison')
    
    args = parser.parse_args()
    
    # Determine partition type
    if args.iid:
        partition = "iid"
    elif args.extreme:
        partition = "extreme"
    else:
        partition = "noniid"
    
    if args.visualize_only:
        # Find existing results
        print("\nSearching for existing results...")
        
        results = {}
        
        # Try to find results for each strategy
        for strategy, name in [
            ('fedavg', f'FedAvg ({partition})'),
            ('fedprox', f'FedProx ({partition})'),
            ('scaffold', f'SCAFFOLD ({partition})')
        ]:
            path = find_latest_results(strategy, 
                                       "iid" if partition == "iid" else "dirichlet")
            if path:
                results[name] = path
                print(f"  Found: {name} -> {path}")
        
        if not results:
            print("\nNo results found! Run experiments first:")
            print("  python run_comparison.py")
            return
    else:
        # Run all experiments
        results = run_all_experiments(partition)
    
    if not results:
        print("\nNo results to visualize!")
        return
    
    # Create comparison visualization
    print(f"\n{'='*70}")
    print("CREATING COMPARISON VISUALIZATION")
    print(f"{'='*70}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    
    create_comparison_report(results, output_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {output_dir}/")
    print("\nFiles generated:")
    print("  - full_comparison.png      (side-by-side accuracy + bar chart)")
    print("  - convergence_comparison.png (accuracy over rounds)")
    print("  - comparison_summary.txt   (detailed text report)")


if __name__ == "__main__":
    main()
