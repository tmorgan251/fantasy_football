"""
Simple script to run all baseline calculations.

Make sure you have:
1. Collected data with Position column (re-run data_collector if needed)
2. CSV files in data/raw/espn/ directory
"""

import sys
import os

# Add src to path if needed (for imports to work)
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get src directory (this file is in src/analysis/)
src_dir = os.path.dirname(script_dir)
# Add to Python path if not already there (allows importing from analysis module)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from analysis.baseline_calculator import BaselineCalculator

def calculate_all_baselines(verbose=True):
    """
    Calculate all baselines and return results.
    
    Args:
        verbose: If True, print progress and results
    
    Returns:
        dict with all baseline results
    """
    if verbose:
        print("=" * 60)
        print("Fantasy Football Baseline Calculator")
        print("=" * 60)
    
    # Initialize calculator
    calculator = BaselineCalculator()
    
    results = {}
    
    if verbose:
        print("\n1. Calculating Draft Baseline...")
        print("-" * 60)
    draft_baseline = calculator.calculate_draft_baseline()
    results['draft'] = draft_baseline
    if verbose:
        if len(draft_baseline) > 0:
            print(f"Found {len(draft_baseline)} players from 100% auto-drafted leagues")
            print("\nTop 10 Players by Total Points:")
            print(draft_baseline.head(10).to_string(index=False))
        else:
            print("No 100% auto-drafted leagues found.")
    
    if verbose:
        print("\n2. Calculating Trade Baseline...")
        print("-" * 60)
    trade_baseline = calculator.calculate_trade_baseline()
    results['trade'] = trade_baseline
    if verbose:
        print(f"Trade Baseline: {trade_baseline['description']}")
        print(f"Total trades in dataset: {trade_baseline['trade_count']}")
    
    if verbose:
        print("\n3. Calculating Waiver Wire Baseline...")
        print("-" * 60)
    waiver_baseline = calculator.calculate_waiver_baseline()
    results['waiver'] = waiver_baseline
    if verbose:
        if len(waiver_baseline) > 0:
            print("Average Net Points by Position:")
            print(waiver_baseline.to_string(index=False))
        else:
            print("No waiver transactions with position data found.")
            print("Make sure Position column exists in lineup_data.csv")
    
    if verbose:
        print("\n4. Calculating Start/Sit Baseline...")
        print("-" * 60)
    startsit_baseline = calculator.calculate_startsit_baseline()
    results['startsit'] = startsit_baseline
    if verbose:
        if len(startsit_baseline) > 0:
            print(f"Calculated optimal points for {len(startsit_baseline)} team-weeks")
            print("\nSample Results (first 10):")
            print(startsit_baseline.head(10).to_string(index=False))
            
            # Summary statistics
            print("\nSummary Statistics:")
            print(f"  Mean Optimal Points: {startsit_baseline['Optimal_Points'].mean():.2f}")
            print(f"  Median Optimal Points: {startsit_baseline['Optimal_Points'].median():.2f}")
            print(f"  Min Optimal Points: {startsit_baseline['Optimal_Points'].min():.2f}")
            print(f"  Max Optimal Points: {startsit_baseline['Optimal_Points'].max():.2f}")
        else:
            print("No lineup data with position information found.")
            print("Make sure Position column exists in lineup_data.csv")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Baseline calculations complete!")
        print("=" * 60)
    
    return results

def main():
    """Main function for command-line usage."""
    calculate_all_baselines(verbose=True)

if __name__ == "__main__":
    main()

