import nfl_data_py as nfl
import importlib
import subprocess
import sys

# Ensure dependencies are installed
pkg = 'nfl_data_py'
try:
    importlib.import_module(pkg)
except ImportError:
    print(f"! {pkg} not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    importlib.invalidate_caches()

# Load injury data for the 2024 season
# This returns a pandas DataFrame
injuries_2024 = nfl.import_injuries([2024])

# Display the first few rows
print(injuries_2024.head())

# Optional: Save to a CSV file for your analysis
injuries_2024.to_csv('nfl_injuries_2024.csv', index=False)