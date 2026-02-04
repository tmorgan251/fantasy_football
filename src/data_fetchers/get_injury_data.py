import nfl_data_py as nfl

# Load injury data for the 2024 season
# This returns a pandas DataFrame
injuries_2024 = nfl.import_injuries([2024])

# Display the first few rows
print(injuries_2024.head())

# Optional: Save to a CSV file for your analysis
injuries_2024.to_csv('nfl_injuries_2024.csv', index=False)