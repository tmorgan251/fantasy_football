# Fantasy Football Analysis: Optimizing Draft and Injury Decisions

**Authors:** Tristan Morgan and David Brand  
**Course:** Michigan Applied Data Science - Milestone Project I

## Overview

This project analyzes fantasy football data to identify the most meaningful elements for making optimized decisions. We examine three key areas:

1. **Draft Value Analysis**: Using autodrafted teams as a baseline, we quantify the value of human draft decisions and identify optimal draft strategies.
2. **Injury Impact Analysis**: We measure how injuries affect player performance by position, injury type, and status to inform start/sit decisions.
3. **Baseline Calculations**: We establish four baselines (Draft, Waiver Wire, Trade, Start/Sit) for performance comparison.

## Key Findings

- **Draft Advantage**: Human drafters show measurable advantages over autodraft in early rounds, with diminishing returns in later rounds.
- **Injury Impact**: Certain injury types (e.g., hamstring) have position-specific impacts that significantly affect point differentials.
- **Decision Support**: Our analysis provides actionable insights for draft strategy and weekly lineup decisions.

## Data Sources

- **ESPN Fantasy Football API**: Historical league data (2021-2024) including drafts, lineups, and transactions
- **NFL Injury Database (nfl_data_py)**: Player injury reports with type, status, and timing

## Project Structure

```
fantasy_football/
├── Fantasy_Football_Analysis.ipynb  # Main analysis notebook
├── src/
│   ├── data_fetchers/
│   │   └── data_collector.py        # ESPN API data collection
│   └── analysis/
│       ├── baseline_calculator.py   # Baseline calculations
│       ├── draft_value_analyzer.py  # Draft value analysis
│       └── injury_analyzer.py        # Injury impact analysis
├── data/
│   ├── raw/                         # Raw collected data
│   └── preprocessed/                # Processed data files
└── README.md                        # This file
```

## Setup

### Prerequisites

```bash
pip install pandas matplotlib seaborn espn-api nfl-data-py
```

### Running the Analysis

1. Open `Fantasy_Football_Analysis.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially - data collection runs in the background
3. All analysis is modularized into classes for easy execution

## Python Syntax Guide

This project uses some Python syntax features that may be unfamiliar. Here's a quick guide:

### Return Type Annotations (`-> Type`)

Return type annotations document what type a function returns. They appear after the function parameters and before the colon:

```python
def compute_optimal_startable_points(self, lineups: pd.DataFrame) -> pd.DataFrame:
    """Compute optimal startable points for each team-week."""
    # ... function code ...
    return result_dataframe
```

**Why use them?**
- **IDE Support**: Enables autocomplete and type checking in IDEs (VS Code, PyCharm, etc.)
- **Self-Documentation**: Makes it immediately clear what a function returns without reading the code
- **Type Checking**: Tools like `mypy` can verify you're using return values correctly
- **Prevents Bugs**: Especially important for functions returning `None` (side effects only) or tuples (need to unpack)

**Common patterns in this codebase:**
- `-> pd.DataFrame`: Returns a pandas DataFrame
- `-> None`: Function performs side effects but doesn't return a value
- `-> Tuple[pd.DataFrame, pd.DataFrame]`: Returns a tuple of two DataFrames (must unpack: `df1, df2 = func()`)
- `-> List[Dict[str, Any]]`: Returns a list of dictionaries
- `-> bool`: Returns True or False (predicate functions)

**Example from code:**
```python
def load_multi_season_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Without the annotation, you wouldn't know you need to unpack:
    draft, lineups = analyzer.load_multi_season_data()  # ✓ Correct
    result = analyzer.load_multi_season_data()          # ✗ Wrong - result is a tuple!
```

### Decorators (`@`)

Decorators are special syntax that modify or wrap functions. They appear above function definitions with the `@` symbol:

```python
@staticmethod
def normalize_player_name(name: str) -> str:
    """Normalize player name for consistent matching."""
    # ... function code ...
```

**Why use them?**
- **Code Reuse**: Avoids writing boilerplate code
- **Clear Intent**: Makes function behavior explicit (e.g., `@staticmethod` means "doesn't need instance")
- **Automatic Generation**: Some decorators automatically generate code (e.g., `@dataclass` creates `__init__`, `__repr__`, etc.)

**Common decorators in this codebase:**

1. **`@staticmethod`**: Method doesn't need access to `self` (the instance). Can be called directly on the class:
   ```python
   @staticmethod
   def normalize_player_name(name: str) -> str:
       # Can call: DraftValueAnalyzer.normalize_player_name("John Smith")
       # Without creating an instance first
   ```

2. **`@dataclass`**: Automatically generates special methods (`__init__`, `__repr__`, `__eq__`, etc.) for a class:
   ```python
   @dataclass
   class State:
       attempts: int = 0
       targets: int = 0
       delay: float = 0.10
   
   # Instead of manually writing:
   # def __init__(self, attempts=0, targets=0, delay=0.10):
   #     self.attempts = attempts
   #     self.targets = targets
   #     self.delay = delay
   ```

**Note**: Return type annotations and decorators are optional in Python - the code works without them, but they improve code quality, IDE support, and make the code more maintainable.

## Visualization Parameters Reference

All visualization methods support optional parameters for customization. Here's a comprehensive reference:

### Draft Value Visualizations

#### `plot_team_total_valid_points_distribution()`

Compare team total points distributions between autodrafted and non-autodrafted teams.

**Parameters:**
- `draft_with_valid` (DataFrame, optional): Draft data with valid points (uses computed if None)
- `fully_autodraft_only` (bool, optional): 
  - `True`: Show only fully autodrafted teams
  - `False`: Show only non-autodrafted teams
  - `None`: Show all teams
- `bins` (int, default=20): Number of histogram bins
- `title` (str, optional): Custom plot title (auto-generated if None)

**Returns:** matplotlib Figure object

---

#### `plot_expected_by_pick()`

Show expected fantasy points by draft pick position with variance bands.

**Parameters:**
- `expected_by_pick_pooled` (DataFrame, optional): Pooled expected values (uses computed if None)
- `expected_by_pick_poly` (DataFrame, optional): Polynomial fit (uses computed if None)
- `show_variance` (bool, default=True): Show variance bands (25-75% and 10-90% ranges)
- `zoom_first_25` (bool, default=False): Create additional zoomed plot of first 25 picks

**Returns:** matplotlib Figure object

---

#### `plot_per_season_expected_values()`

Display expected values by pick, broken down by season.

**Parameters:**
- `draft_with_valid` (DataFrame, optional): Draft data (uses computed if None)
- `zoom_first_25` (bool, default=True): Create zoomed plots of first 25 picks per season

**Returns:** matplotlib Figure object

---

#### `plot_human_advantage_by_year()`

Compare human draft advantage vs autodraft by year.

**Parameters:**
- `draft_scored` (DataFrame, optional): Scored draft data (uses computed if None)
- `baseline` (str, default="Poly"): Baseline to use - "Poly", "Pooled", or "Year"

**Returns:** DataFrame with year-by-year advantage statistics

---

#### `plot_human_advantage_by_round()`

Compare human draft advantage vs autodraft by round.

**Parameters:**
- `draft_scored` (DataFrame, optional): Scored draft data (uses computed if None)
- `baseline` (str, default="Poly"): Baseline to use - "Poly", "Pooled", or "Year"

**Returns:** DataFrame with round-by-round advantage statistics

---

#### `plot_human_advantage_by_position()`

Compare human draft advantage vs autodraft by position.

**Parameters:**
- `draft_scored` (DataFrame, optional): Scored draft data (uses computed if None)
- `baseline` (str, default="Poly"): Baseline to use - "Poly", "Pooled", or "Year"

**Returns:** DataFrame with position-by-position advantage statistics

---

### Injury Impact Visualizations

#### `plot_heatmap_type_position()`

Heatmap showing average point differential (actual - projected) by injury type and position.

**Parameters:**
- `save_path` (str, optional): Path to save figure (e.g., "figures/heatmap_type_position.png")
- `min_population` (int, default=30): Minimum observations per cell for statistical reliability
- `show_sample_sizes` (bool, default=True): Show sample size (n=X) in each cell

**Returns:** matplotlib Figure object

---

#### `plot_heatmap_type_status()`

Heatmap showing average point differential by injury type and status (No Injury, Questionable, Doubtful, Out).

**Parameters:**
- `save_path` (str, optional): Path to save figure
- `min_population` (int, default=30): Minimum observations per cell
- `show_sample_sizes` (bool, default=True): Show sample size in each cell

**Returns:** matplotlib Figure object

---

#### `plot_heatmap_position_status()`

Heatmap showing average point differential by position and injury status.

**Parameters:**
- `save_path` (str, optional): Path to save figure
- `min_population` (int, default=30): Minimum observations per cell
- `show_sample_sizes` (bool, default=True): Show sample size in each cell

**Returns:** matplotlib Figure object

---

#### `plot_point_differential_distribution()`

Distribution plots showing point differential distributions for injured vs healthy players, broken down by status, position, and injury type.

**Parameters:**
- `save_path` (str, optional): Path to save figure

**Returns:** matplotlib Figure object

---

#### `plot_top_injury_impacts()`

Bar chart showing the top N best and worst performing injury combinations.

**Parameters:**
- `agg_stats` (DataFrame, optional): Aggregated statistics (uses computed if None)
- `top_n` (int, default=10): Number of top/bottom combinations to show
- `save_path` (str, optional): Path to save figure

**Returns:** matplotlib Figure object

---

## Usage Examples

### Draft Value Analysis

```python
from src.analysis.draft_value_analyzer import DraftValueAnalyzer

# Initialize analyzer
analyzer = DraftValueAnalyzer(
    raw_base="data/raw/espn",
    out_dir="data/preprocessed",
    years=range(2021, 2025),
    verbose=False
)

# Run full pipeline
draft_with_valid = analyzer.run_full_pipeline(
    clean_data=True,
    filter_standard=True,
    filter_scoring=True,
    filter_draft_length=True,
    compute_optimal=True,
    save_intermediate=True
)

# Create visualizations
analyzer.plot_team_total_valid_points_distribution(fully_autodraft_only=True)
analyzer.plot_expected_by_pick(show_variance=True, zoom_first_25=True)
analyzer.plot_human_advantage_by_round(baseline="Poly")
```

### Injury Impact Analysis

```python
from src.analysis.injury_analyzer import InjuryAnalyzer

# Initialize analyzer
injury_analyzer = InjuryAnalyzer(
    injury_data_path="data/raw/injuries/nfl_injuries_2021_2024.csv",
    lineup_data_base="data/raw/espn",
    years=range(2021, 2025)
)

# Run full analysis
agg_stats = injury_analyzer.run_full_analysis()

# Create visualizations
injury_analyzer.plot_heatmap_type_position(min_population=30, show_sample_sizes=True)
injury_analyzer.plot_point_differential_distribution()
injury_analyzer.plot_top_injury_impacts(top_n=10)
```

## Notes

- All visualization methods return matplotlib Figure objects (except those that return DataFrames)
- Use `plt.show()` to display figures or `fig.savefig()` to save them
- Data collection runs in the background - you can continue working while it executes
- All analysis classes support saving intermediate results for faster re-runs

## Limitations

1. **Data Scope**: Analysis limited to ESPN Fantasy Football leagues (2021-2024). Results may not generalize to other platforms or scoring systems.
2. **Injury Data Quality**: Injury reports may not capture all injuries or may include non-injury designations (e.g., rest days).
3. **Projection Accuracy**: Analysis assumes ESPN projections are reasonable baselines. Actual projection quality may vary.
4. **Sample Size**: Some injury type/position combinations have limited data (<30 observations), requiring population thresholds.

## Future Work

1. **Machine Learning Models**: Develop predictive models that combine draft position, injury status, and historical performance to optimize draft and lineup decisions.
2. **Real-Time Decision Support**: Create a tool that provides real-time recommendations during drafts and weekly lineup decisions based on this analysis.
3. **Expanded Data Sources**: Incorporate additional data sources (e.g., weather, opponent strength, player snap counts) to improve injury impact predictions.
4. **Longitudinal Analysis**: Track how injury impacts change over multiple weeks to identify recovery patterns and optimal return-to-play timing.
5. **Platform Comparison**: Extend analysis to other fantasy platforms (Yahoo, Sleeper) to validate generalizability of findings.

## Disclaimer

**Educational Purpose Only**: This analysis and all associated code, data, and visualizations are provided solely for educational and research purposes as part of an academic project. The methods, findings, and insights presented here are intended for learning and understanding fantasy football analytics.

**Not for Gambling**: This work should **NOT** be used for gambling, betting, or any form of wagering. Fantasy sports can involve real money in some contexts, and this analysis is not intended to provide gambling advice or to be used in any gambling-related activities.

**No Warranty**: The authors make no guarantees about the accuracy, completeness, or applicability of these findings. Past performance does not guarantee future results, and fantasy football outcomes are inherently uncertain.

**Responsible Use**: If you choose to use these insights for fantasy football decisions, please do so responsibly and within the rules and regulations of your jurisdiction.

---

*This project is part of the Michigan Applied Data Science program and is intended for academic learning and research purposes only.*
