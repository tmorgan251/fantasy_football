# Baseline Implementation Plan

## Import Issue Fix

**Problem**: You're trying to import `from src.data_fetchers.data_grab import FantasyDataCollector` but the file is named `data_collector.py`.

**Solution**: Use one of these import statements:

```python
# Option 1: Direct import (if running from fantasy_football directory)
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))
from data_fetchers.data_collector import FantasyDataCollector

# Option 2: Relative import (if in a package structure)
from src.data_fetchers.data_collector import FantasyDataCollector

# Option 3: Add __init__.py files to make it a proper package
# Create: src/__init__.py and src/data_fetchers/__init__.py (empty files)
# Then: from src.data_fetchers.data_collector import FantasyDataCollector
```

## Baseline Implementation Analysis

Based on your requirements and current data structure, here's the implementation plan:

### Current Data Available

1. **draft_data.csv**: League_ID, Year, Player, Team, Round, Pick, Overall, Is_Autodrafted, Auto_Draft_Type_ID
2. **lineup_data.csv**: League_ID, Week, Team, Player, Slot, Points, Projected_Points, Is_Starter
3. **transaction_data.csv**: League_ID, Week, Team, Player, Action (WAIVER_ADD, DROP, TRADE_JOIN)

### Baseline Requirements

#### 1. Draft Baseline
**Requirement**: Filter for 100% auto drafts, sum each player's total points over season.

**Implementation Notes**:
- Need to identify leagues where ALL picks are autodrafted (Is_Autodrafted = 1 for all picks)
- Sum Points from lineup_data.csv for each player in those leagues
- Group by Player and sum Points across all weeks

**Challenges**:
- Need to ensure we're only counting leagues where 100% of picks are auto
- May need to aggregate by League_ID first to verify 100% auto status

#### 2. Waiver Wire Baseline
**Requirement**: Add total points of player added, subtract total points of player removed, average with respect to position.

**Implementation Notes**:
- Filter transaction_data.csv for WAIVER_ADD and DROP actions (exclude TRADE_JOIN)
- For each WAIVER_ADD: Get player's total Points from lineup_data.csv
- For each DROP: Subtract player's total Points from lineup_data.csv
- Need to identify player positions (requires additional data or API call)
- Average the net points by position

**Challenges**:
- Need player position data (not currently in our CSVs)
- Need to match players across transaction_data and lineup_data
- Position identification may require ESPN API or external source

#### 3. Trade Baseline
**Requirement**: No trade (basically ignore trades)

**Implementation Notes**:
- Simply filter out TRADE_JOIN actions from analysis
- This is the simplest baseline - essentially "do nothing"

#### 4. Start/Sit Baseline
**Requirement**: Add total points of all players that would have started if the highest ESPN projected lineup was used.

**Implementation Notes**:
- Use Projected_Points from lineup_data.csv
- For each week/team, select the optimal lineup based on highest Projected_Points
- Need to respect position limits (QB, RB, WR, TE, FLEX, K, D/ST)
- Sum actual Points for players in the optimal projected lineup

**Challenges**:
- Position limits need to be defined (varies by league settings)
- Need to identify player positions
- FLEX position complicates lineup optimization
- May need to pull ESPN projected lineups via API if not already captured

### Missing Data Requirements

1. **Player Positions**: Not in current CSV files
   - Solution: Add position extraction to data_collector.py or create separate mapping
   - Could use ESPN API player objects which have position info

2. **ESPN Projected Lineups**: Need to verify if Projected_Points in lineup_data.csv is sufficient
   - Current data has Projected_Points per player per week
   - May need to pull optimal projected lineup separately if ESPN provides that endpoint

3. **League Settings**: Position limits, roster sizes
   - May need to extract from league.settings in espn-api
   - Or assume standard settings (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX, 1 K, 1 D/ST)

### Recommended Implementation Steps

1. **Enhance Data Collection**:
   - Add player position to lineup_data.csv
   - Add league settings (position limits) to a new league_settings.csv
   - Verify Projected_Points is sufficient for Start/Sit baseline

2. **Create Baseline Calculator Module**:
   - `src/analysis/baseline_calculator.py`
   - Separate functions for each baseline
   - Helper functions for position identification and lineup optimization

3. **Position Data**:
   - Extract from espn-api player objects during data collection
   - Or create player position mapping from external source

4. **Lineup Optimization**:
   - Create function to select optimal lineup based on projected points
   - Respect position constraints
   - Handle FLEX position intelligently

### Code Structure Recommendation

```
fantasy_football/
├── src/
│   ├── data_fetchers/
│   │   ├── data_collector.py (enhance to capture positions)
│   │   └── ...
│   └── analysis/
│       ├── __init__.py
│       ├── baseline_calculator.py (new)
│       └── position_mapper.py (new if needed)
```

### Next Steps

1. Fix import issue (add __init__.py files or adjust import path)
2. Enhance data_collector.py to capture player positions
3. Create baseline_calculator.py with the four baseline functions
4. Test each baseline calculation
5. Document results and methodology

