"""
Baseline Calculator for Fantasy Football Analysis

Implements four baselines:
1. Draft Baseline: 100% auto-drafted leagues
2. Waiver Wire Baseline: Net points from waiver adds/drops by position
3. Trade Baseline: No trades (baseline = 0 or ignore)
4. Start/Sit Baseline: Optimal projected lineup points
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Tuple


class BaselineCalculator:
    """Calculate fantasy football baselines for comparison."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the baseline calculator.
        
        Args:
            data_dir: Path to directory containing CSV files. If None, uses default.
        """
        if data_dir is None:
            # Default to data/raw/espn relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, '../../data/raw/espn')
            data_dir = os.path.normpath(data_dir)
        
        self.data_dir = data_dir
        self._draft_df = None
        self._lineup_df = None
        self._transaction_df = None
    
    def _load_data(self):
        """Load CSV files into memory."""
        draft_path = os.path.join(self.data_dir, 'draft_data.csv')
        lineup_path = os.path.join(self.data_dir, 'lineup_data.csv')
        transaction_path = os.path.join(self.data_dir, 'transaction_data.csv')
        
        if os.path.exists(draft_path):
            self._draft_df = pd.read_csv(draft_path)
        else:
            raise FileNotFoundError(f"Draft data not found: {draft_path}")
        
        if os.path.exists(lineup_path):
            self._lineup_df = pd.read_csv(lineup_path)
        else:
            raise FileNotFoundError(f"Lineup data not found: {lineup_path}")
        
        if os.path.exists(transaction_path):
            self._transaction_df = pd.read_csv(transaction_path)
        else:
            raise FileNotFoundError(f"Transaction data not found: {transaction_path}")
    
    def calculate_draft_baseline(self) -> pd.DataFrame:
        """
        Calculate Draft Baseline: Sum total points for players in 100% auto-drafted leagues.
        
        Returns:
            DataFrame with Player and Total_Points columns
        """
        if self._draft_df is None or self._lineup_df is None:
            self._load_data()
        
        # Identify leagues where ALL picks are autodrafted
        league_auto_status = self._draft_df.groupby('League_ID')['Is_Autodrafted'].agg(['sum', 'count'])
        league_auto_status['pct_auto'] = league_auto_status['sum'] / league_auto_status['count']
        fully_auto_leagues = league_auto_status[league_auto_status['pct_auto'] == 1.0].index.tolist()
        
        if len(fully_auto_leagues) == 0:
            print("Warning: No 100% auto-drafted leagues found.")
            return pd.DataFrame(columns=['Player', 'Total_Points'])
        
        # Get unique players from fully auto leagues and sum their total points
        auto_draft_players = self._draft_df[
            self._draft_df['League_ID'].isin(fully_auto_leagues)
        ]['Player'].unique()
        player_points = self._lineup_df[
            self._lineup_df['Player'].isin(auto_draft_players)
        ].groupby('Player')['Points'].sum().reset_index()
        player_points.columns = ['Player', 'Total_Points']
        
        return player_points.sort_values('Total_Points', ascending=False)
    
    def calculate_waiver_baseline(self, position_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Calculate Waiver Wire Baseline: Net points from adds/drops, averaged by position.
        
        Args:
            position_map: Optional dict mapping player names to positions.
                        If None, will try to use Position column from lineup_data.csv.
        
        Returns:
            DataFrame with Position and Avg_Net_Points columns
        """
        if self._transaction_df is None or self._lineup_df is None:
            self._load_data()
        
        # Filter for waiver transactions (exclude trades)
        waiver_trans = self._transaction_df[
            self._transaction_df['Action'].isin(['WAIVER_ADD', 'DROP'])
        ].copy()
        
        if len(waiver_trans) == 0:
            print("Warning: No waiver transactions found.")
            return pd.DataFrame(columns=['Position', 'Avg_Net_Points'])
        
        # Get total points and position for each player
        if 'Position' in self._lineup_df.columns:
            player_info = self._lineup_df.groupby('Player').agg({
                'Points': 'sum',
                'Position': 'first'
            }).reset_index()
            player_info.columns = ['Player', 'Total_Points', 'Position']
        else:
            player_info = self._lineup_df.groupby('Player')['Points'].sum().reset_index()
            player_info.columns = ['Player', 'Total_Points']
        
        # Merge with transaction data and calculate net points (positive for adds, negative for drops)
        waiver_trans = waiver_trans.merge(player_info, on='Player', how='left')
        waiver_trans['Net_Points'] = waiver_trans.apply(
            lambda row: row['Total_Points'] if row['Action'] == 'WAIVER_ADD' 
            else -row['Total_Points'], axis=1
        )
        
        # Add positions if not already present
        if 'Position' not in waiver_trans.columns and position_map:
            waiver_trans['Position'] = waiver_trans['Player'].map(position_map)
        elif 'Position' not in waiver_trans.columns:
            print("Warning: No position data available. Returning transaction-level data.")
            return waiver_trans[['Player', 'Action', 'Net_Points']]
        
        # Drop rows without position
        waiver_trans = waiver_trans.dropna(subset=['Position'])
        
        if len(waiver_trans) == 0:
            print("Warning: No transactions with position data found.")
            return pd.DataFrame(columns=['Position', 'Avg_Net_Points'])
        
        # Average net points by position
        position_avg = waiver_trans.groupby('Position')['Net_Points'].mean().reset_index()
        position_avg.columns = ['Position', 'Avg_Net_Points']
        return position_avg.sort_values('Avg_Net_Points', ascending=False)
    
    def calculate_trade_baseline(self) -> Dict:
        """
        Calculate Trade Baseline: No trades (baseline = 0).
        
        Returns:
            Dict with baseline information
        """
        if self._transaction_df is None:
            self._load_data()
        
        trade_count = len(self._transaction_df[
            self._transaction_df['Action'] == 'TRADE_JOIN'
        ])
        
        return {
            'baseline_value': 0,
            'trade_count': trade_count,
            'description': 'No trade baseline - assumes no trades occur'
        }
    
    def calculate_startsit_baseline(
        self, 
        position_map: Optional[Dict[str, str]] = None,
        position_limits: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Calculate Start/Sit Baseline: Points from optimal projected lineup.
        
        Args:
            position_map: Optional dict mapping player names to positions.
                         If None, will try to use Position column from lineup_data.csv.
            position_limits: Dict with position limits (e.g., {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'FLEX': 1})
                           If None, uses standard settings
        
        Returns:
            DataFrame with League_ID, Week, Team, and Optimal_Points
        """
        if self._lineup_df is None:
            self._load_data()
        
        # Check if Position column exists in lineup data
        if 'Position' not in self._lineup_df.columns:
            if position_map is None:
                print("Warning: No Position column in lineup_data.csv and no position_map provided.")
                print("Cannot optimize lineups without position data.")
                return pd.DataFrame()
            else:
                # Use provided position_map
                lineup_with_pos = self._lineup_df.copy()
                lineup_with_pos['Position'] = lineup_with_pos['Player'].map(position_map)
        else:
            # Use Position from CSV
            lineup_with_pos = self._lineup_df.copy()
        
        # Drop rows without position
        lineup_with_pos = lineup_with_pos.dropna(subset=['Position'])
        
        if len(lineup_with_pos) == 0:
            print("Warning: No lineup data with position information found.")
            return pd.DataFrame()
        
        if position_limits is None:
            position_limits = {
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 
                'FLEX': 1, 'K': 1, 'D/ST': 1
            }
        
        results = []
        
        # Process each league/week/team combination
        for (league_id, week, team), group in lineup_with_pos.groupby(['League_ID', 'Week', 'Team']):
            optimal_points = self._optimize_lineup(
                group, position_limits, use_projected=True
            )
            results.append({
                'League_ID': league_id,
                'Week': week,
                'Team': team,
                'Optimal_Points': optimal_points
            })
        
        return pd.DataFrame(results)
    
    def _optimize_lineup(
        self, 
        player_data: pd.DataFrame, 
        position_limits: Dict[str, int],
        use_projected: bool = True
    ) -> float:
        """
        Select optimal lineup based on projected points.
        
        Args:
            player_data: DataFrame with players for one team/week
            position_limits: Dict with position limits
            use_projected: If True, use Projected_Points; else use Points
        
        Returns:
            Total points from optimal lineup
        """
        # Choose which column to use for optimization (projected vs actual)
        score_col = 'Projected_Points' if use_projected else 'Points'
        
        # Sort players by score (descending) to prioritize best players
        players_sorted = player_data.sort_values(score_col, ascending=False)
        
        selected = []
        
        # Fill position-specific slots (QB, RB, WR, TE, K, D/ST)
        for pos, limit in position_limits.items():
            if pos == 'FLEX':
                continue  # Fill FLEX after position-specific slots
            
            # Select top N players at this position (sorted by projected points)
            pos_players = players_sorted[players_sorted['Position'] == pos].head(limit)
            
            for _, player in pos_players.iterrows():
                selected.append(player['Player'])
        
        # Fill FLEX position with best remaining RB/WR/TE
        if 'FLEX' in position_limits:
            flex_limit = position_limits['FLEX']
            remaining = players_sorted[
                (players_sorted['Position'].isin(['RB', 'WR', 'TE'])) &
                (~players_sorted['Player'].isin(selected))
            ].head(flex_limit)
            
            for _, player in remaining.iterrows():
                selected.append(player['Player'])
        
        # Sum actual points for selected players
        selected_players = player_data[player_data['Player'].isin(selected)]
        total_points = selected_players['Points'].sum()
        
        return total_points

