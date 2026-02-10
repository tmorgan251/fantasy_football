"""
Injury Impact Analysis for Fantasy Football

This module provides comprehensive analysis of how injuries affect player performance,
including point differentials, recovery timelines, and position-specific impacts.
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import normalize_player_name from draft_value_analyzer
from src.analysis.draft_value_analyzer import DraftValueAnalyzer


class InjuryAnalyzer:
    """
    Analyzes the impact of injuries on fantasy football player performance.
    
    Features:
    - Matches injuries to lineup data by player name and week
    - Calculates point differentials (actual - projected) for injured vs healthy players
    - Tracks consecutive injury weeks with same type and status
    - Generates visualizations (heatmaps, recovery timelines, distributions)
    """
    
    def __init__(
        self,
        injury_dir: str = "data/raw/injuries",
        lineup_data_path: Optional[str] = None,
        years: Optional[List[int]] = None,
        verbose: bool = False
    ):
        """
        Initialize the InjuryAnalyzer.
        
        Args:
            injury_dir: Directory containing injury CSV files (nfl_injuries_YYYY.csv)
            lineup_data_path: Path to lineup_data.csv (if None, searches in data/raw/espn)
            years: List of years to analyze (if None, auto-detects from injury files)
            verbose: If True, print detailed progress messages
        """
        self.injury_dir = Path(injury_dir)
        self.verbose = verbose
        self.years = years
        
        # Set up lineup data path
        if lineup_data_path is None:
            # Try to find lineup_data.csv in data/raw/espn
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            # Check if there's a single file or year-specific files
            potential_paths = [
                project_root / "data" / "raw" / "espn" / "lineup_data.csv",
                project_root / "data" / "raw" / "espn" / "2024" / "lineup_data.csv"
            ]
            for path in potential_paths:
                if path.exists():
                    lineup_data_path = str(path)
                    break
            
            if lineup_data_path is None:
                raise FileNotFoundError(
                    "Could not find lineup_data.csv. Please specify lineup_data_path."
                )
        
        self.lineup_data_path = lineup_data_path
        
        # Load and prepare data
        self.injury_df = None
        self.lineup_df = None
        self.merged_df = None
        self.player_baselines = None
        
        if self.verbose:
            print("Initializing InjuryAnalyzer...")
    
    def load_injury_data(self) -> pd.DataFrame:
        """
        Load injury data from CSV files for all specified years.
        
        Returns:
            DataFrame with injury data
        """
        if self.verbose:
            print("\nLoading injury data...")
        
        all_injuries = []
        
        # Determine years to load
        if self.years is None:
            # Auto-detect years from available files
            injury_files = list(self.injury_dir.glob("nfl_injuries_*.csv"))
            years = sorted([int(f.stem.split("_")[-1]) for f in injury_files])
            if self.verbose:
                print(f"  Auto-detected years: {years}")
        else:
            years = self.years
        
        for year in years:
            injury_file = self.injury_dir / f"nfl_injuries_{year}.csv"
            if not injury_file.exists():
                if self.verbose:
                    print(f"  ⚠ Injury file not found for {year}: {injury_file}")
                continue
            
            df = pd.read_csv(injury_file)
            if self.verbose:
                print(f"  ✓ Loaded {len(df):,} records from {year}")
            all_injuries.append(df)
        
        if not all_injuries:
            raise FileNotFoundError(f"No injury data files found in {self.injury_dir}")
        
        injury_df = pd.concat(all_injuries, ignore_index=True)
        
        # Clean and standardize injury data
        # Filter to regular season only
        injury_df = injury_df[injury_df['game_type'] == 'REG'].copy()
        
        # Handle empty and "Not injury related" as "no injury"
        injury_df['has_injury'] = (
            injury_df['report_primary_injury'].notna() & 
            ~injury_df['report_primary_injury'].str.contains('Not injury related', case=False, na=False) &
            (injury_df['report_primary_injury'].str.strip() != '')
        )
        
        # Standardize injury status
        injury_df['injury_status'] = injury_df['report_status'].copy()
        injury_df.loc[~injury_df['has_injury'], 'injury_status'] = 'No Injury'
        injury_df.loc[injury_df['injury_status'].isna(), 'injury_status'] = 'No Injury'
        
        # Standardize injury type
        injury_df['injury_type'] = injury_df['report_primary_injury'].copy()
        injury_df.loc[~injury_df['has_injury'], 'injury_type'] = 'None'
        injury_df.loc[injury_df['injury_type'].isna(), 'injury_type'] = 'None'
        
        # Normalize player names for matching
        injury_df['player_norm'] = injury_df['full_name'].apply(
            lambda x: DraftValueAnalyzer.normalize_player_name(x) if pd.notna(x) else x
        )
        
        # Standardize position
        injury_df['position'] = injury_df['position'].str.upper()
        
        if self.verbose:
            print(f"\n  Total injury records: {len(injury_df):,}")
            print(f"  Years: {injury_df['season'].min()} - {injury_df['season'].max()}")
            print(f"  Unique players: {injury_df['player_norm'].nunique():,}")
            print(f"  Injury statuses: {injury_df['injury_status'].value_counts().to_dict()}")
        
        self.injury_df = injury_df
        return injury_df
    
    def load_lineup_data(self) -> pd.DataFrame:
        """
        Load lineup data from CSV file.
        
        Returns:
            DataFrame with lineup data
        """
        if self.verbose:
            print("\nLoading lineup data...")
        
        # Always use year-specific directories (more reliable)
        # Check if single file path was provided but doesn't have year info
        use_year_dirs = True
        if os.path.isfile(self.lineup_data_path):
            # Check if path contains a year
            path_parts = Path(self.lineup_data_path).parts
            has_year_in_path = any(
                part.isdigit() and len(part) == 4 and 2000 <= int(part) <= 2100
                for part in path_parts
            )
            if has_year_in_path:
                # Single file with year in path - use it
                use_year_dirs = False
        
        if not use_year_dirs and os.path.isfile(self.lineup_data_path):
            # Single file with year in path
            lineup_df = pd.read_csv(
                self.lineup_data_path,
                on_bad_lines='skip',
                engine='python'
            )
            # Extract year from path
            path_parts = Path(self.lineup_data_path).parts
            for part in path_parts:
                if part.isdigit() and len(part) == 4 and 2000 <= int(part) <= 2100:
                    lineup_df['Year'] = int(part)
                    break
        else:
            # Search in year subdirectories
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            base_dir = project_root / "data" / "raw" / "espn"
            
            all_lineups = []
            for year_dir in sorted(base_dir.glob("[0-9][0-9][0-9][0-9]")):
                lineup_file = year_dir / "lineup_data.csv"
                if lineup_file.exists():
                    try:
                        df = pd.read_csv(
                            lineup_file,
                            on_bad_lines='skip',  # Skip malformed rows
                            engine='python'  # More forgiving parser
                        )
                        # Add Year column from directory name
                        year = int(year_dir.name)
                        df['Year'] = year
                        all_lineups.append(df)
                        if self.verbose:
                            print(f"  ✓ Loaded {len(df):,} records from {year_dir.name}")
                    except Exception as e:
                        if self.verbose:
                            print(f"  ⚠️  Error loading {year_dir.name}: {e}")
                        continue
            
            if not all_lineups:
                raise FileNotFoundError(f"No lineup data found in {base_dir}")
            
            lineup_df = pd.concat(all_lineups, ignore_index=True)
        
        # Ensure Year column exists
        if 'Year' not in lineup_df.columns:
            raise ValueError("Year column not found in lineup data and could not be inferred")
        
        # Normalize player names
        lineup_df['player_norm'] = lineup_df['Player'].apply(
            lambda x: DraftValueAnalyzer.normalize_player_name(x) if pd.notna(x) else x
        )
        
        # Calculate point differential (actual - projected)
        lineup_df['point_differential'] = (
            lineup_df['Points'].fillna(0) - lineup_df['Projected_Points'].fillna(0)
        )
        
        if self.verbose:
            print(f"\n  Total lineup records: {len(lineup_df):,}")
            print(f"  Years: {lineup_df['Year'].min()} - {lineup_df['Year'].max()}")
            print(f"  Unique players: {lineup_df['player_norm'].nunique():,}")
        
        self.lineup_df = lineup_df
        return lineup_df
    
    def calculate_player_baselines(self) -> pd.DataFrame:
        """
        Calculate baseline point differentials for each player.
        
        Baseline = average (actual - projected) points in weeks with no injuries.
        Includes previous year data if available (except for rookies).
        
        OPTIMIZED: Uses vectorized operations instead of per-player loops.
        
        Returns:
            DataFrame with columns: player_norm, position, baseline_avg, baseline_std, n_weeks
        """
        if self.verbose:
            print("\nCalculating player baselines (healthy week performance)...")
            import time
            start_time = time.time()
        
        if self.injury_df is None:
            self.load_injury_data()
        if self.lineup_df is None:
            self.load_lineup_data()
        
        # VECTORIZED APPROACH: Merge all at once instead of per-player
        # Merge lineup with injury data to identify injury weeks
        merged = self.lineup_df.merge(
            self.injury_df[['season', 'week', 'player_norm', 'has_injury']],
            left_on=['Year', 'Week', 'player_norm'],
            right_on=['season', 'week', 'player_norm'],
            how='left',
            suffixes=('', '_injury')
        )
        
        # Mark weeks with no injury data as "no injury"
        merged['has_injury'] = merged['has_injury'].fillna(False)
        
        # Filter to healthy weeks only
        healthy_weeks = merged[~merged['has_injury']].copy()
        
        # Group by player and calculate baselines (vectorized - much faster!)
        baselines = healthy_weeks.groupby('player_norm').agg({
            'point_differential': ['mean', 'std', 'count'],
            'Position': 'first'  # Get position from lineup data
        }).reset_index()
        
        # Flatten column names
        baselines.columns = ['player_norm', 'baseline_avg', 'baseline_std', 'n_weeks', 'position']
        
        # Fill NaN std with 0 (only one data point)
        baselines['baseline_std'] = baselines['baseline_std'].fillna(0.0)
        
        if self.verbose:
            elapsed = time.time() - start_time
            print(f"  ✓ Calculated baselines for {len(baselines):,} players in {elapsed:.1f} seconds")
            print(f"  Average baseline: {baselines['baseline_avg'].mean():.2f} points")
            print(f"  Players with 1+ healthy weeks: {(baselines['n_weeks'] >= 1).sum():,}")
            print(f"  Players with 5+ healthy weeks: {(baselines['n_weeks'] >= 5).sum():,}")
        
        self.player_baselines = baselines
        return baselines
    
    def merge_injury_lineup_data(self) -> pd.DataFrame:
        """
        Merge injury data with lineup data to create analysis dataset.
        
        Returns:
            Merged DataFrame with injury and performance data
        """
        if self.verbose:
            print("\nMerging injury and lineup data...")
        
        if self.injury_df is None:
            self.load_injury_data()
        if self.lineup_df is None:
            self.load_lineup_data()
        
        # Merge on year, week, and normalized player name
        merged = self.lineup_df.merge(
            self.injury_df[[
                'season', 'week', 'player_norm', 'position', 'has_injury',
                'injury_status', 'injury_type'
            ]],
            left_on=['Year', 'Week', 'player_norm'],
            right_on=['season', 'week', 'player_norm'],
            how='left',
            suffixes=('_lineup', '_injury')
        )
        
        # Handle weeks with no injury data
        merged['has_injury'] = merged['has_injury'].fillna(False)
        merged['injury_status'] = merged['injury_status'].fillna('No Injury')
        merged['injury_type'] = merged['injury_type'].fillna('None')
        
        # Use position from lineup data (more reliable), fallback to injury data position
        # Note: 'Position' (capital P) is from lineup_df, 'position' (lowercase) is from injury_df
        # Since they have different capitalization, both columns exist after merge
        if 'Position' in merged.columns and 'position' in merged.columns:
            merged['position'] = merged['Position'].fillna(merged['position'])
        elif 'Position' in merged.columns:
            merged['position'] = merged['Position']
        elif 'position' in merged.columns:
            merged['position'] = merged['position']
        else:
            # If neither exists, create empty column
            merged['position'] = None
        
        # Add baseline if available
        if self.player_baselines is None:
            self.calculate_player_baselines()
        
        merged = merged.merge(
            self.player_baselines[['player_norm', 'baseline_avg', 'baseline_std', 'n_weeks']],
            on='player_norm',
            how='left'
        )
        
        # Calculate point differential
        merged['point_differential'] = (
            merged['Points'].fillna(0) - merged['Projected_Points'].fillna(0)
        )
        
        if self.verbose:
            print(f"  Merged {len(merged):,} records")
            print(f"  Injury weeks: {merged['has_injury'].sum():,}")
            print(f"  Healthy weeks: {(~merged['has_injury']).sum():,}")
        
        self.merged_df = merged
        return merged
    
    def track_consecutive_injuries(self) -> pd.DataFrame:
        """
        Track consecutive weeks with the same injury type and status.
        
        OPTIMIZED: Uses vectorized operations with groupby instead of per-player loops.
        Also resets between years to prevent cross-season counting.
        
        Returns:
            DataFrame with consecutive_injury_weeks column added
        """
        if self.merged_df is None:
            self.merge_injury_lineup_data()
        
        if self.verbose:
            print("\nTracking consecutive injury weeks...")
            import time
            start_time = time.time()
        
        df = self.merged_df.copy()
        df = df.sort_values(['player_norm', 'Year', 'Week']).reset_index(drop=True)
        
        # Initialize consecutive week counter
        df['consecutive_injury_weeks'] = 0
        
        # OPTIMIZED: Use groupby with apply for vectorized processing
        # Group by player, league, AND year to track per-league injuries
        def track_consecutive_group(group):
            """Track consecutive injuries for a single player-league-year group."""
            group = group.copy()
            group = group.sort_values('Week')
            group['consecutive_injury_weeks'] = 0
            
            consecutive_count = 0
            prev_injury_type = None
            prev_injury_status = None
            prev_has_injury = False
            prev_week = None
            
            for idx in group.index:
                current_week = group.loc[idx, 'Week']
                has_injury = group.loc[idx, 'has_injury']
                
                # Reset if there's a gap in weeks (e.g., week 1 then week 3)
                if prev_week is not None and current_week != prev_week + 1:
                    consecutive_count = 0
                    prev_injury_type = None
                    prev_injury_status = None
                    prev_has_injury = False
                
                if has_injury:
                    current_type = group.loc[idx, 'injury_type']
                    current_status = group.loc[idx, 'injury_status']
                    
                    # Check if same injury type AND status as previous week
                    if (prev_has_injury and 
                        current_type == prev_injury_type and 
                        current_status == prev_injury_status and
                        current_type != 'None' and
                        prev_week is not None and
                        current_week == prev_week + 1):  # Must be consecutive weeks
                        consecutive_count += 1
                    else:
                        # Reset count (new injury or different type/status)
                        consecutive_count = 1
                    
                    group.loc[idx, 'consecutive_injury_weeks'] = consecutive_count
                    prev_injury_type = current_type
                    prev_injury_status = current_status
                    prev_has_injury = True
                else:
                    # Reset on healthy week
                    consecutive_count = 0
                    prev_injury_type = None
                    prev_injury_status = None
                    prev_has_injury = False
                
                prev_week = current_week
            
            return group
        
        # Group by player, league, AND year (resets between seasons and leagues)
        # This prevents counting the same injury across multiple leagues
        if 'League_ID' in df.columns:
            df = df.groupby(['player_norm', 'League_ID', 'Year'], group_keys=False).apply(track_consecutive_group).reset_index(drop=True)
        else:
            # Fallback if League_ID not available
            df = df.groupby(['player_norm', 'Year'], group_keys=False).apply(track_consecutive_group).reset_index(drop=True)
        
        if self.verbose:
            elapsed = time.time() - start_time
            injured_players = df[df['has_injury']]
            if len(injured_players) > 0:
                max_consecutive = injured_players['consecutive_injury_weeks'].max()
                print(f"  ✓ Tracked consecutive injuries in {elapsed:.1f} seconds")
                print(f"  Max consecutive injury weeks: {max_consecutive}")
                print(f"  Players with 2+ consecutive weeks: {(injured_players['consecutive_injury_weeks'] >= 2).sum():,}")
        
        self.merged_df = df
        return df
    
    def analyze_injury_impact(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform comprehensive injury impact analysis.
        
        Returns:
            Tuple of (aggregated_stats_df, individual_records_df)
        """
        if self.verbose:
            print("\nAnalyzing injury impact...")
        
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df.copy()
        
        # Filter to relevant columns for analysis
        analysis_cols = [
            'player_norm', 'Year', 'Week', 'position', 'has_injury',
            'injury_status', 'injury_type', 'consecutive_injury_weeks',
            'Points', 'Projected_Points', 'point_differential',
            'baseline_avg', 'baseline_std', 'n_weeks'
        ]
        analysis_df = df[analysis_cols].copy()
        
        # Calculate impact metrics
        analysis_df['vs_baseline'] = analysis_df['point_differential'] - analysis_df['baseline_avg']
        analysis_df['vs_projected'] = analysis_df['point_differential']
        
        # Create aggregated statistics
        agg_stats = []
        
        # Group by injury type, position, and status
        for injury_type in analysis_df['injury_type'].unique():
            if injury_type == 'None':
                continue
            
            for position in analysis_df['position'].dropna().unique():
                for status in analysis_df['injury_status'].unique():
                    if status == 'No Injury':
                        continue
                    
                    mask = (
                        (analysis_df['injury_type'] == injury_type) &
                        (analysis_df['position'] == position) &
                        (analysis_df['injury_status'] == status)
                    )
                    
                    if mask.sum() == 0:
                        continue
                    
                    subset = analysis_df[mask]
                    
                    agg_stats.append({
                        'injury_type': injury_type,
                        'position': position,
                        'injury_status': status,
                        'n_weeks': len(subset),
                        'n_players': subset['player_norm'].nunique(),
                        'avg_point_differential': subset['point_differential'].mean(),
                        'median_point_differential': subset['point_differential'].median(),
                        'std_point_differential': subset['point_differential'].std(),
                        'avg_vs_baseline': subset['vs_baseline'].mean(),
                        'median_vs_baseline': subset['vs_baseline'].median(),
                    })
        
        agg_stats_df = pd.DataFrame(agg_stats)
        
        if self.verbose:
            print(f"  Generated {len(agg_stats_df):,} aggregated statistics")
            print(f"  Individual records: {len(analysis_df):,}")
        
        return agg_stats_df, analysis_df
    
    def plot_heatmap_type_position(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap: Injury Type (rows) × Position (columns), color = avg point differential.
        
        Args:
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df[self.merged_df['has_injury']].copy()
        
        # Aggregate by injury type and position
        heatmap_data = df.groupby(['injury_type', 'position'])['point_differential'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='injury_type', columns='position', values='point_differential')
        
        # Filter out 'None' injury type
        if 'None' in heatmap_pivot.index:
            heatmap_pivot = heatmap_pivot.drop('None')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(8, len(heatmap_pivot) * 0.5)))
        sns.heatmap(
            heatmap_pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            center=0,
            vmin=-10,
            vmax=10,
            cbar_kws={'label': 'Avg Point Differential (Actual - Projected)'},
            ax=ax
        )
        ax.set_title('Injury Impact: Type × Position\n(Average Point Differential)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Injury Type', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap_type_status(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap: Injury Type (rows) × Status (columns), color = avg point differential.
        
        Args:
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df[self.merged_df['has_injury']].copy()
        
        # Aggregate by injury type and status
        heatmap_data = df.groupby(['injury_type', 'injury_status'])['point_differential'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='injury_type', columns='injury_status', values='point_differential')
        
        # Filter out 'None' injury type
        if 'None' in heatmap_pivot.index:
            heatmap_pivot = heatmap_pivot.drop('None')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, max(8, len(heatmap_pivot) * 0.5)))
        sns.heatmap(
            heatmap_pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            center=0,
            vmin=-10,
            vmax=10,
            cbar_kws={'label': 'Avg Point Differential (Actual - Projected)'},
            ax=ax
        )
        ax.set_title('Injury Impact: Type × Status\n(Average Point Differential)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Injury Status', fontsize=12)
        ax.set_ylabel('Injury Type', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_heatmap_position_status(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap: Position (rows) × Status (columns), color = avg point differential.
        
        Args:
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df[self.merged_df['has_injury']].copy()
        
        # Aggregate by position and status
        heatmap_data = df.groupby(['position', 'injury_status'])['point_differential'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='position', columns='injury_status', values='point_differential')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            center=0,
            vmin=-10,
            vmax=10,
            cbar_kws={'label': 'Avg Point Differential (Actual - Projected)'},
            ax=ax
        )
        ax.set_title('Injury Impact: Position × Status\n(Average Point Differential)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Injury Status', fontsize=12)
        ax.set_ylabel('Position', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_recovery_timeline_by_type(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot recovery timeline showing performance over consecutive weeks for same injury type.
        Shows how expected points fall/climb/stay unchanged.
        
        Args:
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df[self.merged_df['has_injury']].copy()
        
        # Filter to players with 2+ consecutive weeks of same injury
        df = df[df['consecutive_injury_weeks'] >= 1].copy()
        
        # Group by injury type and consecutive week number
        timeline_data = df.groupby(['injury_type', 'consecutive_injury_weeks']).agg({
            'point_differential': ['mean', 'std', 'count'],
            'Projected_Points': 'mean',
            'Points': 'mean'
        }).reset_index()
        
        timeline_data.columns = [
            'injury_type', 'consecutive_week', 'avg_diff', 'std_diff', 'n_weeks',
            'avg_projected', 'avg_actual'
        ]
        
        # Filter to injury types with sufficient data
        injury_counts = df.groupby('injury_type').size()
        common_injuries = injury_counts[injury_counts >= 20].index.tolist()
        timeline_data = timeline_data[timeline_data['injury_type'].isin(common_injuries)]
        
        # Create figure
        n_types = len(common_injuries)
        fig, axes = plt.subplots(1, min(3, n_types), figsize=(15, 5))
        if n_types == 1:
            axes = [axes]
        elif n_types == 2:
            axes = list(axes)
        
        for idx, injury_type in enumerate(common_injuries[:3]):  # Top 3 most common
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            injury_timeline = timeline_data[timeline_data['injury_type'] == injury_type]
            
            # Plot average point differential over consecutive weeks
            ax.plot(
                injury_timeline['consecutive_week'],
                injury_timeline['avg_diff'],
                marker='o',
                linewidth=2,
                markersize=8,
                label='Avg Point Diff'
            )
            
            # Add error bars (std)
            ax.errorbar(
                injury_timeline['consecutive_week'],
                injury_timeline['avg_diff'],
                yerr=timeline_data['std_diff'],
                alpha=0.3,
                capsize=3
            )
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Consecutive Week with Injury', fontsize=10)
            ax.set_ylabel('Avg Point Differential', fontsize=10)
            ax.set_title(f'{injury_type}\n(n={injury_timeline["n_weeks"].sum()} weeks)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        fig.suptitle('Recovery Timeline by Injury Type\n(Performance Over Consecutive Weeks)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_recovery_timeline_by_status(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot recovery timeline showing performance over consecutive weeks by injury status.
        
        Args:
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df[self.merged_df['has_injury']].copy()
        
        # Group by status and consecutive week number
        timeline_data = df.groupby(['injury_status', 'consecutive_injury_weeks']).agg({
            'point_differential': ['mean', 'std', 'count'],
            'Projected_Points': 'mean',
            'Points': 'mean'
        }).reset_index()
        
        timeline_data.columns = [
            'injury_status', 'consecutive_week', 'avg_diff', 'std_diff', 'n_weeks',
            'avg_projected', 'avg_actual'
        ]
        
        # Create figure
        statuses = ['Questionable', 'Doubtful', 'Out']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for status in statuses:
            status_timeline = timeline_data[timeline_data['injury_status'] == status]
            if len(status_timeline) == 0:
                continue
            
            ax.plot(
                status_timeline['consecutive_week'],
                status_timeline['avg_diff'],
                marker='o',
                linewidth=2,
                markersize=8,
                label=status
            )
            
            # Add error bars
            ax.errorbar(
                status_timeline['consecutive_week'],
                status_timeline['avg_diff'],
                yerr=status_timeline['std_diff'],
                alpha=0.3,
                capsize=3
            )
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Consecutive Week with Injury', fontsize=12)
        ax.set_ylabel('Avg Point Differential (Actual - Projected)', fontsize=12)
        ax.set_title('Recovery Timeline by Injury Status\n(Performance Over Consecutive Weeks)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_point_differential_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of point differentials for injured vs healthy players.
        
        Args:
            save_path: Optional path to save figure
        
        Returns:
            matplotlib Figure
        """
        if self.merged_df is None:
            self.track_consecutive_injuries()
        
        df = self.merged_df.copy()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall distribution: Injured vs Healthy
        ax1 = axes[0, 0]
        injured = df[df['has_injury']]['point_differential'].dropna()
        healthy = df[~df['has_injury']]['point_differential'].dropna()
        
        ax1.hist(healthy, bins=50, alpha=0.6, label='Healthy', color='green', density=True)
        ax1.hist(injured, bins=50, alpha=0.6, label='Injured', color='red', density=True)
        ax1.axvline(x=0, color='black', linestyle=' --', alpha=0.5)
        ax1.set_xlabel('Point Differential (Actual - Projected)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Overall: Injured vs Healthy', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. By injury status
        ax2 = axes[0, 1]
        statuses = ['Questionable', 'Doubtful', 'Out']
        colors = ['orange', 'red', 'darkred']
        for status, color in zip(statuses, colors):
            status_data = df[df['injury_status'] == status]['point_differential'].dropna()
            if len(status_data) > 0:
                ax2.hist(status_data, bins=30, alpha=0.5, label=status, color=color, density=True)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Point Differential (Actual - Projected)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('By Injury Status', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. By position (top 4 positions)
        ax3 = axes[1, 0]
        top_positions = df[df['has_injury']]['position'].value_counts().head(4).index
        for pos in top_positions:
            pos_data = df[(df['has_injury']) & (df['position'] == pos)]['point_differential'].dropna()
            if len(pos_data) > 0:
                ax3.hist(pos_data, bins=30, alpha=0.5, label=pos, density=True)
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Point Differential (Actual - Projected)', fontsize=11)
        ax3.set_ylabel('Density', fontsize=11)
        ax3.set_title('By Position (Injured Only)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        ax4 = axes[1, 1]
        box_data = []
        box_labels = []
        for status in ['No Injury', 'Questionable', 'Doubtful', 'Out']:
            status_data = df[df['injury_status'] == status]['point_differential'].dropna()
            if len(status_data) > 10:  # Only include if enough data
                box_data.append(status_data)
                box_labels.append(status)
        
        if box_data:
            ax4.boxplot(box_data, labels=box_labels)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_ylabel('Point Differential (Actual - Projected)', fontsize=11)
            ax4.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle('Point Differential Distributions: Injured vs Healthy Players', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def run_full_analysis(
        self,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete injury analysis pipeline.
        
        For faster execution, you can call individual steps separately:
        1. analyzer.load_injury_data()
        2. analyzer.load_lineup_data()
        3. analyzer.calculate_player_baselines()
        4. analyzer.merge_injury_lineup_data()
        5. analyzer.track_consecutive_injuries()
        6. agg_stats, individual = analyzer.analyze_injury_impact()
        
        Args:
            save_results: If True, save results to CSV files
            output_dir: Directory to save results (default: data/preprocessed)
        
        Returns:
            Tuple of (aggregated_stats_df, individual_records_df)
        """
        if self.verbose:
            print("\n" + "="*70)
            print("RUNNING FULL INJURY ANALYSIS PIPELINE")
            print("="*70)
            import time
            pipeline_start = time.time()
        
        # Step 1: Load data
        if self.verbose:
            print("\n[Step 1/5] Loading data...")
        self.load_injury_data()
        self.load_lineup_data()
        
        # Step 2: Calculate baselines (now optimized - much faster!)
        if self.verbose:
            print("\n[Step 2/5] Calculating player baselines...")
        self.calculate_player_baselines()
        
        # Step 3: Merge data
        if self.verbose:
            print("\n[Step 3/5] Merging injury and lineup data...")
        self.merge_injury_lineup_data()
        
        # Step 4: Track consecutive injuries (now optimized)
        if self.verbose:
            print("\n[Step 4/5] Tracking consecutive injury weeks...")
        self.track_consecutive_injuries()
        
        # Step 5: Analyze impact
        if self.verbose:
            print("\n[Step 5/5] Analyzing injury impact...")
        agg_stats, individual_records = self.analyze_injury_impact()
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                script_dir = Path(__file__).parent
                project_root = script_dir.parent.parent
                output_dir = project_root / "data" / "preprocessed"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            agg_path = output_dir / "injury_impact_aggregated.csv"
            individual_path = output_dir / "injury_impact_individual.csv"
            
            agg_stats.to_csv(agg_path, index=False)
            individual_records.to_csv(individual_path, index=False)
            
            if self.verbose:
                print(f"\n✓ Saved aggregated stats to: {agg_path}")
                print(f"✓ Saved individual records to: {individual_path}")
        
        if self.verbose:
            pipeline_elapsed = time.time() - pipeline_start
            print("\n" + "="*70)
            print(f"ANALYSIS COMPLETE! (Total time: {pipeline_elapsed/60:.1f} minutes)")
            print("="*70)
        
        return agg_stats, individual_records

