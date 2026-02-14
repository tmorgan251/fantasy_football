"""
Draft Value Analysis Module

This module provides a comprehensive pipeline for analyzing fantasy football draft value
using autodrafted teams as a baseline. It includes data cleaning, filtering, enrichment,
optimal lineup computation, and statistical analysis.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Optional, Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from scipy.stats.mstats import winsorize


class DraftValueAnalyzer:
    """
    Main class for analyzing fantasy football draft value.
    
    This class orchestrates the entire pipeline from raw data to final analysis,
    including data cleaning, filtering, enrichment, and statistical analysis.
    """
    
    def __init__(
        self,
        raw_base: Path | str = "data/raw/espn",
        out_dir: Path | str = "data/preprocessed",
        years: range | List[int] = None,
        expected_starters: Dict[str, int] = None,
        expected_draft_length: int = 160,
        anchors_by_pos: Dict[str, List[str]] = None,
        anchor_synonyms: Dict[str, str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Draft Value Analyzer.
        
        Args:
            raw_base: Base directory containing year subdirectories with CSV files
            out_dir: Output directory for processed data
            years: Years to process (default: range(2021, 2025))
            expected_starters: Expected starter configuration (default: standard 1QB/2RB/2WR/1TE/1FLEX/1K/1DST)
            expected_draft_length: Expected draft length (default: 160)
            anchors_by_pos: Anchor players for scoring rule detection
            anchor_synonyms: Player name synonyms for anchor matching
            verbose: If True, print progress messages
        """
        self.raw_base = Path(raw_base)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.years = years if years is not None else range(2021, 2025)
        self.raw_files = ["lineup_data.csv", "draft_data.csv", "transaction_data.csv"]
        
        self.expected_starters = expected_starters or {
            "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, 
            "K": 1, "D/ST": 1, "OP": 0, "STARTERS": 9
        }
        self.expected_draft_length = expected_draft_length
        
        self.anchors_by_pos = anchors_by_pos or {
            "QB": ["Josh Allen", "Patrick Mahomes"],
            "RB": ["Christian McCaffrey", "Derrick Henry"],
            "WR": ["Justin Jefferson", "Tyreek Hill"],
            "TE": ["Travis Kelce", "Mark Andrews"],
            "K": ["Justin Tucker"],
            "D/ST": ["Patriots D/ST", "49ers D/ST", "Cowboys D/ST"],
        }
        self.anchor_synonyms = anchor_synonyms or {}
        
        self.verbose = verbose
        
        # Core position set
        self.core_pos = {"QB", "RB", "WR", "TE", "K", "D/ST", "DST", "DEF"}
        
        # Data storage
        self.draft_raw = None
        self.lineups_raw = None
        self.draft_enriched = None
        self.draft_with_valid = None
        self.draft_scored = None
        self.lineups_filt = None  # Filtered lineups (for waiver wire, start/sit analysis)
        self.optimal_selected = None  # Optimal lineup selections (for waiver wire, start/sit analysis)
        
    # ==================== UTILITY METHODS ====================
    
    # @staticmethod decorator: This method doesn't need access to 'self' (the instance).
    # It can be called directly on the class: DraftValueAnalyzer.normalize_player_name(...)
    # without creating an instance. This is useful for utility functions that logically
    # belong to the class but don't use instance data.
    @staticmethod
    def normalize_player_name(name: str, synonyms: Dict[str, str] = None) -> str:
        """
        Normalize player name for consistent matching.
        
        Return type annotation (-> str): Tells Python and IDEs what type this function
        returns. This helps with type checking, autocomplete, and makes the code
        self-documenting. Without it, you'd have to read the code to know it returns a string.
        """
        if pd.isna(name):
            return name
        s = str(name).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" Jr.", "").replace(" Sr.", "")
        if synonyms:
            s = synonyms.get(s, s)
        return s
    
    # @staticmethod decorator: Same reasoning as above - utility function that doesn't
    # need instance data, so it can be called without creating an object.
    @staticmethod
    def normalize_slot(slot: str) -> str:
        """
        Normalize lineup slot designation.
        
        Return type annotation (-> str): Documents that this returns a string, enabling
        better IDE support and type checking tools.
        """
        s = str(slot).strip().upper()
        if s in {"DST", "DEF", "D/ST"}:
            return "D/ST"
        if s in {"BE", "BENCH"}:
            return "BE"
        if s in {"IR"}:
            return "IR"
        if s in {"RB/WR/TE", "FLEX"}:
            return "FLEX"
        if s in {"OP", "SUPERFLEX", "QB/RB/WR/TE"}:
            return "OP"
        return s
    
    # ==================== DATA LOADING ====================
    
    # Return type annotation (-> None): This method modifies files but doesn't return
    # a value. The annotation makes it clear there's no return value to use.
    def clean_raw_data(self) -> None:
        """Drop duplicate rows in raw CSV files."""
        if self.verbose:
            print("\n=== Cleaning Raw Data (Dropping Duplicates) ===")
        
        for year in self.years:
            if self.verbose:
                print(f"\n=== YEAR {year} ===")
            year_dir = self.raw_base / str(year)
            
            for fname in self.raw_files:
                path = year_dir / fname
                if not path.exists():
                    if self.verbose:
                        print(f"  {fname}: missing")
                    continue
                
                df = pd.read_csv(path)
                before = len(df)
                df2 = df.drop_duplicates()
                after = len(df2)
                dropped = before - after
                
                if dropped == 0:
                    if self.verbose:
                        print(f"  {fname}: no duplicates ({before:,} rows)")
                else:
                    df2.to_csv(path, index=False)
                    if self.verbose:
                        print(f"  {fname}: dropped {dropped:,} duplicate rows ({before:,} → {after:,})")
    
    # Return type annotation (-> Tuple[pd.DataFrame, pd.DataFrame]): Documents that this
    # returns a tuple of two DataFrames. Without this, you'd have to read the code to know
    # you need to unpack: draft, lineups = analyzer.load_multi_season_data()
    def load_multi_season_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load draft and lineup data across multiple seasons."""
        if self.verbose:
            print("\n=== Loading Multi-Season Data ===")
        
        draft_parts = []
        lineup_parts = []
        
        for year_dir in sorted([
            p for p in self.raw_base.iterdir() 
            if p.is_dir() and p.name.isdigit() and len(p.name) == 4
        ]):
            year = int(year_dir.name)
            if year not in self.years:
                continue
            
            dpath = year_dir / "draft_data.csv"
            lpath = year_dir / "lineup_data.csv"
            
            if dpath.exists():
                draft_df = self._load_draft(dpath, year)
                draft_parts.append(draft_df)
            else:
                if self.verbose:
                    print(f"Skipping {year}: missing {dpath}")
            
            if lpath.exists():
                lineup_df = self._load_lineups(lpath, year)
                lineup_parts.append(lineup_df)
            else:
                if self.verbose:
                    print(f"Skipping {year}: missing {lpath}")
        
        if not draft_parts:
            raise FileNotFoundError(f"No draft_data.csv found under {self.raw_base}/<YEAR>/draft_data.csv")
        if not lineup_parts:
            raise FileNotFoundError(f"No lineup_data.csv found under {self.raw_base}/<YEAR>/lineup_data.csv")
        
        draft_all = pd.concat(draft_parts, ignore_index=True)
        lineups_all = pd.concat(lineup_parts, ignore_index=True)
        
        self.draft_raw = draft_all
        self.lineups_raw = lineups_all
        
        if self.verbose:
            print(f"Loaded {len(draft_all):,} draft records and {len(lineups_all):,} lineup records")
        
        return draft_all, lineups_all
    
    def _load_draft(self, path: Path, year: int) -> pd.DataFrame:
        """Load and normalize draft data for a single year."""
        df = pd.read_csv(path)
        required = ["League_ID", "Player", "Team", "Round", "Pick", "Overall", 
                   "Is_Autodrafted", "Auto_Draft_Type_ID"]
        # List comprehension: Build list of missing column names
        # Iterates through required columns, keeps only those not in df.columns
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} draft_data is missing columns: {missing}")
        
        df = df.copy()
        df["Year"] = int(year)
        df["Player_norm"] = df["Player"].map(lambda x: self.normalize_player_name(x, self.anchor_synonyms))
        df["League_ID"] = df["League_ID"].astype(int)
        df["Year"] = df["Year"].astype(int)
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce")
        df["Is_Autodrafted"] = pd.to_numeric(df["Is_Autodrafted"], errors="coerce").fillna(0).astype(int)
        return df
    
    def _load_lineups(self, path: Path, year: int) -> pd.DataFrame:
        """Load and normalize lineup data for a single year."""
        df = pd.read_csv(path)
        required = ["League_ID", "Week", "Team", "Player", "Slot", "Points"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} lineup_data is missing columns: {missing}")
        
        df = df.copy()
        df["Year"] = int(year)
        # map() with lambda: Apply normalize_player_name to each player name
        # Lambda needed because normalize_player_name requires the synonyms parameter
        df["Player_norm"] = df["Player"].map(lambda x: self.normalize_player_name(x, self.anchor_synonyms))
        df["League_ID"] = df["League_ID"].astype(int)
        df["Year"] = df["Year"].astype(int)
        df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
        df["Slot"] = df["Slot"].astype(str).str.strip()
        if "Is_Starter" in df.columns:
            df["Is_Starter"] = pd.to_numeric(df["Is_Starter"], errors="coerce")
        return df
    
    # ==================== DATA FILTERING ====================
    
    def filter_standard_leagues(
        self,
        draft: pd.DataFrame = None,
        lineups: pd.DataFrame = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter to leagues with standard starter configurations."""
        if draft is None:
            draft = self.draft_raw
        if lineups is None:
            lineups = self.lineups_raw
        
        if self.verbose:
            print("\n=== Filtering Standard Leagues ===")
        
        league_sig = self._infer_league_starter_signature(lineups)
        
        keep = (
            (league_sig["QB"] == self.expected_starters["QB"]) &
            (league_sig["RB"] == self.expected_starters["RB"]) &
            (league_sig["WR"] == self.expected_starters["WR"]) &
            (league_sig["TE"] == self.expected_starters["TE"]) &
            (league_sig["FLEX"] == self.expected_starters["FLEX"]) &
            (league_sig["K"] == self.expected_starters["K"]) &
            (league_sig["DST"] == self.expected_starters["D/ST"]) &
            (league_sig["OP"] == self.expected_starters["OP"]) &
            (league_sig["Starters_Total"] == self.expected_starters["STARTERS"])
        )
        league_sig["Keep_Standard"] = keep
        
        kept = league_sig[league_sig["Keep_Standard"]][["League_ID", "Year"]]
        draft_filt = draft.merge(kept, on=["League_ID", "Year"], how="inner")
        lineups_filt = lineups.merge(kept, on=["League_ID", "Year"], how="inner")
        
        if self.verbose:
            total = len(league_sig)
            k = int(league_sig["Keep_Standard"].sum())
            print(f"Keeping {k}/{total} league-years ({k/total:.1%})")
            if total - k > 0:
                print("Top non-standard signatures:")
                print(league_sig.loc[~league_sig["Keep_Standard"], "Signature"].value_counts().head(10))
        
        return draft_filt, lineups_filt, league_sig
    
    def _infer_league_starter_signature(self, lineups: pd.DataFrame) -> pd.DataFrame:
        """
        Infer starter configuration for each league-year.
        
        Analyzes lineup data to determine how many starters each league uses at each position.
        Uses the mode (most common) starter count across all team-weeks to identify the
        league's standard configuration.
        """
        df = lineups.copy()
        required = {"League_ID", "Year", "Team", "Week", "Slot", "Is_Starter"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"lineups missing columns for starter signature: {sorted(missing)}")
        
        df["Slot_norm"] = df["Slot"].map(self.normalize_slot)
        # Filter to only starter slots (Is_Starter == 1)
        starters = df[df["Is_Starter"].fillna(0).astype(int) == 1].copy()
        
        # Count starters by position for each team-week
        tw = (
            starters.groupby(["League_ID", "Year", "Team", "Week", "Slot_norm"])
            .size()
            .reset_index(name="n")
        )
        
        # Pivot to get position counts as columns (one row per team-week)
        pivot = (
            tw.pivot_table(
                index=["League_ID", "Year", "Team", "Week"],
                columns="Slot_norm",
                values="n",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        
        # Ensure all position columns exist (some leagues may not use all positions)
        for col in ["QB", "RB", "WR", "TE", "K", "D/ST", "FLEX", "OP"]:
            if col not in pivot.columns:
                pivot[col] = 0
        
        pivot["Starters_Total"] = pivot[["QB", "RB", "WR", "TE", "K", "D/ST", "FLEX", "OP"]].sum(axis=1)
        
        # Helper function to get mode (most common value) from a series
        def mode_int(s: pd.Series) -> int:
            vc = s.value_counts()
            return int(vc.index[0]) if len(vc) else 0
        
        # Aggregate to league-year level: use mode across all team-weeks
        # This handles cases where a team might have different lineups due to bye weeks
        league_sig = (
            pivot.groupby(["League_ID", "Year"], dropna=False)
            .agg(
                QB=("QB", mode_int),
                RB=("RB", mode_int),
                WR=("WR", mode_int),
                TE=("TE", mode_int),
                K=("K", mode_int),
                DST=("D/ST", mode_int),
                FLEX=("FLEX", mode_int),
                OP=("OP", mode_int),
                Starters_Total=("Starters_Total", mode_int),
            )
            .reset_index()
        )
        
        # Create human-readable signature string for easy identification
        league_sig["Signature"] = (
            "QB=" + league_sig["QB"].astype(str) +
            ",RB=" + league_sig["RB"].astype(str) +
            ",WR=" + league_sig["WR"].astype(str) +
            ",TE=" + league_sig["TE"].astype(str) +
            ",FLEX=" + league_sig["FLEX"].astype(str) +
            ",K=" + league_sig["K"].astype(str) +
            ",DST=" + league_sig["DST"].astype(str) +
            ",OP=" + league_sig["OP"].astype(str) +
            ",START=" + league_sig["Starters_Total"].astype(str)
        )
        return league_sig
    
    def filter_scoring_rule_outliers(
        self,
        draft: pd.DataFrame,
        lineups: pd.DataFrame,
        z_thresh: float = 3.5,
        min_anchors_hit: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter out league-years with unusual scoring rules using anchor players."""
        if self.verbose:
            print("\n=== Filtering Scoring Rule Outliers ===")
        
        pts = self._season_points_by_league_year(lineups)
        
        anchor_rows = []
        for pos, names in self.anchors_by_pos.items():
            for n in names:
                anchor_rows.append((pos, self.normalize_player_name(n, self.anchor_synonyms)))
        anchors = pd.DataFrame(anchor_rows, columns=["AnchorPos", "Player_norm"]).drop_duplicates()
        
        a = pts.merge(anchors, on="Player_norm", how="inner")
        if a.empty:
            meta = pd.DataFrame(columns=["League_ID", "Year", "Anchors_Hit", "Anchors_Outlier", "Drop"])
            return draft, lineups, meta
        
        # Compute robust z-scores using median and MAD (Median Absolute Deviation)
        # This is more robust to outliers than standard z-scores using mean/std
        # The constant 0.6745 scales MAD to match standard deviation for normal distributions
        def robust_z(s: pd.Series) -> pd.Series:
            x = s.astype(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            if mad == 0 or np.isnan(mad):
                return (x - med) * 0.0
            return 0.6745 * (x - med) / mad
        
        # Calculate robust z-scores grouped by year and player
        # This compares each league's anchor player points to the distribution across all leagues
        a["rz"] = a.groupby(["Year", "Player_norm"], dropna=False)["Season_Total_Points"].transform(robust_z)
        a["Is_Outlier"] = a["rz"].abs() >= z_thresh
        
        meta = (
            a.groupby(["League_ID", "Year"], dropna=False)
            .agg(
                Anchors_Hit=("Player_norm", "nunique"),
                AnchorRows=("Player_norm", "size"),
                Anchors_Outlier=("Is_Outlier", "sum"),
                # Lambda function: Find maximum absolute robust z-score for this league-year
                # Converts to numpy array, takes absolute value, finds max, handles empty case
                MaxAbsRZ=("rz", lambda s: float(np.nanmax(np.abs(s.to_numpy()))) if len(s) else np.nan),
            )
            .reset_index()
        )
        
        meta["Drop"] = (meta["Anchors_Hit"] < min_anchors_hit) | (meta["Anchors_Outlier"] >= 1)
        
        kept = meta[~meta["Drop"]][["League_ID", "Year"]]
        draft_filt = draft.merge(kept, on=["League_ID", "Year"], how="inner")
        lineups_filt = lineups.merge(kept, on=["League_ID", "Year"], how="inner")
        
        if self.verbose:
            total = meta.shape[0]
            kept_n = kept.shape[0]
            print(f"Keeping {kept_n}/{total} league-years ({kept_n/total:.1%})")
            if total - kept_n > 0:
                print("Examples of dropped league-years:")
                print(meta[meta["Drop"]].sort_values(
                    ["Anchors_Hit", "Anchors_Outlier", "MaxAbsRZ"], 
                    ascending=[True, False, False]
                ).head(10))
        
        return draft_filt, lineups_filt, meta
    
    def _season_points_by_league_year(self, lineups: pd.DataFrame) -> pd.DataFrame:
        """Compute season total points by league-year-player."""
        df = lineups.copy()
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
        return (
            df.groupby(["League_ID", "Year", "Player_norm"], dropna=False)["Points"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"Points": "Season_Total_Points"})
        )
    
    def filter_draft_length(
        self,
        draft: pd.DataFrame,
        lineups: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter to leagues with expected draft length."""
        if self.verbose:
            print(f"\n=== Filtering Draft Length ({self.expected_draft_length}) ===")
        
        d = draft.copy()
        if "Draft_Length" not in d.columns:
            d = self._add_draft_length(d)
        
        kept = (
            d.groupby(["League_ID", "Year"], dropna=False)["Draft_Length"]
            .max()
            .reset_index()
        )
        kept = kept[kept["Draft_Length"] == int(self.expected_draft_length)][["League_ID", "Year"]]
        
        d2 = d.merge(kept, on=["League_ID", "Year"], how="inner")
        l2 = lineups.merge(kept, on=["League_ID", "Year"], how="inner")
        
        if self.verbose:
            total = d.groupby(["League_ID", "Year"]).ngroups
            kept_n = kept.shape[0]
            print(f"Keeping {kept_n}/{total} league-years ({kept_n/total:.1%})")
        
        return d2, l2
    
    def _add_draft_length(self, draft_df: pd.DataFrame) -> pd.DataFrame:
        """Add draft length column."""
        out = draft_df.copy()
        out["Draft_Length"] = (
            out.groupby(["League_ID", "Year"], dropna=False)["Overall"]
            .transform("max")
        )
        out["Draft_Length"] = pd.to_numeric(out["Draft_Length"], errors="coerce").fillna(-1).astype(int)
        return out
    
    # ==================== DATA ENRICHMENT ====================
    
    def enrich_draft_data(
        self,
        draft: pd.DataFrame,
        lineups: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich draft data with position and season total points."""
        if self.verbose:
            print("\n=== Enriching Draft Data ===")
        
        d = draft.copy()
        # map() with lambda: Re-normalize player names (in case they weren't normalized before)
        # Lambda needed to pass synonyms parameter to normalize_player_name
        d["Player_norm"] = d["Player_norm"].map(
            lambda x: self.normalize_player_name(x, self.anchor_synonyms)
        )
        
        pos = self._infer_position_from_slots(lineups)
        pts = self._season_points_from_lineups(lineups)
        
        out = d.merge(pos, on=["League_ID", "Year", "Player_norm"], how="left")
        out = out.merge(pts, on=["League_ID", "Year", "Player_norm"], how="left")
        
        # Deduplicate after merges
        # Merges can create duplicates if source data has duplicates (e.g., same player
        # drafted multiple times in same league, or multiple lineup entries for same player)
        # Keep first occurrence of each League_ID/Year/Overall combination
        before_dedup = len(out)
        out = out.drop_duplicates(subset=["League_ID", "Year", "Overall"], keep="first")
        after_dedup = len(out)
        
        if before_dedup != after_dedup and self.verbose:
            print(f"  Dropped {before_dedup - after_dedup:,} duplicate rows after enrichment")
        
        out["Season_Total_Points"] = pd.to_numeric(out["Season_Total_Points"], errors="coerce").fillna(0.0)
        
        cols = [
            "League_ID", "Year", "Team", "Player",
            "Round", "Pick", "Overall",
            "Draft_Length",
            "Position", "Season_Total_Points",
            "Is_Autodrafted", "Auto_Draft_Type_ID",
            "Player_norm",
        ]
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        
        self.draft_enriched = out[cols].copy()
        
        if self.verbose:
            print(f"Enriched {len(self.draft_enriched):,} draft records")
        
        return self.draft_enriched
    
    def _infer_position_from_slots(self, lineups: pd.DataFrame) -> pd.DataFrame:
        """
        Infer player position from lineup slots.
        
        Uses the most common core position slot (QB/RB/WR/TE/K/D/ST) a player appears in.
        Prioritizes core positions over flex slots to get the player's true position.
        """
        x = lineups.copy()
        x["Slot_norm"] = x["Slot"].map(self.normalize_slot)
        # Identify core positions (non-flex slots that indicate true position)
        x["Is_core_pos"] = x["Slot_norm"].isin({"QB", "RB", "WR", "TE", "K", "D/ST"})
        
        # Count appearances by slot for each player
        counts = (
            x.groupby(["League_ID", "Year", "Player_norm", "Slot_norm", "Is_core_pos"], dropna=False)
            .size().reset_index(name="n")
            # Sort to prioritize core positions, then by frequency
            .sort_values(by=["League_ID", "Year", "Player_norm", "Is_core_pos", "n"],
                         ascending=[True, True, True, False, False])
        )
        
        # Take first (most common core position) for each player
        top = (
            counts.drop_duplicates(subset=["League_ID", "Year", "Player_norm"])
            .rename(columns={"Slot_norm": "Position"})[["League_ID", "Year", "Player_norm", "Position"]]
        )
        return top
    
    def _season_points_from_lineups(self, lineups: pd.DataFrame) -> pd.DataFrame:
        """Compute season total points from lineups."""
        df = lineups.copy()
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
        return (
            df.groupby(["League_ID", "Year", "Player_norm"], dropna=False)["Points"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"Points": "Season_Total_Points"})
        )
    
    # ==================== OPTIMAL LINEUP COMPUTATION ====================
    
    def compute_optimal_startable_points(
        self,
        lineups: pd.DataFrame,
        slot_counts: Dict[str, int] = None,
        flex_eligible: Set[str] = None,
        status_every: int = 250
    ) -> pd.DataFrame:
        """Compute optimal startable points for each team-week."""
        if slot_counts is None:
            slot_counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1}
        if flex_eligible is None:
            flex_eligible = {"RB", "WR", "TE"}
        
        if self.verbose:
            print("\n=== Computing Optimal Startable Points ===")
            t_all = time.time()
        
        df = lineups.copy()
        df["Player_norm"] = df["Player_norm"] if "Player_norm" in df.columns else df["Player"].astype(str).str.strip()
        df["Slot"] = df["Slot"].map(self._normalize_slot_defense)
        
        if "Year" not in df.columns:
            raise ValueError("lineups must include a 'Year' column for multi-season processing.")
        
        if self.verbose:
            print(f"Processing {len(df):,} lineup records")
        
        pos = self._infer_player_position_by_core_starts(df)
        pw = self._build_player_week_points(df)
        
        pw = pw.merge(pos, on=["League_ID", "Year", "Team", "Player_norm"], how="left")
        pw["Position"] = pw["Position"].fillna("UNKNOWN")
        
        gb = pw.groupby(["League_ID", "Year", "Team", "Week"], sort=False)
        n_groups = gb.ngroups
        
        if self.verbose:
            print(f"Optimizing {n_groups:,} team-weeks...")
        
        # Process each team-week group and track progress
        # t0 tracks start time for calculating processing rate
        t0 = time.time()
        selected_parts = []
        # enumerate(gb, start=1) gives us (1, (key, group)), (2, (key, group)), etc.
        # We use start=1 so progress shows 1/N instead of 0/N
        for i, (_, g) in enumerate(gb, start=1):
            # Print progress at intervals, on first iteration, and on last iteration
            # This ensures we always see start and completion, plus periodic updates
            if i % status_every == 0 or i == 1 or i == n_groups:
                elapsed = time.time() - t0
                # Calculate rate: groups processed per second
                # Handle division by zero (shouldn't happen, but defensive programming)
                rate = i / elapsed if elapsed > 0 else float("inf")
                if self.verbose:
                    print(f"  - progress: {i:,}/{n_groups:,} groups ({rate:.1f} groups/s)")
            # Process this team-week's optimal lineup and append to results list
            selected_parts.append(self._choose_optimal_lineup_for_group(g, slot_counts, flex_eligible))
        
        selected = pd.concat(selected_parts, ignore_index=True)
        
        if self.verbose:
            print(f"Completed in {time.time() - t_all:.2f}s")
        
        return selected
    
    def _normalize_slot_defense(self, slot: str) -> str:
        """Normalize defense slot names."""
        slot = str(slot).strip()
        return {"DST": "D/ST", "DEF": "D/ST"}.get(slot, slot)
    
    def _infer_player_position_by_core_starts(self, lineups: pd.DataFrame) -> pd.DataFrame:
        """Infer player position from core slot starts."""
        CORE = {"QB", "RB", "WR", "TE", "K", "D/ST"}
        df = lineups.copy()
        df["Slot"] = df["Slot"].map(self._normalize_slot_defense)
        
        if "Year" not in df.columns:
            raise ValueError("lineups must include a 'Year' column for multi-season processing.")
        
        core_rows = df[df["Slot"].isin(CORE)].copy()
        if core_rows.empty:
            raise ValueError("No core slots found in lineup_data Slot column (QB/RB/WR/TE/K/DST).")
        
        # Group by player and get most common slot (position) they played
        # Lambda function: value_counts() counts occurrences, .index[0] gets most common
        # This handles cases where a player might occasionally play a different position
        pos = (
            core_rows.groupby(["League_ID", "Year", "Team", "Player_norm"])["Slot"]
            .agg(lambda s: s.value_counts().index[0])
            .reset_index()
            .rename(columns={"Slot": "Position"})
        )
        return pos
    
    def _build_player_week_points(self, lineups: pd.DataFrame) -> pd.DataFrame:
        """Build player-week points table."""
        df = lineups.copy()
        df["Slot"] = df["Slot"].map(self._normalize_slot_defense)
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0.0)
        
        if "Year" not in df.columns:
            raise ValueError("lineups must include a 'Year' column for multi-season processing.")
        
        pw = (
            df.groupby(["League_ID", "Year", "Team", "Week", "Player_norm"], dropna=False)["Points"]
            .max()
            .reset_index()
            .rename(columns={"Points": "WeekPoints"})
        )
        return pw
    
    def _choose_optimal_lineup_for_group(
        self,
        g: pd.DataFrame,
        slot_counts: Dict[str, int],
        flex_eligible: Set[str]
    ) -> pd.DataFrame:
        """
        Choose optimal lineup for a single team-week using greedy selection.
        
        Selects the highest-scoring players at each position, respecting position limits.
        FLEX positions are filled after all required positions, using remaining eligible players.
        This is a greedy algorithm that may not be globally optimal but is fast and effective.
        """
        g = g.copy()
        g["SelectedOptimal"] = False
        used = set()  # Track players already selected to avoid duplicates
        
        def select_top(pos, n):
            """Select top N players at a specific position, excluding already-used players."""
            nonlocal used
            if n <= 0:
                return []
            # Filter by position, exclude used players, sort by points descending
            cand = g[(g["Position"] == pos) & (~g["Player_norm"].isin(used))].sort_values("WeekPoints", ascending=False)
            chosen = cand.head(n)["Player_norm"].tolist()
            used.update(chosen)  # Mark as used
            return chosen
        
        def select_flex(n):
            """Select top N players from flex-eligible positions (RB/WR/TE), excluding used players."""
            nonlocal used
            if n <= 0:
                return []
            # Filter by flex-eligible positions, exclude used players, sort by points descending
            cand = g[(g["Position"].isin(flex_eligible)) & (~g["Player_norm"].isin(used))].sort_values("WeekPoints", ascending=False)
            chosen = cand.head(n)["Player_norm"].tolist()
            used.update(chosen)  # Mark as used
            return chosen
        
        # Fill required positions first (QB, RB, WR, TE, K, D/ST)
        selected = []
        for pos in ["QB", "RB", "WR", "TE", "K", "D/ST"]:
            selected += select_top(pos, slot_counts.get(pos, 0))
        # Fill FLEX positions last (from remaining RB/WR/TE)
        selected += select_flex(slot_counts.get("FLEX", 0))
        
        g.loc[g["Player_norm"].isin(selected), "SelectedOptimal"] = True
        return g
    
    def add_valid_points(
        self,
        draft_enriched: pd.DataFrame,
        optimal_selected: pd.DataFrame
    ) -> pd.DataFrame:
        """Add season total valid points to draft data."""
        if self.verbose:
            print("\n=== Adding Valid Points ===")
        
        for col in ["Year", "League_ID", "Team", "Player_norm"]:
            if col not in draft_enriched.columns:
                raise ValueError(f"draft_enriched missing required column '{col}' for multi-season merge.")
        if "Year" not in optimal_selected.columns:
            raise ValueError("optimal_selected missing 'Year' column.")
        
        valid = (
            optimal_selected[optimal_selected["SelectedOptimal"]]
            .groupby(["League_ID", "Year", "Team", "Player_norm"], dropna=False)["WeekPoints"]
            .sum()
            .reset_index()
            .rename(columns={"WeekPoints": "Season_Total_Points_Valid"})
        )
        
        out = draft_enriched.merge(valid, on=["League_ID", "Year", "Team", "Player_norm"], how="left")
        out["Season_Total_Points_Valid"] = pd.to_numeric(out["Season_Total_Points_Valid"], errors="coerce").fillna(0.0)
        
        self.draft_with_valid = out
        
        if self.verbose:
            print(f"Added valid points to {len(self.draft_with_valid):,} draft records")
        
        return self.draft_with_valid
    
    # ==================== FULL PIPELINE ====================
    
    def run_full_pipeline(
        self,
        clean_data: bool = True,
        filter_standard: bool = True,
        filter_scoring: bool = True,
        filter_draft_length: bool = True,
        compute_optimal: bool = True,
        save_intermediate: bool = True
    ) -> pd.DataFrame:
        """
        Run the complete analysis pipeline.
        
        Args:
            clean_data: If True, clean raw data (drop duplicates)
            filter_standard: If True, filter to standard leagues
            filter_scoring: If True, filter scoring rule outliers
            filter_draft_length: If True, filter by draft length
            compute_optimal: If True, compute optimal lineups
            save_intermediate: If True, save intermediate results
        
        Returns:
            Final enriched draft DataFrame with valid points
        """
        if self.verbose:
            print("=" * 70)
            print("DRAFT VALUE ANALYSIS PIPELINE")
            print("=" * 70)
        
        # Step 1: Clean data
        if clean_data:
            self.clean_raw_data()
        
        # Step 2: Load data
        self.load_multi_season_data()
        
        # Step 3: Add draft length
        self.draft_raw = self._add_draft_length(self.draft_raw)
        
        # Step 4: Filter standard leagues
        draft_filt, lineups_filt, _ = self.filter_standard_leagues()
        
        # Step 5: Filter scoring outliers
        if filter_scoring:
            draft_filt, lineups_filt, _ = self.filter_scoring_rule_outliers(draft_filt, lineups_filt)
        
        # Step 6: Filter draft length
        if filter_draft_length:
            draft_filt, lineups_filt = self.filter_draft_length(draft_filt, lineups_filt)
        
        # Store filtered lineups for later use (waiver wire, start/sit analysis)
        self.lineups_filt = lineups_filt
        
        # Step 7: Enrich draft data
        draft_enriched = self.enrich_draft_data(draft_filt, lineups_filt)
        
        if save_intermediate:
            draft_enriched_path = self.out_dir / "draft_enriched_filtered.csv"
            draft_enriched.to_csv(draft_enriched_path, index=False)
            if self.verbose:
                print(f"\n[saved] {draft_enriched_path} ({len(draft_enriched):,} rows)")
        
        # Step 8: Compute optimal lineups
        if compute_optimal:
            optimal_selected = self.compute_optimal_startable_points(
                lineups_filt,
                slot_counts={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1},
                flex_eligible={"RB", "WR", "TE"}
            )
            
            # Store optimal selections for later use (waiver wire, start/sit analysis)
            self.optimal_selected = optimal_selected
            
            # Step 9: Add valid points
            draft_with_valid = self.add_valid_points(draft_enriched, optimal_selected)
            
            if save_intermediate:
                draft_with_valid_path = self.out_dir / "draft_with_valid_points_filtered.csv"
                draft_with_valid.to_csv(draft_with_valid_path, index=False)
                if self.verbose:
                    print(f"[saved] {draft_with_valid_path} ({len(draft_with_valid):,} rows)")
            
            return draft_with_valid
        
        return draft_enriched
    
    # ==================== EXPECTED VALUE CALCULATIONS ====================
    
    def compute_expected_values(
        self,
        draft_with_valid: pd.DataFrame = None,
        estimator: str = "trimmed_mean",
        trim: float = 0.10,
        smooth_window: int = 5,
        poly_degree: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute expected values using multiple baselines.
        
        Returns:
            expected_by_pick_year: Per-year expected values
            expected_by_pick_pooled: Pooled denoised expected values
            expected_by_pick_poly: Polynomial fit to pooled denoised
        """
        if draft_with_valid is None:
            draft_with_valid = self.draft_with_valid
        
        if draft_with_valid is None:
            raise ValueError("No draft data with valid points. Run pipeline first.")
        
        if self.verbose:
            print("\n=== Computing Expected Values ===")
        
        # Per-year baseline
        expected_by_pick_year = self._expected_by_pick_by_year(draft_with_valid)
        
        # Pooled denoised baseline
        expected_by_pick_pooled = self._expected_by_pick_pooled_denoised(
            draft_with_valid, estimator=estimator, trim=trim, smooth_window=smooth_window
        )
        
        # Polynomial fit
        expected_by_pick_poly = self._fit_polynomial_baseline(
            expected_by_pick_pooled, y_col="Expected_Smoothed", w_col="N", degree=poly_degree
        )
        
        return expected_by_pick_year, expected_by_pick_pooled, expected_by_pick_poly
    
    def _expected_by_pick_by_year(self, draft_with_valid: pd.DataFrame) -> pd.DataFrame:
        """Compute expected values per year."""
        df = draft_with_valid.copy()
        team_auto = (
            df.groupby(["League_ID", "Year", "Team"], dropna=False)["Is_Autodrafted"]
            .agg(Total_Picks="size", Auto_Picks="sum")
            .reset_index()
        )
        team_auto["Is_Fully_Autodrafted"] = team_auto["Total_Picks"].eq(team_auto["Auto_Picks"])
        
        df = df.merge(team_auto[["League_ID", "Year", "Team", "Is_Fully_Autodrafted"]],
                     on=["League_ID", "Year", "Team"], how="left")
        df_auto = df[df["Is_Fully_Autodrafted"]].copy()
        
        expected = (
            df_auto.groupby(["Year", "Overall"], dropna=False)["Season_Total_Points_Valid"]
            .agg(
                Expected_Valid_Points="mean",
                Std_Valid_Points="std",
                N="count"
            )
            .reset_index()
            .sort_values(["Year", "Overall"])
        )
        return expected
    
    def _expected_by_pick_pooled_denoised(
        self,
        draft_with_valid: pd.DataFrame,
        estimator: str = "trimmed_mean",
        trim: float = 0.10,
        smooth_window: int = 5
    ) -> pd.DataFrame:
        """
        Compute pooled denoised expected values across all years.
        
        Uses only fully-autodrafted teams to establish baseline expected value at each pick.
        Applies robust statistical estimators (trimmed mean, winsorized mean) to reduce
        impact of outliers, then smooths the curve with a rolling average.
        """
        df = draft_with_valid.copy()
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce").astype(int)
        df["Is_Autodrafted"] = pd.to_numeric(df["Is_Autodrafted"], errors="coerce").fillna(0).astype(int)
        df["Season_Total_Points_Valid"] = pd.to_numeric(df["Season_Total_Points_Valid"], errors="coerce")
        
        # Identify fully-autodrafted teams (all picks were autodrafted)
        team_auto = (
            df.groupby(["League_ID", "Year", "Team"], dropna=False)["Is_Autodrafted"]
            .agg(Total_Picks="size", Autodrafted_Picks="sum")
            .reset_index()
        )
        team_auto["Is_Fully_Autodrafted"] = team_auto["Total_Picks"].eq(team_auto["Autodrafted_Picks"])
        df = df.merge(team_auto[["League_ID", "Year", "Team", "Is_Fully_Autodrafted"]],
                     on=["League_ID", "Year", "Team"], how="left")
        
        df_auto = df[df["Is_Fully_Autodrafted"]].copy()
        if df_auto.empty:
            raise ValueError("No fully-autodrafted teams found.")
        
        # Select aggregation function based on estimator type
        # Trimmed mean removes extreme values, winsorized mean caps them
        # Select aggregation function based on estimator type
        # Lambda functions convert pandas Series to numpy array, apply transformation, return float
        if estimator == "mean":
            agg_func = "mean"
        elif estimator == "median":
            # nanmedian ignores NaN values when computing median
            agg_func = lambda s: float(np.nanmedian(s.to_numpy()))
        elif estimator == "trimmed_mean":
            # trim_mean removes extreme values (top/bottom trim%) before computing mean
            agg_func = lambda s: float(trim_mean(s.dropna().to_numpy(), proportiontocut=trim))
        elif estimator == "winsor_mean":
            # winsorize caps extreme values (limits=trim) instead of removing them
            agg_func = lambda s: float(np.mean(winsorize(s.dropna().to_numpy(), limits=trim)))
        else:
            raise ValueError("estimator must be one of: mean, median, trimmed_mean, winsor_mean")
        
        # Aggregate by pick number across all years
        pooled = (
            df_auto.groupby("Overall", dropna=False)["Season_Total_Points_Valid"]
            .agg(Expected_Valid_Points=agg_func, Std_Valid_Points="std", N="count")
            .reset_index()
            .sort_values("Overall")
        )
        
        # Apply rolling average smoothing to reduce noise
        # Center=True means the window is centered on each point
        if smooth_window and smooth_window > 1:
            pooled["Expected_Smoothed"] = (
                pooled["Expected_Valid_Points"].rolling(window=smooth_window, center=True, min_periods=1).mean()
            )
        else:
            pooled["Expected_Smoothed"] = pooled["Expected_Valid_Points"]
        
        return pooled
    
    def _fit_polynomial_baseline(
        self,
        expected_pooled: pd.DataFrame,
        x_col: str = "Overall",
        y_col: str = "Expected_Valid_Points",
        w_col: str = "N",
        degree: int = 4
    ) -> pd.DataFrame:
        """
        Fit polynomial baseline to expected curve using weighted least squares.
        
        Weights the fit by sample size (N) at each pick, so picks with more data
        have more influence on the curve. This creates a smooth baseline that
        captures the general trend while reducing impact of noisy individual picks.
        """
        df = expected_pooled.copy()
        
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        
        # Use sample size as weights (more data = higher weight)
        if w_col in df.columns:
            w = pd.to_numeric(df[w_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        else:
            w = np.ones_like(x)
        
        # Filter out invalid values (NaN, inf, zero weights)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x, y, w = x[m], y[m], w[m]
        
        if len(x) < degree + 2:
            raise ValueError(f"Not enough points to fit degree={degree} polynomial.")
        
        # Fit weighted polynomial using least squares
        coeff = np.polyfit(x, y, deg=degree, w=w)
        p = np.poly1d(coeff)
        
        # Evaluate polynomial at all x values (including those filtered out)
        df["Poly_Expected"] = p(pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float))
        df["Poly_Degree"] = degree
        return df
    
    # ==================== SCORING ====================
    
    def score_picks(
        self,
        draft_with_valid: pd.DataFrame = None,
        expected_by_pick_year: pd.DataFrame = None,
        expected_by_pick_pooled: pd.DataFrame = None,
        expected_by_pick_poly: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Score all picks against multiple baselines."""
        if draft_with_valid is None:
            draft_with_valid = self.draft_with_valid
        
        if draft_with_valid is None:
            raise ValueError("No draft data with valid points. Run pipeline first.")
        
        if expected_by_pick_year is None or expected_by_pick_pooled is None or expected_by_pick_poly is None:
            if self.verbose:
                print("Computing expected values first...")
            expected_by_pick_year, expected_by_pick_pooled, expected_by_pick_poly = self.compute_expected_values(draft_with_valid)
        
        if self.verbose:
            print("\n=== Scoring Picks ===")
        
        df = draft_with_valid.copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce").astype(int)
        
        # Per-year baseline
        expected_year = expected_by_pick_year.rename(columns={
            "Expected_Valid_Points": "Expected_Year",
            "Std_Valid_Points": "Std_Year",
            "N": "N_Year",
        })
        
        draft_scored = df.merge(
            expected_year[["Year", "Overall", "Expected_Year", "Std_Year", "N_Year"]],
            on=["Year", "Overall"],
            how="left",
        )
        
        draft_scored["Points_Added_Year"] = draft_scored["Season_Total_Points_Valid"] - draft_scored["Expected_Year"]
        draft_scored["Z_Year"] = np.where(
            (draft_scored["Std_Year"].notna()) & (draft_scored["Std_Year"] > 0),
            draft_scored["Points_Added_Year"] / draft_scored["Std_Year"],
            np.nan
        )
        
        # Pooled baselines
        expected_pooled_renamed = expected_by_pick_poly.rename(columns={
            "Expected_Smoothed": "Expected_Pooled_Denoised",
            "Std_Valid_Points": "Std_Pooled",
            "N": "N_Pooled",
            "Poly_Expected": "Expected_Poly",
        })
        
        draft_scored = draft_scored.merge(
            expected_pooled_renamed[["Overall", "Expected_Pooled_Denoised", "Expected_Poly", "Std_Pooled", "N_Pooled"]],
            on="Overall",
            how="left",
        )
        
        draft_scored["Points_Added_Pooled"] = (
            draft_scored["Season_Total_Points_Valid"] - draft_scored["Expected_Pooled_Denoised"]
        )
        draft_scored["Z_Pooled"] = np.where(
            (draft_scored["Std_Pooled"].notna()) & (draft_scored["Std_Pooled"] > 0),
            draft_scored["Points_Added_Pooled"] / draft_scored["Std_Pooled"],
            np.nan
        )
        
        draft_scored["Points_Added_Poly"] = (
            draft_scored["Season_Total_Points_Valid"] - draft_scored["Expected_Poly"]
        )
        draft_scored["Z_Poly"] = np.where(
            (draft_scored["Std_Pooled"].notna()) & (draft_scored["Std_Pooled"] > 0),
            draft_scored["Points_Added_Poly"] / draft_scored["Std_Pooled"],
            np.nan
        )
        
        self.draft_scored = draft_scored
        
        if self.verbose:
            print(f"Scored {len(draft_scored):,} picks")
        
        return draft_scored
    
    # ==================== VISUALIZATION METHODS ====================
    
    def plot_team_total_valid_points_distribution(
        self,
        draft_with_valid: pd.DataFrame = None,
        fully_autodraft_only: bool = None,
        bins: int = 20,
        title: str = None
    ) -> pd.DataFrame:
        """
        Plot distribution of team total valid points.
        
        Args:
            draft_with_valid: Draft data with valid points (uses self.draft_with_valid if None)
            fully_autodraft_only: True for autodrafted only, False for non-autodrafted, None for all
            bins: Number of histogram bins
            title: Plot title (auto-generated if None)
        
        Returns:
            DataFrame with team totals
        """
        if draft_with_valid is None:
            draft_with_valid = self.draft_with_valid
        
        if draft_with_valid is None:
            raise ValueError("No draft data with valid points. Run pipeline first.")
        
        team_totals = self._compute_team_total_valid_points(draft_with_valid, fully_autodraft_only)
        
        x = team_totals["Total_Valid_Points"].astype(float).to_numpy()
        avg = float(np.mean(x))
        
        if title is None:
            if fully_autodraft_only is True:
                title = "Autodrafted Teams: Total Valid Points Distribution"
            elif fully_autodraft_only is False:
                title = "NOT 100% Autodrafted Teams: Total Valid Points Distribution"
            else:
                title = "All Teams: Total Valid Points Distribution"
            title += f" (n={len(x)})"
        
        plt.figure(figsize=(10, 5))
        plt.hist(x, bins=bins, density=True, alpha=0.6)
        plt.axvline(avg, linestyle="--", color="red", label=f"Mean = {avg:.1f}")
        plt.xlabel("Total Valid Points (Optimal-Start Weeks Only)")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        if self.verbose:
            print(f"Average Total Valid Points: {avg:.2f}")
        
        return team_totals
    
    def _compute_team_total_valid_points(
        self,
        draft_with_valid: pd.DataFrame,
        fully_autodraft_only: bool = None
    ) -> pd.DataFrame:
        """Compute team total valid points."""
        df = draft_with_valid.copy()
        
        team_auto = (
            df.groupby(["League_ID", "Year", "Team"], dropna=False)["Is_Autodrafted"]
            .agg(Total_Picks="size", Autodrafted_Picks="sum")
            .reset_index()
        )
        team_auto["Is_Fully_Autodrafted"] = team_auto["Total_Picks"].eq(team_auto["Autodrafted_Picks"])
        df = df.merge(team_auto[["League_ID", "Year", "Team", "Is_Fully_Autodrafted"]],
                     on=["League_ID", "Year", "Team"], how="left")
        
        if fully_autodraft_only is True:
            df = df[df["Is_Fully_Autodrafted"]].copy()
        elif fully_autodraft_only is False:
            df = df[~df["Is_Fully_Autodrafted"]].copy()
        
        team_totals = (
            df.groupby(["League_ID", "Year", "Team"], dropna=False)["Season_Total_Points_Valid"]
            .sum()
            .reset_index()
            .rename(columns={"Season_Total_Points_Valid": "Total_Valid_Points"})
            .sort_values("Total_Valid_Points", ascending=False)
        )
        
        if team_totals.empty:
            raise ValueError("No teams found after filtering.")
        return team_totals
    
    def plot_expected_by_pick(
        self,
        expected_by_pick_pooled: pd.DataFrame = None,
        expected_by_pick_poly: pd.DataFrame = None,
        show_variance: bool = True,
        zoom_first_25: bool = False
    ) -> None:
        """
        Plot expected valid points by pick with variance bands.
        
        Args:
            expected_by_pick_pooled: Pooled expected values (uses computed if None)
            expected_by_pick_poly: Polynomial fit (uses computed if None)
            show_variance: If True, show variance bands (requires distribution calculation)
            zoom_first_25: If True, create zoomed plot of first 25 picks
        """
        if expected_by_pick_pooled is None or expected_by_pick_poly is None:
            if self.draft_with_valid is None:
                raise ValueError("No draft data. Run pipeline first.")
            _, expected_by_pick_pooled, expected_by_pick_poly = self.compute_expected_values()
        
        # Plot 1: Denoised pooled curve
        plt.figure(figsize=(10, 5))
        plt.plot(expected_by_pick_pooled["Overall"], expected_by_pick_pooled["Expected_Smoothed"], 
                linewidth=2, label="Denoised (rolling)")
        plt.plot(expected_by_pick_poly["Overall"], expected_by_pick_poly["Poly_Expected"], 
                linewidth=2, linestyle="--", label=f"Polynomial (deg={int(expected_by_pick_poly['Poly_Degree'].iloc[0])})")
        plt.xlabel("Overall Pick")
        plt.ylabel("Valid Season Points")
        plt.title("Expected Valid Points by Pick — Denoised vs Polynomial Baseline")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot 2: With variance bands (if requested)
        if show_variance and self.draft_with_valid is not None:
            dist = self._expected_distribution(self.draft_with_valid)
            
            plt.figure(figsize=(10, 5))
            plt.plot(dist["Overall"], dist["Mean"], label="Expected (Mean)", linewidth=2)
            plt.fill_between(dist["Overall"], dist["P25"], dist["P75"], alpha=0.25, label="25–75% range")
            plt.fill_between(dist["Overall"], dist["P10"], dist["P90"], alpha=0.15, label="10–90% range")
            plt.xlabel("Overall Pick")
            plt.ylabel("Valid Season Points")
            plt.title("Autodraft Expected Valid Points by Pick (with Variance)")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
            
            # Plot 3: Zoom first 25 picks
            if zoom_first_25:
                dist_25 = dist[dist["Overall"] <= 25].copy()
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.plot(dist_25["Overall"], dist_25["Mean"], label="Expected (Mean)", linewidth=2)
                ax.fill_between(dist_25["Overall"], dist_25["P25"], dist_25["P75"], alpha=0.30, label="25–75% range")
                ax.fill_between(dist_25["Overall"], dist_25["P10"], dist_25["P90"], alpha=0.18, label="10–90% range")
                ax.set_xlabel("Overall Pick", fontsize=12)
                ax.set_ylabel("Valid Season Points", fontsize=12)
                ax.set_title("Autodraft Expected Valid Points by Pick\nFirst 25 Picks (Variance Zoom)", fontsize=14)
                
                for _, r in dist_25.iterrows():
                    ax.text(
                        r["Overall"], r["P10"] - 5,
                        f"{int(r['Unique_Players'])} \nUnique\nPlayers\n(n={int(r['N_Picks'])})",
                        ha="center", va="top", fontsize=9, alpha=0.7
                    )
                
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                plt.show()
    
    def _expected_distribution(self, draft_with_valid: pd.DataFrame) -> pd.DataFrame:
        """Compute expected value distribution with percentiles."""
        df = draft_with_valid.copy()
        team_auto = (
            df.groupby(["League_ID", "Year", "Team"], dropna=False)["Is_Autodrafted"]
            .agg(Total_Picks="size", Auto_Picks="sum")
            .reset_index()
        )
        team_auto["Is_Fully_Autodrafted"] = team_auto["Total_Picks"].eq(team_auto["Auto_Picks"])
        df = df.merge(team_auto[["League_ID", "Year", "Team", "Is_Fully_Autodrafted"]],
                     on=["League_ID", "Year", "Team"], how="left")
        df = df[df["Is_Fully_Autodrafted"]].copy()
        
        dist = (
            df.groupby("Overall", dropna=False)
            .agg(
                Mean=("Season_Total_Points_Valid", "mean"),
                Std=("Season_Total_Points_Valid", "std"),
                N_Picks=("Season_Total_Points_Valid", "count"),
                Unique_Players=("Player_norm", "nunique"),
                P10=("Season_Total_Points_Valid", lambda s: s.quantile(0.10)),
                P25=("Season_Total_Points_Valid", lambda s: s.quantile(0.25)),
                P50=("Season_Total_Points_Valid", lambda s: s.quantile(0.50)),
                P75=("Season_Total_Points_Valid", lambda s: s.quantile(0.75)),
                P90=("Season_Total_Points_Valid", lambda s: s.quantile(0.90)),
            )
            .reset_index()
            .sort_values("Overall")
        )
        return dist
    
    def plot_human_advantage_by_year(
        self,
        draft_scored: pd.DataFrame = None,
        baseline: str = "Poly"
    ) -> pd.DataFrame:
        """
        Plot human draft advantage by year.
        
        Args:
            draft_scored: Scored draft data (uses self.draft_scored if None)
            baseline: Which baseline to use ("Poly", "Pooled", or "Year")
        
        Returns:
            Summary DataFrame
        """
        if draft_scored is None:
            draft_scored = self.draft_scored
        
        if draft_scored is None:
            raise ValueError("No scored draft data. Run score_picks() first.")
        
        manual = draft_scored[draft_scored["Is_Autodrafted"] == 0].copy()
        
        if baseline == "Poly":
            points_col = "Points_Added_Poly"
        elif baseline == "Pooled":
            points_col = "Points_Added_Pooled"
        else:
            points_col = "Points_Added_Year"
        
        summary = (
            manual.groupby("Year", dropna=False)
            .agg(
                Manual_Picks=(points_col, "size"),
                Avg_Points_Added=(points_col, "mean"),
            )
            .reset_index()
            .sort_values("Year")
        )
        
        x = np.arange(len(summary))
        w = 0.6
        
        plt.figure(figsize=(10, 5))
        plt.bar(x, summary["Avg_Points_Added"], width=w, label=f"{baseline} Baseline")
        plt.axhline(0, linestyle="--", color="gray", alpha=0.6)
        plt.xticks(x, summary["Year"])
        plt.ylabel("Avg Points Above Expected (per manual pick)")
        plt.title("Human Draft Advantage by Year")
        plt.legend()
        
        # Annotate n
        for i, r in summary.iterrows():
            y_top = r["Avg_Points_Added"]
            plt.text(i, y_top, f"n={int(r['Manual_Picks'])}", ha="center", 
                    va="bottom" if y_top >= 0 else "top", fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return summary
    
    def plot_human_advantage_by_round(
        self,
        draft_scored: pd.DataFrame = None,
        baseline: str = "Poly"
    ) -> pd.DataFrame:
        """Plot human draft advantage by round."""
        if draft_scored is None:
            draft_scored = self.draft_scored
        
        if draft_scored is None:
            raise ValueError("No scored draft data. Run score_picks() first.")
        
        manual = draft_scored[draft_scored["Is_Autodrafted"] == 0].copy()
        
        if baseline == "Poly":
            points_col = "Points_Added_Poly"
        elif baseline == "Pooled":
            points_col = "Points_Added_Pooled"
        else:
            points_col = "Points_Added_Year"
        
        summary = (
            manual.groupby("Round", dropna=False)
            .agg(
                Manual_Picks=(points_col, "size"),
                Avg_Points_Added=(points_col, "mean"),
            )
            .reset_index()
        )
        summary["Round"] = pd.to_numeric(summary["Round"], errors="coerce")
        summary = summary.sort_values("Round")
        
        x = np.arange(len(summary))
        w = 0.6
        
        plt.figure(figsize=(12, 6))
        plt.bar(x, summary["Avg_Points_Added"], width=w, label=f"{baseline} Baseline")
        plt.axhline(0, linestyle="--", color="gray", alpha=0.6)
        plt.xticks(x, summary["Round"])
        plt.xlabel("Draft Round")
        plt.ylabel("Avg Points Above Expected (per manual pick)")
        plt.title("Human Draft Advantage by Round")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return summary
    
    def plot_human_advantage_by_position(
        self,
        draft_scored: pd.DataFrame = None,
        baseline: str = "Poly"
    ) -> pd.DataFrame:
        """Plot human draft advantage by position."""
        if draft_scored is None:
            draft_scored = self.draft_scored
        
        if draft_scored is None:
            raise ValueError("No scored draft data. Run score_picks() first.")
        
        manual = draft_scored[draft_scored["Is_Autodrafted"] == 0].copy()
        
        if baseline == "Poly":
            points_col = "Points_Added_Poly"
        elif baseline == "Pooled":
            points_col = "Points_Added_Pooled"
        else:
            points_col = "Points_Added_Year"
        
        summary = (
            manual.groupby("Position", dropna=False)
            .agg(
                Manual_Picks=(points_col, "size"),
                Avg_Points_Added=(points_col, "mean"),
            )
            .reset_index()
            .sort_values("Avg_Points_Added", ascending=False)
        )
        
        x = np.arange(len(summary))
        w = 0.6
        
        plt.figure(figsize=(12, 6))
        plt.bar(x, summary["Avg_Points_Added"], width=w, label=f"{baseline} Baseline")
        plt.axhline(0, linestyle="--", color="gray", alpha=0.6)
        plt.xticks(x, summary["Position"])
        plt.xlabel("Position")
        plt.ylabel("Avg Points Above Expected (per manual pick)")
        plt.title("Human Draft Advantage by Position")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return summary
    
    def plot_per_season_expected_values(
        self,
        draft_with_valid: pd.DataFrame = None,
        zoom_first_25: bool = True
    ) -> None:
        """
        Plot expected values separately for each season.
        
        Args:
            draft_with_valid: Draft data (uses self.draft_with_valid if None)
            zoom_first_25: If True, also create zoomed plots of first 25 picks per season
        """
        if draft_with_valid is None:
            draft_with_valid = self.draft_with_valid
        
        if draft_with_valid is None:
            raise ValueError("No draft data. Run pipeline first.")
        
        expected_by_pick_year = self._expected_by_pick_by_year(draft_with_valid)
        dist_by_year = self._expected_distribution_by_year(draft_with_valid)
        
        # Plot per-season curves
        for year, g in expected_by_pick_year.groupby("Year", sort=True):
            plt.figure(figsize=(10, 5))
            plt.plot(g["Overall"], g["Expected_Valid_Points"], linewidth=2)
            plt.xlabel("Overall Pick")
            plt.ylabel("Expected Valid Season Points")
            plt.title(f"Autodraft Expected Valid Points by Pick (NOT Smoothed) — {int(year)}")
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
        
        # Plot per-season variance (first 25 picks)
        if zoom_first_25:
            for year, g in dist_by_year.groupby("Year", sort=True):
                g25 = g[g["Overall"] <= 25].copy()
                if g25.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.plot(g25["Overall"], g25["Mean"], linewidth=2, label="Expected (Mean)")
                ax.fill_between(g25["Overall"], g25["P25"], g25["P75"], alpha=0.30, label="25–75% range")
                ax.fill_between(g25["Overall"], g25["P10"], g25["P90"], alpha=0.18, label="10–90% range")
                ax.set_xlabel("Overall Pick", fontsize=12)
                ax.set_ylabel("Valid Season Points", fontsize=12)
                ax.set_title(f"Autodraft Expected Valid Points by Pick (Variance) — {int(year)}\nFirst 25 Picks", fontsize=14)
                
                for _, r in g25.iterrows():
                    y_annot = (r["P10"] if pd.notna(r["P10"]) else r["Mean"]) - 5
                    ax.text(
                        r["Overall"], y_annot,
                        f"{int(r['Unique_Players'])} uniq\nn={int(r['N_Picks'])}",
                        ha="center", va="top", fontsize=9, alpha=0.7
                    )
                
                ax.legend()
                ax.grid(alpha=0.2)
                fig.tight_layout()
                plt.show()
    
    def _expected_distribution_by_year(self, draft_with_valid: pd.DataFrame) -> pd.DataFrame:
        """Compute expected distribution by year."""
        df = draft_with_valid.copy()
        team_auto = (
            df.groupby(["League_ID", "Year", "Team"], dropna=False)["Is_Autodrafted"]
            .agg(Total_Picks="size", Auto_Picks="sum")
            .reset_index()
        )
        team_auto["Is_Fully_Autodrafted"] = team_auto["Total_Picks"].eq(team_auto["Auto_Picks"])
        df = df.merge(team_auto[["League_ID", "Year", "Team", "Is_Fully_Autodrafted"]],
                     on=["League_ID", "Year", "Team"], how="left")
        df = df[df["Is_Fully_Autodrafted"]].copy()
        
        dist = (
            df.groupby(["Year", "Overall"], dropna=False)
            .agg(
                Mean=("Season_Total_Points_Valid", "mean"),
                Std=("Season_Total_Points_Valid", "std"),
                N_Picks=("Season_Total_Points_Valid", "count"),
                Unique_Players=("Player_norm", "nunique"),
                P10=("Season_Total_Points_Valid", lambda s: s.quantile(0.10)),
                P25=("Season_Total_Points_Valid", lambda s: s.quantile(0.25)),
                P50=("Season_Total_Points_Valid", lambda s: s.quantile(0.50)),
                P75=("Season_Total_Points_Valid", lambda s: s.quantile(0.75)),
                P90=("Season_Total_Points_Valid", lambda s: s.quantile(0.90)),
            )
            .reset_index()
            .sort_values(["Year", "Overall"])
        )
        return dist
    
    # ==================== WAIVER WIRE BASELINE METHODS ====================
    
    def build_waiver_add_stints(
        self,
        transactions: pd.DataFrame,
        *,
        season_end_week: int = 17,
        include_actions: Optional[Set[str]] = None,
        drop_actions: Optional[Set[str]] = None,
        min_week_after_add: int = 0
    ) -> pd.DataFrame:
        """
        Build waiver add stints from transaction data.
        
        For each waiver/free-agent add, finds the next drop of that same player
        by that same team in the same league-year. Creates stints representing
        the period from add week to drop week (or season end if never dropped).
        
        Args:
            transactions: DataFrame with columns: League_ID, Year, Week, Team, Player_norm, Action_norm
            season_end_week: Last week of the season (default: 17)
            include_actions: Set of action types to consider as adds (default: {"WAIVER_ADD", "FREEAGENT_ADD"})
            drop_actions: Set of action types to consider as drops (default: {"DROP"})
            min_week_after_add: Minimum weeks after add before counting points (default: 0)
        
        Returns:
            DataFrame with columns: League_ID, Year, Team, Player_norm, Add_Week,
            Start_Week_For_Credit, End_Week, Add_Type
        """
        if include_actions is None:
            include_actions = {"WAIVER_ADD", "FREEAGENT_ADD"}
        if drop_actions is None:
            drop_actions = {"DROP"}
        
        df = transactions.copy()
        required = {"League_ID", "Year", "Week", "Team", "Player_norm", "Action_norm"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"transactions missing columns: {sorted(missing)}")
        
        adds = df[df["Action_norm"].isin(include_actions)].copy()
        drops = df[df["Action_norm"].isin(drop_actions)].copy()
        
        if adds.empty:
            raise ValueError("No waiver/free-agent add events found after filtering.")
        
        # For each add, find the next drop of that same player by that same team in same league-year
        adds = adds.sort_values(["League_ID", "Year", "Team", "Player_norm", "Week"]).reset_index(drop=True)
        adds["Add_ID"] = np.arange(len(adds), dtype=int)
        
        cand = adds.merge(
            drops[["League_ID", "Year", "Team", "Player_norm", "Week"]].rename(columns={"Week": "Drop_Week"}),
            on=["League_ID", "Year", "Team", "Player_norm"],
            how="left"
        )
        
        cand = cand[cand["Drop_Week"].isna() | (cand["Drop_Week"] >= cand["Week"])].copy()
        
        next_drop = (
            cand.groupby("Add_ID", dropna=False)["Drop_Week"]
            .min()
            .reset_index()
        )
        
        adds = adds.merge(next_drop, on="Add_ID", how="left")
        adds["Add_Week"] = pd.to_numeric(adds["Week"], errors="coerce").astype(int)
        adds["Drop_Week"] = pd.to_numeric(adds["Drop_Week"], errors="coerce")
        
        # End week: drop_week - 1 if drop exists, else season end
        adds["End_Week"] = np.where(
            adds["Drop_Week"].notna(),
            adds["Drop_Week"] - 1,
            int(season_end_week)
        ).astype(int)
        
        # Optional conservative rule: don't count the add week itself
        adds["Start_Week_For_Credit"] = adds["Add_Week"] + int(min_week_after_add)
        
        out = adds[["League_ID", "Year", "Team", "Player_norm", "Add_Week", "Start_Week_For_Credit", "End_Week", "Action_norm"]].copy()
        out = out.rename(columns={"Action_norm": "Add_Type"})
        
        # Sanity clamp
        out["End_Week"] = out["End_Week"].clip(lower=0, upper=season_end_week)
        out = out[out["End_Week"] >= out["Start_Week_For_Credit"]].copy()
        
        return out
    
    def compute_valid_waiver_points(
        self,
        waiver_stints: pd.DataFrame,
        optimal_selected: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute valid waiver points for each stint using optimal lineup selections.
        
        Valid points are only counted when the player was selected in the optimal lineup
        during their stint window. Aggregates to stint-level totals.
        
        Args:
            waiver_stints: DataFrame from build_waiver_add_stints with columns:
                League_ID, Year, Team, Player_norm, Start_Week_For_Credit, End_Week, Add_Week, Add_Type
            optimal_selected: DataFrame from compute_optimal_startable_points with columns:
                League_ID, Year, Team, Week, Player_norm, WeekPoints, SelectedOptimal, Position
        
        Returns:
            DataFrame with columns: League_ID, Year, Team, Player_norm, Add_Week,
            Start_Week_For_Credit, End_Week, Add_Type, Position, Valid_Points, Weeks_Observed
        """
        req_s = {"League_ID", "Year", "Team", "Player_norm", "Start_Week_For_Credit", "End_Week", "Add_Week", "Add_Type"}
        missing = req_s - set(waiver_stints.columns)
        if missing:
            raise ValueError(f"waiver_stints missing columns: {sorted(missing)}")
        
        req_o = {"League_ID", "Year", "Team", "Week", "Player_norm", "WeekPoints", "SelectedOptimal", "Position"}
        missing = req_o - set(optimal_selected.columns)
        if missing:
            raise ValueError(f"optimal_selected missing columns: {sorted(missing)}")
        
        osel = optimal_selected.copy()
        osel["Week"] = pd.to_numeric(osel["Week"], errors="coerce").astype(int)
        osel["WeekPoints"] = pd.to_numeric(osel["WeekPoints"], errors="coerce").fillna(0.0)
        osel["ValidWeekPoints"] = np.where(osel["SelectedOptimal"].astype(bool), osel["WeekPoints"], 0.0)
        
        # Join stints to weekly player rows (same league-year-team-player)
        joined = waiver_stints.merge(
            osel[["League_ID", "Year", "Team", "Week", "Player_norm", "Position", "ValidWeekPoints"]],
            on=["League_ID", "Year", "Team", "Player_norm"],
            how="left"
        )
        
        # Keep only weeks in stint window
        joined = joined[
            (joined["Week"].notna()) &
            (joined["Week"].astype(int) >= joined["Start_Week_For_Credit"].astype(int)) &
            (joined["Week"].astype(int) <= joined["End_Week"].astype(int))
        ].copy()
        
        # Aggregate to stint
        stint_points = (
            joined.groupby(["League_ID", "Year", "Team", "Player_norm", "Add_Week", "Start_Week_For_Credit", "End_Week", "Add_Type", "Position"], dropna=False)
            .agg(
                Valid_Points=("ValidWeekPoints", "sum"),
                Weeks_Observed=("Week", "nunique"),
            )
            .reset_index()
        )
        
        # Add explicit zeros for stints where player never appeared in optimal_selected
        out = waiver_stints.merge(
            stint_points,
            on=["League_ID", "Year", "Team", "Player_norm", "Add_Week", "Start_Week_For_Credit", "End_Week", "Add_Type"],
            how="left"
        )
        out["Valid_Points"] = pd.to_numeric(out["Valid_Points"], errors="coerce").fillna(0.0)
        out["Weeks_Observed"] = pd.to_numeric(out["Weeks_Observed"], errors="coerce").fillna(0).astype(int)
        out["Position"] = out["Position"].fillna("UNKNOWN")
        
        return out
    
    def load_transactions(self, path: Path, year: int) -> pd.DataFrame:
        """
        Load and normalize transaction data for a single year.
        
        Args:
            path: Path to transaction_data.csv
            year: Year of the data
        
        Returns:
            DataFrame with normalized transaction data
        """
        df = pd.read_csv(path)
        required = ["League_ID", "Week", "Team", "Player", "Action"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{path} transaction_data is missing columns: {missing}")
        
        df = df.copy()
        df["Year"] = int(year)
        df["League_ID"] = df["League_ID"].astype(int)
        df["Year"] = df["Year"].astype(int)
        df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype(int)
        df["Team"] = pd.to_numeric(df["Team"], errors="coerce").astype(int)
        
        df["Player_norm"] = df["Player"].map(lambda x: self.normalize_player_name(x, self.anchor_synonyms))
        df["Action_norm"] = df["Action"].astype(str).str.strip().str.upper()
        
        return df
    
    def load_multi_season_transactions(self, lineups_filt: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load transaction data across multiple seasons.
        
        Args:
            lineups_filt: Filtered lineup data to match league-years (optional)
        
        Returns:
            DataFrame with all transaction data
        """
        if self.verbose:
            print("\n=== Loading Multi-Season Transaction Data ===")
        
        parts = []
        for year_dir in sorted([
            p for p in self.raw_base.iterdir()
            if p.is_dir() and p.name.isdigit() and len(p.name) == 4
        ]):
            year = int(year_dir.name)
            if year not in self.years:
                continue
            
            tpath = year_dir / "transaction_data.csv"
            if tpath.exists():
                parts.append(self.load_transactions(tpath, year))
            else:
                if self.verbose:
                    print(f"Skipping {year}: missing {tpath}")
        
        if not parts:
            raise FileNotFoundError(f"No transaction_data.csv found under {self.raw_base}/<YEAR>/transaction_data.csv")
        
        transactions_all = pd.concat(parts, ignore_index=True)
        
        # Filter to same league-years as lineups if provided
        if lineups_filt is not None:
            kept_league_years = lineups_filt[["League_ID", "Year"]].drop_duplicates()
            transactions_all = transactions_all.merge(kept_league_years, on=["League_ID", "Year"], how="inner")
            if self.verbose:
                print(f"Filtered to {len(transactions_all):,} transaction rows across {len(kept_league_years):,} league-years")
        
        if self.verbose:
            print(f"Loaded {len(transactions_all):,} transaction records")
        
        return transactions_all
    
    def compute_waiver_baseline_candidates(
        self,
        waiver_with_valid: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Compute waiver baseline candidates using minimal competency framing.
        
        Calculates multiple baseline options (mean, median, Q25) for waiver wire performance.
        The Q25 baseline represents minimal competency, similar to autodraft baseline.
        
        Args:
            waiver_with_valid: DataFrame from compute_valid_waiver_points with columns:
                League_ID, Year, Team, Valid_Points, Weeks_Observed
        
        Returns:
            Tuple of (baseline_candidates dict, team_season DataFrame)
        """
        w = waiver_with_valid.copy()
        req = {"League_ID", "Year", "Team", "Valid_Points", "Weeks_Observed"}
        missing = req - set(w.columns)
        if missing:
            raise ValueError(f"waiver_with_valid missing columns: {sorted(missing)}")
        
        w["Year"] = pd.to_numeric(w["Year"], errors="coerce").astype(int)
        w["Valid_Points"] = pd.to_numeric(w["Valid_Points"], errors="coerce").fillna(0.0)
        w["Weeks_Observed"] = pd.to_numeric(w["Weeks_Observed"], errors="coerce").fillna(0).astype(int)
        w = w[w["Weeks_Observed"] > 0].copy()
        
        if w.empty:
            raise ValueError("No waiver stints with Weeks_Observed > 0.")
        
        w["Stint_Weekly_Rate"] = w["Valid_Points"] / w["Weeks_Observed"]
        
        team_season = (
            w.groupby(["League_ID", "Year", "Team"], dropna=False)
            .agg(
                TeamSeason_Valid=("Valid_Points", "sum"),
                TeamSeason_Weeks=("Weeks_Observed", "sum"),
            )
            .reset_index()
        )
        team_season = team_season[team_season["TeamSeason_Weeks"] > 0].copy()
        team_season["TeamSeason_Weekly_Rate"] = team_season["TeamSeason_Valid"] / team_season["TeamSeason_Weeks"]
        
        baseline_candidates = {
            "avg_stint_week": float(w["Valid_Points"].sum() / w["Weeks_Observed"].sum()),
            "median_team_season": float(team_season["TeamSeason_Weekly_Rate"].median()),
            "q25_team_season": float(team_season["TeamSeason_Weekly_Rate"].quantile(0.25)),
        }
        
        if self.verbose:
            print(f"Waiver baseline candidates: {baseline_candidates}")
            print(f"Zero-rate share among stints: {(w['Stint_Weekly_Rate'] == 0).mean():.1%}")
        
        return baseline_candidates, team_season
    
    def compute_startsit_metrics(
        self,
        lineups_filt: pd.DataFrame,
        slot_counts: Dict[str, int] = None,
        flex_eligible: Set[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute Start/Sit metrics using projected-optimal baseline.
        
        Compares actual starter points to projected-optimal lineup points.
        Only includes complete team-weeks (all expected starters present).
        
        Args:
            lineups_filt: Filtered lineup data with Projected_Points column
            slot_counts: Position slot requirements (default: standard 1QB/2RB/2WR/1TE/1FLEX/1K/1DST)
            flex_eligible: Positions eligible for FLEX (default: {"RB", "WR", "TE"})
        
        Returns:
            Tuple of (startsit_weekly DataFrame, startsit_weekly_clean DataFrame)
            startsit_weekly_clean only includes complete team-weeks
        """
        if "Projected_Points" not in lineups_filt.columns:
            raise ValueError("lineups_filt missing Projected_Points.")
        
        if slot_counts is None:
            slot_counts = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "D/ST": 1}
        if flex_eligible is None:
            flex_eligible = {"RB", "WR", "TE"}
        
        expected_starter_count = int(self.expected_starters.get("STARTERS", 9))
        starter_slots = {"QB", "RB", "WR", "TE", "FLEX", "K", "D/ST", "DST", "DEF", "RB/WR/TE", "OP", "SUPERFLEX", "QB/RB/WR/TE"}
        
        df = lineups_filt.copy()
        df["Week"] = pd.to_numeric(df["Week"], errors="coerce")
        df["Points"] = pd.to_numeric(df["Points"], errors="coerce").fillna(0.0)
        df["Projected_Points"] = pd.to_numeric(df["Projected_Points"], errors="coerce").fillna(0.0)
        df["Slot_norm"] = df["Slot"].astype(str).str.upper().str.strip()
        
        if "Is_Starter" in df.columns:
            df["Is_Starter"] = pd.to_numeric(df["Is_Starter"], errors="coerce").fillna(0).astype(int)
        else:
            df["Is_Starter"] = 0
        
        if "Player_norm" not in df.columns:
            df["Player_norm"] = df["Player"].astype(str).str.strip()
        
        # Robust starter flag: ESPN starter flag OR starter-typed slot
        df["Is_Starter_Robust"] = (df["Is_Starter"].eq(1) | df["Slot_norm"].isin(starter_slots)).astype(int)
        
        # Position inference
        pos = self._infer_player_position_by_core_starts(df[["League_ID", "Year", "Team", "Slot", "Player_norm"]].copy())
        
        # One row per player-week with actual + projected
        pw = (
            df.groupby(["League_ID", "Year", "Team", "Week", "Player_norm"], dropna=False)
            .agg(
                ActualWeekPoints=("Points", "max"),
                ProjWeekPoints=("Projected_Points", "max"),
            )
            .reset_index()
        )
        pw = pw.merge(pos, on=["League_ID", "Year", "Team", "Player_norm"], how="left")
        pw["Position"] = pw["Position"].fillna("UNKNOWN")
        
        # Projection-optimal lineup selection (by projected points)
        selected_parts = []
        for _, g in pw.groupby(["League_ID", "Year", "Team", "Week"], sort=False):
            work = g.copy().rename(columns={"ProjWeekPoints": "WeekPoints"})
            work_sel = self._choose_optimal_lineup_for_group(work, slot_counts, flex_eligible)
            selected_parts.append(work_sel)
        
        proj_selected = pd.concat(selected_parts, ignore_index=True)
        
        proj_opt_actual = (
            proj_selected[proj_selected["SelectedOptimal"]]
            .groupby(["League_ID", "Year", "Team", "Week"], dropna=False)["ActualWeekPoints"]
            .sum()
            .reset_index(name="ProjOptimal_ActualPoints")
        )
        proj_opt_proj = (
            proj_selected[proj_selected["SelectedOptimal"]]
            .groupby(["League_ID", "Year", "Team", "Week"], dropna=False)["WeekPoints"]
            .sum()
            .reset_index(name="ProjOptimal_ProjectedPoints")
        )
        proj_opt_n = (
            proj_selected[proj_selected["SelectedOptimal"]]
            .groupby(["League_ID", "Year", "Team", "Week"], dropna=False)
            .size().reset_index(name="ProjOptimal_Selected_Count")
        )
        
        # Actual starter totals using robust starter flag
        actual_wk = (
            df[df["Is_Starter_Robust"] == 1]
            .groupby(["League_ID", "Year", "Team", "Week"], dropna=False)
            .agg(
                Actual_Starter_Points=("Points", "sum"),
                Actual_Starter_ProjectedPoints=("Projected_Points", "sum"),
                Actual_Starter_Count=("Player_norm", "nunique"),
            )
            .reset_index()
        )
        
        startsit_weekly = (
            actual_wk
            .merge(proj_opt_actual, on=["League_ID", "Year", "Team", "Week"], how="outer")
            .merge(proj_opt_proj, on=["League_ID", "Year", "Team", "Week"], how="outer")
            .merge(proj_opt_n, on=["League_ID", "Year", "Team", "Week"], how="outer")
            .fillna(0.0)
        )
        startsit_weekly["StartSit_PA"] = startsit_weekly["Actual_Starter_Points"] - startsit_weekly["ProjOptimal_ActualPoints"]
        startsit_weekly["StartSit_ProjGap"] = startsit_weekly["Actual_Starter_ProjectedPoints"] - startsit_weekly["ProjOptimal_ProjectedPoints"]
        
        # Keep only complete team-weeks for Start/Sit scoring
        startsit_weekly["Lineup_Complete"] = startsit_weekly["Actual_Starter_Count"].eq(expected_starter_count)
        startsit_weekly["Proj_Complete"] = startsit_weekly["ProjOptimal_Selected_Count"].eq(expected_starter_count)
        startsit_weekly_clean = startsit_weekly[startsit_weekly["Lineup_Complete"] & startsit_weekly["Proj_Complete"]].copy()
        
        if self.verbose:
            print(f"[Start/Sit] Complete team-weeks kept: {len(startsit_weekly_clean):,} / {len(startsit_weekly):,}")
            print(startsit_weekly_clean[["StartSit_PA", "StartSit_ProjGap"]].describe())
        
        return startsit_weekly, startsit_weekly_clean
    
    # ==================== WAIVER WIRE VISUALIZATION METHODS ====================
    
    def plot_waiver_baseline_exploration(
        self,
        team_season: pd.DataFrame,
        baseline_candidates: Dict[str, float],
        waiver_baseline_name: str = "q25_team_season"
    ) -> None:
        """
        Plot waiver baseline exploration visuals.
        
        Shows distribution of team-season waiver rates with candidate baseline lines,
        and yearly trend of team-season waiver rates.
        
        Args:
            team_season: DataFrame from compute_waiver_baseline_candidates
            baseline_candidates: Dict of baseline name -> value
            waiver_baseline_name: Name of selected baseline (default: "q25_team_season")
        """
        waiver_baseline_value = baseline_candidates.get(waiver_baseline_name)
        if waiver_baseline_value is None:
            raise ValueError(f"waiver_baseline_name '{waiver_baseline_name}' not in baseline_candidates")
        
        # (1) Team-season distribution
        plt.figure(figsize=(10, 5))
        plt.hist(team_season["TeamSeason_Weekly_Rate"], bins=60, alpha=0.75, color="#7AA6C2")
        for name, val in baseline_candidates.items():
            ls = "-" if name == waiver_baseline_name else "--"
            lw = 2.5 if name == waiver_baseline_name else 1.5
            plt.axvline(val, linestyle=ls, linewidth=lw, label=f"{name}: {val:.2f}")
        plt.xlabel("Team-season waiver valid points per stint-week")
        plt.ylabel("Count")
        plt.title("Waiver Baseline Candidate Exploration: Team-Season Distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # (2) Yearly trend
        team_season["Year"] = pd.to_numeric(team_season["Year"], errors="coerce").astype(int)
        yearly = (
            team_season.groupby("Year", dropna=False)["TeamSeason_Weekly_Rate"]
            .agg(["mean", "median", lambda s: s.quantile(0.25)])
            .reset_index()
        )
        yearly.columns = ["Year", "Mean", "Median", "Q25"]
        
        plt.figure(figsize=(10, 5))
        plt.plot(yearly["Year"], yearly["Mean"], marker="o", label="Mean", linewidth=2)
        plt.plot(yearly["Year"], yearly["Median"], marker="s", label="Median", linewidth=2)
        plt.plot(yearly["Year"], yearly["Q25"], marker="^", label="Q25", linewidth=2)
        for name, val in baseline_candidates.items():
            ls = "-" if name == waiver_baseline_name else "--"
            plt.axhline(val, linestyle=ls, alpha=0.5, label=f"{name}: {val:.2f}")
        plt.xlabel("Year")
        plt.ylabel("Team-season waiver rate")
        plt.title("Yearly Trend: Team-Season Waiver Rates")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    def plot_draft_vs_waiver_points(
        self,
        draft_scored: pd.DataFrame,
        waiver_stints: pd.DataFrame,
        optimal_selected: pd.DataFrame,
        waiver_baseline_value: float,
        draft_points_col: str = "Points_Added_Poly",
        manual_draft_only: bool = True,
        season_end_week: int = 17,
        ignore_weeks: Set[int] = None,
        agg_mode: str = "mean_per_team_season"
    ) -> None:
        """
        Plot draft vs waiver points added over time.
        
        Draft points added assigned to Week 0, waiver points added by week.
        
        Args:
            draft_scored: Scored draft data
            waiver_stints: Waiver stints DataFrame
            optimal_selected: Optimal lineup selections
            waiver_baseline_value: Baseline value for waiver points
            draft_points_col: Column name for draft points (default: "Points_Added_Poly")
            manual_draft_only: If True, only include manual draft picks
            season_end_week: Last week of season (default: 17)
            ignore_weeks: Set of weeks to ignore (default: {15})
            agg_mode: Aggregation mode - "total" or "mean_per_team_season" (default)
        """
        if ignore_weeks is None:
            ignore_weeks = {15}
        
        d = draft_scored.copy()
        if manual_draft_only:
            d = d[d["Is_Autodrafted"] == 0].copy()
        d[draft_points_col] = pd.to_numeric(d[draft_points_col], errors="coerce").fillna(0.0)
        draft_total = float(d[draft_points_col].sum())
        team_seasons_n = d[["League_ID", "Year", "Team"]].drop_duplicates().shape[0]
        
        osel = optimal_selected.copy()
        osel["Week"] = pd.to_numeric(osel["Week"], errors="coerce")
        osel["WeekPoints"] = pd.to_numeric(osel["WeekPoints"], errors="coerce").fillna(0.0)
        osel["SelectedOptimal"] = osel["SelectedOptimal"].fillna(False).astype(bool)
        osel["ValidWeekPoints"] = np.where(osel["SelectedOptimal"], osel["WeekPoints"], 0.0)
        
        joined = waiver_stints.merge(
            osel[["League_ID", "Year", "Team", "Week", "Player_norm", "ValidWeekPoints"]],
            on=["League_ID", "Year", "Team", "Player_norm"],
            how="left"
        )
        joined = joined[
            joined["Week"].notna() &
            (joined["Week"].astype(int) >= joined["Start_Week_For_Credit"].astype(int)) &
            (joined["Week"].astype(int) <= joined["End_Week"].astype(int))
        ].copy()
        joined["Week"] = pd.to_numeric(joined["Week"], errors="coerce").astype(int)
        joined = joined[(joined["Week"] >= 1) & (joined["Week"] <= season_end_week)].copy()
        joined = joined[~joined["Week"].isin(ignore_weeks)].copy()
        
        waiver_valid_week = joined.groupby("Week", dropna=False)["ValidWeekPoints"].sum().reset_index(name="Waiver_ValidSum")
        
        exp_rows = []
        for _, r in waiver_stints.iterrows():
            s = int(r["Start_Week_For_Credit"])
            e = int(r["End_Week"])
            if e < s:
                continue
            s = max(1, s)
            e = min(season_end_week, e)
            if e < s:
                continue
            exp_rows.extend([wk for wk in range(s, e + 1) if wk not in ignore_weeks])
        
        active_stints_week = pd.Series(exp_rows, name="Week").value_counts().rename_axis("Week").reset_index(name="Active_StintWeeks")
        waiver_week = active_stints_week.merge(waiver_valid_week, on="Week", how="left")
        waiver_week["Waiver_ValidSum"] = waiver_week["Waiver_ValidSum"].fillna(0.0)
        waiver_week["Waiver_PointsAdded"] = waiver_week["Waiver_ValidSum"] - waiver_baseline_value * waiver_week["Active_StintWeeks"]
        
        weeks = [0] + [wk for wk in range(1, season_end_week + 1) if wk not in ignore_weeks]
        draft_weekly = pd.Series(0.0, index=weeks)
        waiver_weekly = pd.Series(0.0, index=weeks)
        
        draft_weekly.loc[0] = draft_total
        for _, r in waiver_week.iterrows():
            wk = int(r["Week"])
            if wk in waiver_weekly.index:
                waiver_weekly.loc[wk] = float(r["Waiver_PointsAdded"])
        
        if agg_mode == "mean_per_team_season":
            draft_weekly /= team_seasons_n
            waiver_weekly /= team_seasons_n
        
        cum_draft = draft_weekly.cumsum()
        cum_waiver = waiver_weekly.cumsum()
        cum_total = cum_draft + cum_waiver
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_draft.index, cum_draft.values, label="Draft (cumulative)", linewidth=2, marker="o")
        plt.plot(cum_waiver.index, cum_waiver.values, label="Waiver (cumulative)", linewidth=2, marker="s")
        plt.plot(cum_total.index, cum_total.values, label="Total (cumulative)", linewidth=2, linestyle="--", marker="^")
        plt.axhline(0, linestyle=":", color="gray", alpha=0.5)
        plt.xlabel("Week")
        plt.ylabel(f"Points Added Over Expected ({'per team-season' if agg_mode == 'mean_per_team_season' else 'total'})")
        plt.title("Cumulative Points Added: Draft vs Waiver")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()
    
    def plot_yearly_draft_waiver_totals(
        self,
        draft_scored: pd.DataFrame,
        waiver_with_valid: pd.DataFrame,
        baseline_candidates: Dict[str, float],
        manual_draft_only: bool = True
    ) -> pd.DataFrame:
        """
        Plot yearly total points added over expected for Draft + Waiver.
        
        Args:
            draft_scored: Scored draft data
            waiver_with_valid: Waiver data with valid points
            baseline_candidates: Dict of baseline candidates
            manual_draft_only: If True, only include manual draft picks
        
        Returns:
            Summary DataFrame with yearly totals
        """
        d = draft_scored.copy()
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype(int)
        for c in ["Points_Added_Poly", "Points_Added_Pooled"]:
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        if manual_draft_only:
            d = d[d["Is_Autodrafted"] == 0].copy()
        
        d_year = (
            d.groupby("Year", dropna=False)
            .agg(
                Draft_PA_Poly=("Points_Added_Poly", "sum"),
                Draft_PA_Pooled=("Points_Added_Pooled", "sum"),
            )
            .reset_index()
        )
        
        w = waiver_with_valid.copy()
        w["Year"] = pd.to_numeric(w["Year"], errors="coerce").astype(int)
        w["Valid_Points"] = pd.to_numeric(w["Valid_Points"], errors="coerce").fillna(0.0)
        w["Weeks_Observed"] = pd.to_numeric(w["Weeks_Observed"], errors="coerce").fillna(0.0)
        w = w[w["Weeks_Observed"] > 0].copy()
        
        waiver_avg = float(baseline_candidates.get("avg_stint_week", 0))
        waiver_q25 = float(baseline_candidates.get("q25_team_season", 0))
        
        w["Waiver_PA_Avg"] = w["Valid_Points"] - waiver_avg * w["Weeks_Observed"]
        w["Waiver_PA_Q25"] = w["Valid_Points"] - waiver_q25 * w["Weeks_Observed"]
        
        w_year = (
            w.groupby("Year", dropna=False)
            .agg(
                Waiver_PA_Avg=("Waiver_PA_Avg", "sum"),
                Waiver_PA_Q25=("Waiver_PA_Q25", "sum"),
            )
            .reset_index()
        )
        
        yearly_all = d_year.merge(w_year, on="Year", how="outer").fillna(0.0).sort_values("Year")
        
        x = np.arange(len(yearly_all))
        bw = 0.2
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - 1.5 * bw, yearly_all["Draft_PA_Poly"], width=bw, label="Draft PA (Poly)")
        plt.bar(x - 0.5 * bw, yearly_all["Draft_PA_Pooled"], width=bw, label="Draft PA (Pooled)")
        plt.bar(x + 0.5 * bw, yearly_all["Waiver_PA_Q25"], width=bw, label="Waiver PA (Q25 baseline)")
        plt.bar(x + 1.5 * bw, yearly_all["Waiver_PA_Avg"], width=bw, label="Waiver PA (Avg baseline)")
        
        plt.axhline(0, linestyle="--", color="gray", alpha=0.6)
        plt.xticks(x, yearly_all["Year"])
        plt.ylabel("Total Points Added Over Expected")
        plt.title("Yearly Total Points Added: Draft vs Waiver by Baseline")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return yearly_all
    
    # ==================== START/SIT VISUALIZATION METHODS ====================
    
    def plot_startsit_by_year(
        self,
        startsit_weekly_clean: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Plot average Start/Sit points added by year.
        
        Args:
            startsit_weekly_clean: Clean start/sit weekly data
        
        Returns:
            Summary DataFrame
        """
        s = startsit_weekly_clean.copy()
        if "StartSit_PA" not in s.columns:
            raise ValueError("startsit_weekly_clean missing StartSit_PA.")
        
        s["Year"] = pd.to_numeric(s["Year"], errors="coerce").astype(int)
        
        summary_ss_year = (
            s.groupby("Year", dropna=False)
            .agg(
                TeamWeeks=("StartSit_PA", "size"),
                Avg_StartSit_PA=("StartSit_PA", "mean"),
                Total_StartSit_PA=("StartSit_PA", "sum"),
            )
            .reset_index()
            .sort_values("Year")
        )
        
        x = np.arange(len(summary_ss_year))
        plt.figure(figsize=(10, 5))
        plt.bar(x, summary_ss_year["Avg_StartSit_PA"], color="#54A24B", alpha=0.8)
        plt.axhline(0, linestyle="--", color="gray", alpha=0.6)
        plt.xticks(x, summary_ss_year["Year"])
        plt.ylabel("Average Start/Sit Points Added vs Projected-Optimal (per complete team-week)")
        plt.title("Start/Sit Decision Quality by Year (Projected Baseline, Complete Weeks)")
        for i, r in summary_ss_year.iterrows():
            plt.text(i, r["Avg_StartSit_PA"], f"n={int(r['TeamWeeks'])}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.show()
        
        return summary_ss_year
    
    def plot_cumulative_draft_waiver_startsit(
        self,
        draft_scored: pd.DataFrame,
        waiver_stints: pd.DataFrame,
        optimal_selected: pd.DataFrame,
        startsit_weekly_clean: pd.DataFrame,
        waiver_baseline_value: float,
        draft_points_col: str = "Points_Added_Poly",
        manual_draft_only: bool = True,
        season_end_week: int = 17,
        ignore_weeks: Set[int] = None,
        agg_mode: str = "mean_per_team_season"
    ) -> None:
        """
        Plot cumulative points added: Draft + Waiver + Start/Sit.
        
        Args:
            draft_scored: Scored draft data
            waiver_stints: Waiver stints DataFrame
            optimal_selected: Optimal lineup selections
            startsit_weekly_clean: Clean start/sit weekly data
            waiver_baseline_value: Baseline value for waiver points
            draft_points_col: Column name for draft points
            manual_draft_only: If True, only include manual draft picks
            season_end_week: Last week of season
            ignore_weeks: Set of weeks to ignore
            agg_mode: Aggregation mode
        """
        if ignore_weeks is None:
            ignore_weeks = {15}
        
        d = draft_scored.copy()
        if manual_draft_only:
            d = d[d["Is_Autodrafted"] == 0].copy()
        d[draft_points_col] = pd.to_numeric(d[draft_points_col], errors="coerce").fillna(0.0)
        draft_total = float(d[draft_points_col].sum())
        team_seasons_n = d[["League_ID", "Year", "Team"]].drop_duplicates().shape[0]
        
        osel = optimal_selected.copy()
        osel["Week"] = pd.to_numeric(osel["Week"], errors="coerce")
        osel["WeekPoints"] = pd.to_numeric(osel["WeekPoints"], errors="coerce").fillna(0.0)
        osel["SelectedOptimal"] = osel["SelectedOptimal"].fillna(False).astype(bool)
        osel["ValidWeekPoints"] = np.where(osel["SelectedOptimal"], osel["WeekPoints"], 0.0)
        
        joined = waiver_stints.merge(
            osel[["League_ID", "Year", "Team", "Week", "Player_norm", "ValidWeekPoints"]],
            on=["League_ID", "Year", "Team", "Player_norm"],
            how="left"
        )
        joined = joined[
            joined["Week"].notna() &
            (joined["Week"].astype(int) >= joined["Start_Week_For_Credit"].astype(int)) &
            (joined["Week"].astype(int) <= joined["End_Week"].astype(int))
        ].copy()
        joined["Week"] = pd.to_numeric(joined["Week"], errors="coerce").astype(int)
        joined = joined[(joined["Week"] >= 1) & (joined["Week"] <= season_end_week)].copy()
        joined = joined[~joined["Week"].isin(ignore_weeks)].copy()
        
        waiver_valid_week = joined.groupby("Week", dropna=False)["ValidWeekPoints"].sum().reset_index(name="Waiver_ValidSum")
        
        exp_rows = []
        for _, r in waiver_stints.iterrows():
            s = int(r["Start_Week_For_Credit"])
            e = int(r["End_Week"])
            if e < s:
                continue
            s = max(1, s)
            e = min(season_end_week, e)
            if e < s:
                continue
            exp_rows.extend([wk for wk in range(s, e + 1) if wk not in ignore_weeks])
        
        active_stints_week = pd.Series(exp_rows, name="Week").value_counts().rename_axis("Week").reset_index(name="Active_StintWeeks")
        waiver_week = active_stints_week.merge(waiver_valid_week, on="Week", how="left")
        waiver_week["Waiver_ValidSum"] = waiver_week["Waiver_ValidSum"].fillna(0.0)
        waiver_week["Waiver_PA"] = waiver_week["Waiver_ValidSum"] - waiver_baseline_value * waiver_week["Active_StintWeeks"]
        
        ss = startsit_weekly_clean.copy()
        ss["Week"] = pd.to_numeric(ss["Week"], errors="coerce").astype(int)
        ss = ss[(ss["Week"] >= 1) & (ss["Week"] <= season_end_week)].copy()
        ss = ss[~ss["Week"].isin(ignore_weeks)].copy()
        startsit_week = ss.groupby("Week", dropna=False)["StartSit_PA"].sum().reset_index()
        
        weeks = [0] + [wk for wk in range(1, season_end_week + 1) if wk not in ignore_weeks]
        draft_weekly = pd.Series(0.0, index=weeks)
        waiver_weekly = pd.Series(0.0, index=weeks)
        startsit_weekly_series = pd.Series(0.0, index=weeks)
        
        draft_weekly.loc[0] = draft_total
        for _, r in waiver_week.iterrows():
            waiver_weekly.loc[int(r["Week"])] = float(r["Waiver_PA"])
        for _, r in startsit_week.iterrows():
            startsit_weekly_series.loc[int(r["Week"])] = float(r["StartSit_PA"])
        
        if agg_mode == "mean_per_team_season":
            draft_weekly /= team_seasons_n
            waiver_weekly /= team_seasons_n
            startsit_weekly_series /= team_seasons_n
        
        cum_draft = draft_weekly.cumsum()
        cum_waiver = waiver_weekly.cumsum()
        cum_startsit = startsit_weekly_series.cumsum()
        cum_total = cum_draft + cum_waiver + cum_startsit
        
        plt.figure(figsize=(12, 6))
        plt.plot(cum_draft.index, cum_draft.values, label="Draft (cumulative)", linewidth=2, marker="o")
        plt.plot(cum_waiver.index, cum_waiver.values, label="Waiver (cumulative)", linewidth=2, marker="s")
        plt.plot(cum_startsit.index, cum_startsit.values, label="Start/Sit (cumulative)", linewidth=2, marker="^")
        plt.plot(cum_total.index, cum_total.values, label="Total (cumulative)", linewidth=2, linestyle="--", marker="D")
        plt.axhline(0, linestyle=":", color="gray", alpha=0.5)
        plt.xlabel("Week")
        plt.ylabel(f"Points Added Over Expected ({'per team-season' if agg_mode == 'mean_per_team_season' else 'total'})")
        plt.title("Cumulative Points Added: Draft + Waiver + Start/Sit")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

