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
        
    # ==================== UTILITY METHODS ====================
    
    @staticmethod
    def normalize_player_name(name: str, synonyms: Dict[str, str] = None) -> str:
        """Normalize player name for consistent matching."""
        if pd.isna(name):
            return name
        s = str(name).strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" Jr.", "").replace(" Sr.", "")
        if synonyms:
            s = synonyms.get(s, s)
        return s
    
    @staticmethod
    def normalize_slot(slot: str) -> str:
        """Normalize lineup slot designation."""
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
        """Infer starter configuration for each league-year."""
        df = lineups.copy()
        required = {"League_ID", "Year", "Team", "Week", "Slot", "Is_Starter"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"lineups missing columns for starter signature: {sorted(missing)}")
        
        df["Slot_norm"] = df["Slot"].map(self.normalize_slot)
        starters = df[df["Is_Starter"].fillna(0).astype(int) == 1].copy()
        
        tw = (
            starters.groupby(["League_ID", "Year", "Team", "Week", "Slot_norm"])
            .size()
            .reset_index(name="n")
        )
        
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
        
        for col in ["QB", "RB", "WR", "TE", "K", "D/ST", "FLEX", "OP"]:
            if col not in pivot.columns:
                pivot[col] = 0
        
        pivot["Starters_Total"] = pivot[["QB", "RB", "WR", "TE", "K", "D/ST", "FLEX", "OP"]].sum(axis=1)
        
        def mode_int(s: pd.Series) -> int:
            vc = s.value_counts()
            return int(vc.index[0]) if len(vc) else 0
        
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
        
        def robust_z(s: pd.Series) -> pd.Series:
            x = s.astype(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            if mad == 0 or np.isnan(mad):
                return (x - med) * 0.0
            return 0.6745 * (x - med) / mad
        
        a["rz"] = a.groupby(["Year", "Player_norm"], dropna=False)["Season_Total_Points"].transform(robust_z)
        a["Is_Outlier"] = a["rz"].abs() >= z_thresh
        
        meta = (
            a.groupby(["League_ID", "Year"], dropna=False)
            .agg(
                Anchors_Hit=("Player_norm", "nunique"),
                AnchorRows=("Player_norm", "size"),
                Anchors_Outlier=("Is_Outlier", "sum"),
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
        d["Player_norm"] = d["Player_norm"].map(
            lambda x: self.normalize_player_name(x, self.anchor_synonyms)
        )
        
        pos = self._infer_position_from_slots(lineups)
        pts = self._season_points_from_lineups(lineups)
        
        out = d.merge(pos, on=["League_ID", "Year", "Player_norm"], how="left")
        out = out.merge(pts, on=["League_ID", "Year", "Player_norm"], how="left")
        
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
        """Infer player position from lineup slots."""
        x = lineups.copy()
        x["Slot_norm"] = x["Slot"].map(self.normalize_slot)
        x["Is_core_pos"] = x["Slot_norm"].isin({"QB", "RB", "WR", "TE", "K", "D/ST"})
        
        counts = (
            x.groupby(["League_ID", "Year", "Player_norm", "Slot_norm", "Is_core_pos"], dropna=False)
            .size().reset_index(name="n")
            .sort_values(by=["League_ID", "Year", "Player_norm", "Is_core_pos", "n"],
                         ascending=[True, True, True, False, False])
        )
        
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
        
        t0 = time.time()
        selected_parts = []
        for i, (_, g) in enumerate(gb, start=1):
            if i % status_every == 0 or i == 1 or i == n_groups:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else float("inf")
                if self.verbose:
                    print(f"  - progress: {i:,}/{n_groups:,} groups ({rate:.1f} groups/s)")
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
        """Choose optimal lineup for a single team-week."""
        g = g.copy()
        g["SelectedOptimal"] = False
        used = set()
        
        def select_top(pos, n):
            nonlocal used
            if n <= 0:
                return []
            cand = g[(g["Position"] == pos) & (~g["Player_norm"].isin(used))].sort_values("WeekPoints", ascending=False)
            chosen = cand.head(n)["Player_norm"].tolist()
            used.update(chosen)
            return chosen
        
        def select_flex(n):
            nonlocal used
            if n <= 0:
                return []
            cand = g[(g["Position"].isin(flex_eligible)) & (~g["Player_norm"].isin(used))].sort_values("WeekPoints", ascending=False)
            chosen = cand.head(n)["Player_norm"].tolist()
            used.update(chosen)
            return chosen
        
        selected = []
        for pos in ["QB", "RB", "WR", "TE", "K", "D/ST"]:
            selected += select_top(pos, slot_counts.get(pos, 0))
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
        """Compute pooled denoised expected values."""
        df = draft_with_valid.copy()
        df["Overall"] = pd.to_numeric(df["Overall"], errors="coerce").astype(int)
        df["Is_Autodrafted"] = pd.to_numeric(df["Is_Autodrafted"], errors="coerce").fillna(0).astype(int)
        df["Season_Total_Points_Valid"] = pd.to_numeric(df["Season_Total_Points_Valid"], errors="coerce")
        
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
        
        if estimator == "mean":
            agg_func = "mean"
        elif estimator == "median":
            agg_func = lambda s: float(np.nanmedian(s.to_numpy()))
        elif estimator == "trimmed_mean":
            agg_func = lambda s: float(trim_mean(s.dropna().to_numpy(), proportiontocut=trim))
        elif estimator == "winsor_mean":
            agg_func = lambda s: float(np.mean(winsorize(s.dropna().to_numpy(), limits=trim)))
        else:
            raise ValueError("estimator must be one of: mean, median, trimmed_mean, winsor_mean")
        
        pooled = (
            df_auto.groupby("Overall", dropna=False)["Season_Total_Points_Valid"]
            .agg(Expected_Valid_Points=agg_func, Std_Valid_Points="std", N="count")
            .reset_index()
            .sort_values("Overall")
        )
        
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
        """Fit polynomial baseline to expected curve."""
        df = expected_pooled.copy()
        
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        
        if w_col in df.columns:
            w = pd.to_numeric(df[w_col], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        else:
            w = np.ones_like(x)
        
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x, y, w = x[m], y[m], w[m]
        
        if len(x) < degree + 2:
            raise ValueError(f"Not enough points to fit degree={degree} polynomial.")
        
        coeff = np.polyfit(x, y, deg=degree, w=w)
        p = np.poly1d(coeff)
        
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

