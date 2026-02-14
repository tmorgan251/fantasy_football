"""
Notebook-friendly orchestration layer for the fantasy football analysis pipeline.

Goal:
- Make each major notebook block callable with one line.
- Keep heavy logic in DraftValueAnalyzer.
- Maintain shared state between calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from .draft_value_analyzer import DraftValueAnalyzer


@dataclass
class WorkflowState:
    """Container for intermediate dataframes/results produced by block methods."""

    draft_raw: Optional[pd.DataFrame] = None
    lineups_raw: Optional[pd.DataFrame] = None
    draft_filt: Optional[pd.DataFrame] = None
    lineups_filt: Optional[pd.DataFrame] = None
    league_sig: Optional[pd.DataFrame] = None
    scoring_meta: Optional[pd.DataFrame] = None
    draft_enriched: Optional[pd.DataFrame] = None
    optimal_selected: Optional[pd.DataFrame] = None
    draft_with_valid: Optional[pd.DataFrame] = None
    expected_by_pick_year: Optional[pd.DataFrame] = None
    expected_by_pick_pooled: Optional[pd.DataFrame] = None
    expected_by_pick_poly: Optional[pd.DataFrame] = None
    draft_scored: Optional[pd.DataFrame] = None
    transactions_filt: Optional[pd.DataFrame] = None
    waiver_stints: Optional[pd.DataFrame] = None
    waiver_with_valid: Optional[pd.DataFrame] = None
    baseline_candidates: Optional[Dict[str, float]] = None
    team_season_waiver: Optional[pd.DataFrame] = None
    startsit_weekly: Optional[pd.DataFrame] = None
    startsit_weekly_clean: Optional[pd.DataFrame] = None
    startsit_team_season: Optional[pd.DataFrame] = None
    trade_summary_by_year: Optional[pd.DataFrame] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class NotebookWorkflow:
    """
    One-line-call orchestration object for notebook usage.

    Example:
        wf = NotebookWorkflow(verbose=True)
        wf.block_draft_pipeline()
        wf.block_score_draft()
        wf.block_load_transactions()
        wf.block_waiver_pipeline()
        wf.block_startsit()
        wf.plot_cumulative_draft_waiver_startsit()
    """

    def __init__(self, **analyzer_kwargs: Any) -> None:
        self.analyzer = DraftValueAnalyzer(**analyzer_kwargs)
        self.state = WorkflowState()

    # ==================== DRAFT BLOCKS ====================

    def block_clean_raw(self) -> None:
        """Notebook block: drop duplicates in raw CSV files."""
        self.analyzer.clean_raw_data()

    def block_load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Notebook block: load multi-season draft + lineup data."""
        draft_raw, lineups_raw = self.analyzer.load_multi_season_data()
        self.state.draft_raw = draft_raw
        self.state.lineups_raw = lineups_raw
        return draft_raw, lineups_raw

    def block_filter_leagues(
        self,
        *,
        filter_scoring: bool = True,
        filter_draft_length: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Notebook block: apply standard league filters.

        Includes:
        - standard starter signature filter
        - scoring outlier filter (optional)
        - draft length filter (optional)
        """
        if self.state.draft_raw is None or self.state.lineups_raw is None:
            self.block_load_data()

        # Add draft length metadata first
        draft_w_len = self.analyzer._add_draft_length(self.state.draft_raw)  # noqa: SLF001

        draft_filt, lineups_filt, league_sig = self.analyzer.filter_standard_leagues(
            draft_w_len, self.state.lineups_raw
        )
        self.state.league_sig = league_sig

        if filter_scoring:
            draft_filt, lineups_filt, scoring_meta = self.analyzer.filter_scoring_rule_outliers(
                draft_filt, lineups_filt
            )
            self.state.scoring_meta = scoring_meta

        if filter_draft_length:
            draft_filt, lineups_filt = self.analyzer.filter_draft_length(draft_filt, lineups_filt)

        self.state.draft_filt = draft_filt
        self.state.lineups_filt = lineups_filt

        # Keep analyzer attributes synchronized
        self.analyzer.lineups_filt = lineups_filt

        return draft_filt, lineups_filt

    def block_enrich_draft(self, *, save_csv: bool = True) -> pd.DataFrame:
        """Notebook block: enrich filtered draft data with position + season totals."""
        if self.state.draft_filt is None or self.state.lineups_filt is None:
            self.block_filter_leagues()

        draft_enriched = self.analyzer.enrich_draft_data(self.state.draft_filt, self.state.lineups_filt)
        self.state.draft_enriched = draft_enriched

        if save_csv:
            path = self.analyzer.out_dir / "draft_enriched_filtered.csv"
            draft_enriched.to_csv(path, index=False)

        return draft_enriched

    def block_compute_valid_points(self, *, save_csv: bool = True) -> pd.DataFrame:
        """Notebook block: compute optimal lineup selections and add valid points to draft rows."""
        if self.state.draft_enriched is None:
            self.block_enrich_draft(save_csv=save_csv)

        optimal_selected = self.analyzer.compute_optimal_startable_points(self.state.lineups_filt)
        draft_with_valid = self.analyzer.add_valid_points(self.state.draft_enriched, optimal_selected)

        self.state.optimal_selected = optimal_selected
        self.state.draft_with_valid = draft_with_valid

        # Keep analyzer attributes synchronized
        self.analyzer.optimal_selected = optimal_selected
        self.analyzer.draft_with_valid = draft_with_valid

        if save_csv:
            path = self.analyzer.out_dir / "draft_with_valid_points_filtered.csv"
            draft_with_valid.to_csv(path, index=False)

        return draft_with_valid

    def block_score_draft(self, *, save_csv: bool = True) -> pd.DataFrame:
        """Notebook block: compute expected curves and score draft picks against baselines."""
        if self.state.draft_with_valid is None:
            self.block_compute_valid_points(save_csv=save_csv)

        ey, ep, epp = self.analyzer.compute_expected_values(self.state.draft_with_valid)
        draft_scored = self.analyzer.score_picks(self.state.draft_with_valid, ey, ep, epp)

        self.state.expected_by_pick_year = ey
        self.state.expected_by_pick_pooled = ep
        self.state.expected_by_pick_poly = epp
        self.state.draft_scored = draft_scored

        if save_csv:
            path = self.analyzer.out_dir / "draft_scored_all_baselines.csv"
            draft_scored.to_csv(path, index=False)

        return draft_scored

    def block_draft_pipeline(self) -> pd.DataFrame:
        """
        One-line draft pipeline.

        Equivalent to running the major draft blocks in sequence.
        """
        self.block_load_data()
        self.block_filter_leagues()
        self.block_enrich_draft()
        self.block_compute_valid_points()
        return self.block_score_draft()

    # ==================== WAIVER BLOCKS ====================

    def block_load_transactions(self) -> pd.DataFrame:
        """Notebook block: load transactions and align to filtered league-years."""
        if self.state.lineups_filt is None:
            self.block_filter_leagues()
        tx = self.analyzer.load_multi_season_transactions(self.state.lineups_filt)
        self.state.transactions_filt = tx
        return tx

    def block_waiver_stints(
        self,
        *,
        season_end_week: int = 17,
        min_week_after_add: int = 1,
        include_actions: Optional[Set[str]] = None,
        drop_actions: Optional[Set[str]] = None,
    ) -> pd.DataFrame:
        """Notebook block: build waiver add stints."""
        if self.state.transactions_filt is None:
            self.block_load_transactions()
        stints = self.analyzer.build_waiver_add_stints(
            self.state.transactions_filt,
            season_end_week=season_end_week,
            include_actions=include_actions,
            drop_actions=drop_actions,
            min_week_after_add=min_week_after_add,
        )
        self.state.waiver_stints = stints
        return stints

    def block_waiver_valid_points(self) -> pd.DataFrame:
        """Notebook block: compute valid waiver points by stint."""
        if self.state.waiver_stints is None:
            self.block_waiver_stints()
        if self.state.optimal_selected is None:
            self.block_compute_valid_points()
        waiver_with_valid = self.analyzer.compute_valid_waiver_points(
            self.state.waiver_stints, self.state.optimal_selected
        )
        self.state.waiver_with_valid = waiver_with_valid
        return waiver_with_valid

    def block_waiver_baselines(self) -> Dict[str, float]:
        """Notebook block: compute waiver baseline candidates."""
        if self.state.waiver_with_valid is None:
            self.block_waiver_valid_points()
        candidates, team_season = self.analyzer.compute_waiver_baseline_candidates(self.state.waiver_with_valid)
        self.state.baseline_candidates = candidates
        self.state.team_season_waiver = team_season
        return candidates

    def block_waiver_pipeline(self) -> Dict[str, float]:
        """One-line waiver pipeline."""
        self.block_load_transactions()
        self.block_waiver_stints()
        self.block_waiver_valid_points()
        return self.block_waiver_baselines()

    # ==================== START/SIT BLOCKS ====================

    def block_startsit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Notebook block: compute projected-baseline start/sit metrics."""
        if self.state.lineups_filt is None:
            self.block_filter_leagues()
        weekly, weekly_clean = self.analyzer.compute_startsit_metrics(self.state.lineups_filt)
        self.state.startsit_weekly = weekly
        self.state.startsit_weekly_clean = weekly_clean

        team_season = (
            weekly_clean.groupby(["League_ID", "Year", "Team"], dropna=False)["StartSit_PA"]
            .sum()
            .reset_index(name="StartSit_PA_Season")
        )
        self.state.startsit_team_season = team_season

        return weekly, weekly_clean

    # ==================== TRADE BLOCK ====================

    def block_trade_stats(self) -> pd.DataFrame:
        """
        Notebook block: basic trade action summary by year.

        Uses filtered transactions when available.
        """
        if self.state.transactions_filt is None:
            self.block_load_transactions()

        tx = self.state.transactions_filt.copy()
        if "Action_norm" not in tx.columns and "Action" in tx.columns:
            tx["Action_norm"] = tx["Action"].astype(str).str.upper().str.strip()

        trade_rows = tx[tx["Action_norm"].str.contains("TRADE", na=False)].copy()
        out = (
            trade_rows.groupby("Year", dropna=False)
            .size()
            .reset_index(name="Trade_Action_Rows")
            .sort_values("Year")
        )
        self.state.trade_summary_by_year = out
        return out

    # ==================== PLOT WRAPPERS (ONE-LINE) ====================

    def plot_waiver_baseline_exploration(self, waiver_baseline_name: str = "q25_team_season") -> None:
        if self.state.team_season_waiver is None or self.state.baseline_candidates is None:
            self.block_waiver_baselines()
        self.analyzer.plot_waiver_baseline_exploration(
            self.state.team_season_waiver,
            self.state.baseline_candidates,
            waiver_baseline_name=waiver_baseline_name,
        )

    def plot_cumulative_draft_waiver(
        self,
        *,
        waiver_baseline_name: str = "q25_team_season",
        draft_points_col: str = "Points_Added_Poly",
        manual_draft_only: bool = True,
        season_end_week: int = 17,
        ignore_weeks: Optional[Set[int]] = None,
        agg_mode: str = "mean_per_team_season",
    ) -> None:
        if self.state.draft_scored is None:
            self.block_score_draft()
        if self.state.waiver_stints is None or self.state.waiver_with_valid is None:
            self.block_waiver_pipeline()
        baseline_value = self.state.baseline_candidates[waiver_baseline_name]
        self.analyzer.plot_draft_vs_waiver_points(
            self.state.draft_scored,
            self.state.waiver_stints,
            self.state.optimal_selected,
            baseline_value,
            draft_points_col=draft_points_col,
            manual_draft_only=manual_draft_only,
            season_end_week=season_end_week,
            ignore_weeks=ignore_weeks,
            agg_mode=agg_mode,
        )

    def plot_yearly_draft_waiver_totals(self, *, manual_draft_only: bool = True) -> pd.DataFrame:
        if self.state.draft_scored is None:
            self.block_score_draft()
        if self.state.waiver_with_valid is None or self.state.baseline_candidates is None:
            self.block_waiver_pipeline()
        return self.analyzer.plot_yearly_draft_waiver_totals(
            self.state.draft_scored,
            self.state.waiver_with_valid,
            self.state.baseline_candidates,
            manual_draft_only=manual_draft_only,
        )

    def plot_startsit_by_year(self) -> pd.DataFrame:
        if self.state.startsit_weekly_clean is None:
            self.block_startsit()
        return self.analyzer.plot_startsit_by_year(self.state.startsit_weekly_clean)

    def plot_cumulative_draft_waiver_startsit(
        self,
        *,
        waiver_baseline_name: str = "q25_team_season",
        draft_points_col: str = "Points_Added_Poly",
        manual_draft_only: bool = True,
        season_end_week: int = 17,
        ignore_weeks: Optional[Set[int]] = None,
        agg_mode: str = "mean_per_team_season",
    ) -> None:
        if self.state.draft_scored is None:
            self.block_score_draft()
        if self.state.waiver_stints is None or self.state.baseline_candidates is None:
            self.block_waiver_pipeline()
        if self.state.startsit_weekly_clean is None:
            self.block_startsit()

        baseline_value = self.state.baseline_candidates[waiver_baseline_name]
        self.analyzer.plot_cumulative_draft_waiver_startsit(
            self.state.draft_scored,
            self.state.waiver_stints,
            self.state.optimal_selected,
            self.state.startsit_weekly_clean,
            baseline_value,
            draft_points_col=draft_points_col,
            manual_draft_only=manual_draft_only,
            season_end_week=season_end_week,
            ignore_weeks=ignore_weeks,
            agg_mode=agg_mode,
        )
