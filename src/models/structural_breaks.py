"""Deterministic statistical detection of structural regime breaks (PELT + quantitative rules)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import ruptures as rpt
import yaml
from pydantic import BaseModel, Field

from src.transformers.macro_employment import TransformedMacroBundle


class StructuralBreakReport(BaseModel):
    regime_shift_detected: bool = False
    evaluated_up_to_date: str
    pelt_emae_breaks: List[str] = Field(default_factory=list)
    pelt_jobs_breaks: List[str] = Field(default_factory=list)
    latest_24m_emae_cagr_pct: float
    latest_24m_jobs_cagr_pct: float
    latest_pub_exp_gdp_pct: float
    spending_below_critical_40pct: bool
    trigger_reasons: List[str] = Field(default_factory=list)


def load_regimes_config(config_path: Union[str, Path] = "config/regimes.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_pelt_dates(series: pd.Series, min_size: int = 12, penalty: float = 5.0) -> List[str]:
    """Runs the PELT (Pruned Exact Linear Time) algorithm on the standardized series."""
    clean = series.dropna()
    if len(clean) < min_size * 2:
        return []
    vals = clean.values.astype(float)
    std = float(np.std(vals)) or 1.0
    norm_vals = (vals - float(np.mean(vals))) / std

    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(norm_vals)
    bkps = algo.predict(pen=penalty)
    break_dates: List[str] = []
    for idx in bkps[:-1]:
        if 0 < idx < len(clean):
            dt = clean.index[idx]
            break_dates.append(dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt))
    return break_dates


def compute_annualized_slope_pct(series: pd.Series, window_months: int = 24) -> float:
    """Annualized log-linear slope (% per year) over the trailing `window_months` window."""
    tail = series.dropna().iloc[-window_months:]
    if len(tail) < 6:
        return 0.0
    t_years = np.arange(len(tail), dtype=float) / 12.0
    slope, _ = np.polyfit(t_years, np.log(tail.values.astype(float)), 1)
    return float((np.exp(slope) - 1.0) * 100.0)


def evaluate_structural_breaks(
    bundle: TransformedMacroBundle,
    regimes_cfg: Optional[Dict[str, Any]] = None,
    config_path: Union[str, Path] = "config/regimes.yaml",
    force_check_new_regime: bool = False,
) -> StructuralBreakReport:
    """Deterministically evaluates whether incoming observations justify opening a new regime PR proposal."""
    if regimes_cfg is None:
        regimes_cfg = load_regimes_config(config_path)

    rules = regimes_cfg.get("structural_break_rules", {})
    min_size = int(rules.get("pelt_min_size", 12))
    penalty = float(rules.get("pelt_penalty", 5.0))
    window_m = int(rules.get("post_2024_window_months", 24))
    slope_thresh = float(rules.get("sustained_slope_threshold_pct_per_year", 2.5))
    crit_spend = float(rules.get("critical_spending_gdp_threshold", 40.0))
    trigger_after = pd.Timestamp(rules.get("trigger_regime_shift_after_date", "2026-12-01"))

    emae_des = bundle.df_v2["emae_des_idx"]
    jobs_idx = bundle.df_v2["jobs_idx"]

    emae_breaks = detect_pelt_dates(emae_des, min_size=min_size, penalty=penalty)
    jobs_breaks = detect_pelt_dates(jobs_idx, min_size=min_size, penalty=penalty)

    slope_emae = round(compute_annualized_slope_pct(emae_des, window_months=window_m), 2)
    slope_jobs = round(compute_annualized_slope_pct(jobs_idx, window_months=window_m), 2)

    pub_exp_clean = bundle.df_v1["pubExpToGDP"].dropna()
    latest_pub_exp = round(float(pub_exp_clean.iloc[-1]), 2)
    below_40 = bool(latest_pub_exp < crit_spend)

    last_date = bundle.df_v2.index[-1]
    last_date_str = last_date.strftime("%Y-%m-%d")

    reasons: List[str] = []
    if last_date > trigger_after or force_check_new_regime:
        if abs(slope_emae) >= slope_thresh and abs(slope_jobs) >= slope_thresh:
            reasons.append(
                f"Sustained 24m slope exceeds threshold ({slope_emae:+.2f}%/yr EMAE, {slope_jobs:+.2f}%/yr Jobs)"
            )
        recent_breaks = [d for d in emae_breaks if d >= "2025-06-01"]
        if recent_breaks:
            reasons.append(f"Statistical PELT break detected in Seasonally Adjusted EMAE: {recent_breaks}")

    return StructuralBreakReport(
        regime_shift_detected=len(reasons) > 0,
        evaluated_up_to_date=last_date_str,
        pelt_emae_breaks=emae_breaks,
        pelt_jobs_breaks=jobs_breaks,
        latest_24m_emae_cagr_pct=slope_emae,
        latest_24m_jobs_cagr_pct=slope_jobs,
        latest_pub_exp_gdp_pct=latest_pub_exp,
        spending_below_critical_40pct=below_40,
        trigger_reasons=reasons,
    )
