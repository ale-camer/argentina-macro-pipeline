"""Deterministic validators: Pydantic schemas + historical Golden Anchor checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class AnchorRule(BaseModel):
    period: Union[int, str]
    column: str
    expected: float
    tolerance: float = 0.01


class ShareAnchorRule(BaseModel):
    period: str
    shares_pct: Dict[str, float]
    tolerance: float = 0.02


class SourceGoldenSpec(BaseModel):
    description: str
    index_type: str  # "year_int" | "datetime_ms"
    min_year: Optional[int] = None
    start_date: Optional[str] = None
    required_columns: List[str]
    anchors: List[AnchorRule]
    share_anchors: Optional[ShareAnchorRule] = None


class GoldenConfig(BaseModel):
    version: str
    description: str
    default_tolerance: float = 0.01
    sources: Dict[str, SourceGoldenSpec]


class AnchorCheckDetail(BaseModel):
    period: str
    column: str
    expected: float
    actual: Optional[float]
    diff: Optional[float]
    tolerance: float
    passed: bool
    error: Optional[str] = None


class ValidationResult(BaseModel):
    source_id: str
    passed: bool
    row_count: int = 0
    first_period: Optional[str] = None
    last_period: Optional[str] = None
    schema_errors: List[str] = Field(default_factory=list)
    continuity_ok: bool = True
    anchor_checks: List[AnchorCheckDetail] = Field(default_factory=list)


def load_golden_config(config_path: Union[str, Path] = "config/golden_anchors.json") -> GoldenConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return GoldenConfig.model_validate(data)


def validate_series_dataframe(
    source_id: str,
    df: pd.DataFrame,
    golden_config: Optional[GoldenConfig] = None,
    config_path: Union[str, Path] = "config/golden_anchors.json",
) -> ValidationResult:
    """Deterministically validates schema, absence of NaN in closed periods,
    temporal continuity, and exact match within tolerance against Golden Tests."""
    if golden_config is None:
        golden_config = load_golden_config(config_path)

    if source_id not in golden_config.sources:
        return ValidationResult(
            source_id=source_id,
            passed=False,
            schema_errors=[f"Source ID '{source_id}' not found in golden_anchors.json"],
        )

    spec = golden_config.sources[source_id]
    errors: List[str] = []

    if df is None or df.empty:
        return ValidationResult(
            source_id=source_id,
            passed=False,
            schema_errors=["DataFrame is empty or None."],
        )

    # 1. Required columns and numeric types
    missing_cols = [c for c in spec.required_columns if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    for col in spec.required_columns:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' is not numeric (dtype={df[col].dtype}).")
            # Ensure no NaN values in closed historical periods
            closed_slice = df[col].iloc[:-1] if len(df) > 1 else df[col]
            nan_count = int(closed_slice.isna().sum())
            if nan_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} NaN values in closed periods.")

    # 2. Strict temporal index continuity
    continuity_ok = True
    if spec.index_type == "year_int":
        try:
            idx_years = df.index.astype(int)
            diffs = np.diff(idx_years)
            if not np.all(diffs == 1):
                continuity_ok = False
                errors.append("Annual index is not strictly continuous (gaps detected).")
            first_period = str(int(idx_years[0]))
            last_period = str(int(idx_years[-1]))
        except Exception as exc:
            continuity_ok = False
            first_period = str(df.index[0])
            last_period = str(df.index[-1])
            errors.append(f"Invalid annual index: {exc}")
    elif spec.index_type == "datetime_ms":
        if not isinstance(df.index, pd.DatetimeIndex):
            continuity_ok = False
            errors.append("Index must be DatetimeIndex for monthly 'MS' frequency.")
            first_period = str(df.index[0])
            last_period = str(df.index[-1])
        else:
            expected_idx = pd.date_range(start=df.index[0], periods=len(df), freq="MS")
            if not df.index.equals(expected_idx):
                continuity_ok = False
                errors.append("DatetimeIndex does not follow strict monthly ('MS') continuity.")
            first_period = df.index[0].strftime("%Y-%m-%d")
            last_period = df.index[-1].strftime("%Y-%m-%d")
    else:
        first_period = str(df.index[0])
        last_period = str(df.index[-1])

    # 3. Historical Golden Anchor checks
    anchor_checks: List[AnchorCheckDetail] = []
    for anchor in spec.anchors:
        lookup_key: Any = anchor.period
        if spec.index_type == "datetime_ms":
            lookup_key = pd.Timestamp(str(anchor.period))
        elif spec.index_type == "year_int":
            lookup_key = int(anchor.period)

        if lookup_key not in df.index:
            anchor_checks.append(
                AnchorCheckDetail(
                    period=str(anchor.period),
                    column=anchor.column,
                    expected=anchor.expected,
                    actual=None,
                    diff=None,
                    tolerance=anchor.tolerance,
                    passed=False,
                    error=f"Period {anchor.period} not found in index.",
                )
            )
            continue

        if anchor.column not in df.columns:
            anchor_checks.append(
                AnchorCheckDetail(
                    period=str(anchor.period),
                    column=anchor.column,
                    expected=anchor.expected,
                    actual=None,
                    diff=None,
                    tolerance=anchor.tolerance,
                    passed=False,
                    error=f"Column {anchor.column} missing.",
                )
            )
            continue

        actual_val = float(df.loc[lookup_key, anchor.column])
        diff = abs(actual_val - anchor.expected)
        ok = bool(diff <= anchor.tolerance)
        anchor_checks.append(
            AnchorCheckDetail(
                period=str(anchor.period),
                column=anchor.column,
                expected=anchor.expected,
                actual=round(actual_val, 6),
                diff=round(diff, 6),
                tolerance=anchor.tolerance,
                passed=ok,
                error=None if ok else f"Value {actual_val:.6f} differs from {anchor.expected:.6f} by {diff:.6f} > {anchor.tolerance}",
            )
        )

    # 4. Share Anchor checks (e.g., SIPA modality shares on 2012-01-01)
    if spec.share_anchors is not None:
        sa = spec.share_anchors
        ts_key = pd.Timestamp(sa.period)
        if ts_key in df.index and "Total" in df.columns:
            total_val = float(df.loc[ts_key, "Total"])
            for col_name, exp_share in sa.shares_pct.items():
                if col_name in df.columns and total_val > 0:
                    actual_share = (float(df.loc[ts_key, col_name]) / total_val) * 100.0
                    diff = abs(actual_share - exp_share)
                    ok = bool(diff <= sa.tolerance)
                    anchor_checks.append(
                        AnchorCheckDetail(
                            period=f"{sa.period} [share %]",
                            column=col_name,
                            expected=exp_share,
                            actual=round(actual_share, 4),
                            diff=round(diff, 4),
                            tolerance=sa.tolerance,
                            passed=ok,
                            error=None if ok else f"Share {actual_share:.2f}% != {exp_share:.2f}%",
                        )
                    )

    all_anchors_ok = all(a.passed for a in anchor_checks) and len(anchor_checks) > 0
    passed = (len(errors) == 0) and continuity_ok and all_anchors_ok

    return ValidationResult(
        source_id=source_id,
        passed=passed,
        row_count=len(df),
        first_period=first_period,
        last_period=last_period,
        schema_errors=errors,
        continuity_ok=continuity_ok,
        anchor_checks=anchor_checks,
    )
