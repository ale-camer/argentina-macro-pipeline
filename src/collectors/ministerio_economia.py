"""Deterministic collector for Consolidated Public Spending as % of GDP (Ministry of Economy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.collectors import load_sources_config, resolve_raw_file


def parse_gasto_publico_xls(
    xls_path: Union[str, Path],
    sheet_name: str = "% del PIB",
    years_row_idx: int = 2,
    values_row_idx: int = 4,
    start_col_idx: int = 2,
    value_col_name: str = "pubExpToGDP",
) -> pd.DataFrame:
    """Extracts the annual Consolidated Total Public Spending (% of GDP) series from 1980 onwards."""
    gp_df = pd.read_excel(Path(xls_path), sheet_name=sheet_name)
    raw_years = (
        gp_df.iloc[years_row_idx, start_col_idx:]
        .astype(str)
        .str.replace("*", "", regex=False)
        .str.strip()
    )
    raw_vals = pd.to_numeric(gp_df.iloc[values_row_idx, start_col_idx:], errors="coerce")

    valid_mask = pd.to_numeric(raw_years, errors="coerce").notna() & raw_vals.notna()
    years = pd.to_numeric(raw_years[valid_mask]).astype(int).values
    vals = raw_vals[valid_mask].astype(float).values

    df = pd.DataFrame({value_col_name: vals}, index=years)
    df.index.name = "year"
    return df


def collect_gasto_publico(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    parser_override: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["gasto_publico"]
    p_cfg = {**cfg.get("parser_config", {}), **(parser_override or {})}
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)
    return parse_gasto_publico_xls(
        raw_path,
        sheet_name=p_cfg.get("sheet_name", "% del PIB"),
        years_row_idx=int(p_cfg.get("years_row_idx", 2)),
        values_row_idx=int(p_cfg.get("values_row_idx", 4)),
        start_col_idx=int(p_cfg.get("start_col_idx", 2)),
        value_col_name=p_cfg.get("value_col_name", "pubExpToGDP"),
    )
