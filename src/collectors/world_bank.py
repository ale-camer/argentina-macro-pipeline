"""Deterministic collector for World Bank annual indicators (GDP Deflator Inflation and Real GDP Growth)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.collectors import load_sources_config, resolve_raw_file


def parse_world_bank_zip(
    zip_path: Union[str, Path],
    col_name: str,
    country_code: str = "ARG",
    skiprows: int = 4,
) -> pd.DataFrame:
    """Reads the official World Bank ZIP archive and extracts the continuous annual series for the target country."""
    path = Path(zip_path)
    with zipfile.ZipFile(path, "r") as z:
        csv_candidates = [
            n for n in z.namelist() if n.startswith("API_") and n.endswith(".csv")
        ]
        if not csv_candidates:
            raise ValueError(f"No 'API_*.csv' file found inside {path}")
        csv_name = csv_candidates[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f, skiprows=skiprows)

    if "Country Code" not in df.columns:
        raise ValueError(f"Column 'Country Code' not found with skiprows={skiprows}")

    arg = df[df["Country Code"] == country_code]
    if arg.empty:
        raise ValueError(f"Country '{country_code}' not found in {csv_name}")

    yr_cols = [c for c in arg.columns if str(c).isdigit()]
    s = arg[yr_cols].iloc[0].dropna()
    result = pd.DataFrame(
        {col_name: s.values.astype(float)},
        index=s.index.astype(int),
    )
    result.index.name = "year"
    return result


def collect_wb_inflation(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    parser_override: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["wb_inflation"]
    p_cfg = {**cfg.get("parser_config", {}), **(parser_override or {})}
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)
    return parse_world_bank_zip(
        raw_path,
        col_name=p_cfg.get("value_col_name", "inflation"),
        country_code=p_cfg.get("country_code", "ARG"),
        skiprows=int(p_cfg.get("skiprows", 4)),
    )


def collect_wb_gdp_growth(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    parser_override: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["wb_gdp_growth"]
    p_cfg = {**cfg.get("parser_config", {}), **(parser_override or {})}
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)
    return parse_world_bank_zip(
        raw_path,
        col_name=p_cfg.get("value_col_name", "gdpGrowth"),
        country_code=p_cfg.get("country_code", "ARG"),
        skiprows=int(p_cfg.get("skiprows", 4)),
    )
