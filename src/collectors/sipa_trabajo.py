"""Deterministic collector for SIPA Registered Employment (Modalities T.2.1 and Provinces A.5.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from src.collectors import load_sources_config, resolve_raw_file


DEFAULT_MODALIDADES_COLS = [
    ("Unnamed: 1", "Empleo privado"),
    ("Unnamed: 2", "Empleo público"),
    ("Unnamed: 3", "Empleo en casas particulares"),
    ("Unnamed: 4", "Autónomos"),
    ("Unnamed: 5", "Monotributo"),
    ("Unnamed: 6", "Monotributo Social"),
    ("Unnamed: 7", "Total"),
]


def parse_sipa_modalidades(
    xlsx_path: Union[str, Path, pd.ExcelFile],
    sheet_name: str = "T.2.1",
    row_offset: int = 1,
    start_date: str = "2012-01-01",
    check_numeric_col: str = "Unnamed: 7",
    col_indices: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Extracts Series 4a: SIPA Registered Employment by Modality from 01/2012 onwards (Sheet T.2.1)."""
    df_t21 = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df_clean = df_t21.iloc[row_offset:].copy()

    if col_indices is not None and len(col_indices) == 7:
        target_raw_cols = [df_clean.columns[i] for i in col_indices]
        check_col = target_raw_cols[-1]
    else:
        target_raw_cols = [c[0] for c in DEFAULT_MODALIDADES_COLS]
        check_col = check_numeric_col

    df_clean = df_clean[pd.to_numeric(df_clean[check_col], errors="coerce").notna()]
    idx_t21 = pd.date_range(start=start_date, periods=len(df_clean), freq="MS")

    out_cols = [c[1] for c in DEFAULT_MODALIDADES_COLS]
    n_jobs_per_type = pd.DataFrame(
        df_clean[target_raw_cols].astype(float).values,
        index=idx_t21,
        columns=out_cols,
    )
    n_jobs_per_type.index.name = "period"
    return n_jobs_per_type


def parse_sipa_provincias(
    xlsx_path: Union[str, Path, pd.ExcelFile],
    sheet_name: str = "A.5.1",
    row_offset: int = 1,
    start_date: str = "2009-01-01",
    check_numeric_col: str = "Unnamed: 1",
    start_col_idx: int = 1,
) -> pd.DataFrame:
    """Extracts Series 4b: Consolidated Provincial Private Formal Employment from 01/2009 onwards (Sheet A.5.1)."""
    a51 = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    a51_clean = a51.iloc[row_offset:].copy()

    check_col = check_numeric_col if check_numeric_col in a51_clean.columns else a51_clean.columns[start_col_idx]
    a51_clean = a51_clean[pd.to_numeric(a51_clean[check_col], errors="coerce").notna()]
    prov_cols = [
        c for c in a51_clean.columns[start_col_idx:]
        if pd.to_numeric(a51_clean[c], errors="coerce").notna().any()
    ]
    idx_a51 = pd.date_range(start=start_date, periods=len(a51_clean), freq="MS")
    n_jobs_per_province = pd.DataFrame(
        {"total": a51_clean[prov_cols].astype(float).sum(axis=1).values},
        index=idx_a51,
    )
    n_jobs_per_province.index.name = "period"
    return n_jobs_per_province


def collect_sipa_all(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    modalidades_override: Optional[Dict[str, Any]] = None,
    provincias_override: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads/resolves the single SIPA workbook and returns (df_modalidades, df_provincias)."""
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["sipa_trabajo_registrado"]
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)

    mod_cfg = {**cfg.get("parser_config", {}).get("modalidades", {}), **(modalidades_override or {})}
    prov_cfg = {**cfg.get("parser_config", {}).get("provincias", {}), **(provincias_override or {})}

    sipa_xls = pd.ExcelFile(raw_path)
    df_mod = parse_sipa_modalidades(
        sipa_xls,
        sheet_name=mod_cfg.get("sheet_name", "T.2.1"),
        row_offset=int(mod_cfg.get("row_offset", 1)),
        start_date=mod_cfg.get("start_date", "2012-01-01"),
        check_numeric_col=mod_cfg.get("check_numeric_col", "Unnamed: 7"),
        col_indices=mod_cfg.get("col_indices"),
    )
    df_prov = parse_sipa_provincias(
        sipa_xls,
        sheet_name=prov_cfg.get("sheet_name", "A.5.1"),
        row_offset=int(prov_cfg.get("row_offset", 1)),
        start_date=prov_cfg.get("start_date", "2009-01-01"),
        check_numeric_col=prov_cfg.get("check_numeric_col", "Unnamed: 1"),
        start_col_idx=int(prov_cfg.get("start_col_idx", 1)),
    )
    return df_mod, df_prov
