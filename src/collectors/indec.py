"""Deterministic collector for INDEC series:
- Source 5: Current Wage Index (CSV, 10/2015–Present)
- Source 6: Historical Spliced Wage Index (XLS, 10/2001–10/2015)
- Source 7: EMAE Base 2004=100 (XLS, 01/2004–Present)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.collectors import load_sources_config, resolve_raw_file


def parse_indec_salarios_vigente(
    csv_path: Union[str, Path],
    sep: str = ";",
    decimal: str = ",",
    encoding: str = "latin1",
    start_date: str = "2015-10-01",
    private_col: str = "IS_sector_privado_registrado",
    public_col: str = "IS_sector_publico",
) -> pd.DataFrame:
    df_sal_csv = pd.read_csv(Path(csv_path), sep=sep, decimal=decimal, encoding=encoding)
    df_sal_csv = df_sal_csv[
        pd.to_numeric(df_sal_csv[private_col], errors="coerce").notna()
    ].copy()
    idx_sal_csv = pd.date_range(start=start_date, periods=len(df_sal_csv), freq="MS")
    out = pd.DataFrame(
        {
            "sectorPrivado": df_sal_csv[private_col].astype(float).values,
            "sectorPublico": df_sal_csv[public_col].astype(float).values,
        },
        index=idx_sal_csv,
    )
    out.index.name = "period"
    return out


def parse_indec_salarios_historico(
    xls_path: Union[str, Path],
    sheet_name: str = "Serie histórica IS",
    row_offset: int = 6,
    start_date: str = "2001-10-01",
    private_col: str = "Unnamed: 1",
    public_col: str = "Unnamed: 3",
) -> pd.DataFrame:
    df_sh = pd.read_excel(Path(xls_path), sheet_name=sheet_name)
    df_sh_clean = df_sh.iloc[row_offset:].copy()
    df_sh_clean = df_sh_clean[pd.to_numeric(df_sh_clean[private_col], errors="coerce").notna()]
    idx_sh = pd.date_range(start=start_date, periods=len(df_sh_clean), freq="MS")
    out = pd.DataFrame(
        {
            "sectorPrivado": df_sh_clean[private_col].astype(float).values,
            "sectorPublico": df_sh_clean[public_col].astype(float).values,
        },
        index=idx_sh,
    )
    out.index.name = "period"
    return out


def parse_indec_emae(
    xls_path: Union[str, Path],
    row_offset: int = 4,
    start_date: str = "2004-01-01",
    orig_col_idx: int = 2,
    des_col_idx: int = 4,
    tc_col_idx: int = 6,
) -> pd.DataFrame:
    df_emae = pd.read_excel(Path(xls_path))
    first_val = pd.to_numeric(df_emae.iloc[row_offset, orig_col_idx], errors="coerce")
    if pd.isna(first_val):
        raise ValueError(f"Row at row_offset={row_offset} does not contain a valid initial numeric value.")
    emae_orig = pd.to_numeric(df_emae.iloc[row_offset:, orig_col_idx], errors="coerce").dropna().values
    emae_des = pd.to_numeric(df_emae.iloc[row_offset:, des_col_idx], errors="coerce").dropna().values
    emae_tc = pd.to_numeric(df_emae.iloc[row_offset:, tc_col_idx], errors="coerce").dropna().values

    min_len = min(len(emae_orig), len(emae_des), len(emae_tc))
    idx_emae = pd.date_range(start=start_date, periods=min_len, freq="MS")
    out = pd.DataFrame(
        {
            "gdp": emae_orig[:min_len].astype(float),
            "des": emae_des[:min_len].astype(float),
            "tc": emae_tc[:min_len].astype(float),
        },
        index=idx_emae,
    )
    out.index.name = "period"
    return out


def collect_indec_salarios_vigente(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    parser_override: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["indec_salarios_vigente"]
    p_cfg = {**cfg.get("parser_config", {}), **(parser_override or {})}
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)
    return parse_indec_salarios_vigente(
        raw_path,
        sep=p_cfg.get("sep", ";"),
        decimal=p_cfg.get("decimal", ","),
        encoding=p_cfg.get("encoding", "latin1"),
        start_date=p_cfg.get("start_date", "2015-10-01"),
        private_col=p_cfg.get("private_col", "IS_sector_privado_registrado"),
        public_col=p_cfg.get("public_col", "IS_sector_publico"),
    )


def collect_indec_salarios_historico(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    parser_override: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["indec_salarios_historico"]
    p_cfg = {**cfg.get("parser_config", {}), **(parser_override or {})}
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)
    return parse_indec_salarios_historico(
        raw_path,
        sheet_name=p_cfg.get("sheet_name", "Serie histórica IS"),
        row_offset=int(p_cfg.get("row_offset", 6)),
        start_date=p_cfg.get("start_date", "2001-10-01"),
        private_col=p_cfg.get("private_col", "Unnamed: 1"),
        public_col=p_cfg.get("public_col", "Unnamed: 3"),
    )


def collect_indec_emae(
    sources_cfg: Optional[Dict[str, Any]] = None,
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    parser_override: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if sources_cfg is None:
        sources_cfg = load_sources_config()
    cfg = sources_cfg["sources"]["indec_emae"]
    p_cfg = {**cfg.get("parser_config", {}), **(parser_override or {})}
    raw_path = resolve_raw_file(cfg, raw_cache_dir, bootstrap_raw_dir, prefer_network=prefer_network)
    return parse_indec_emae(
        raw_path,
        row_offset=int(p_cfg.get("row_offset", 4)),
        start_date=p_cfg.get("start_date", "2004-01-01"),
        orig_col_idx=int(p_cfg.get("orig_col_idx", 2)),
        des_col_idx=int(p_cfg.get("des_col_idx", 4)),
        tc_col_idx=int(p_cfg.get("tc_col_idx", 6)),
    )
