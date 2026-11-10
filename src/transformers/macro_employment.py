"""100% deterministic mathematical transformations:
- Deduplicated geometric series splicing
- Base-100 chained geometric indices
- Employment-output elasticity regression
- Cumulative point-to-point growth and intercensal demographic gap
- Exact real wage deflation (Fisher equation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


POPULATION_CENSO_2010 = 40_117_096
POPULATION_CENSO_2022 = 47_327_407


@dataclass
class TransformedMacroBundle:
    df_v1: pd.DataFrame
    v1_trends: Dict[Tuple[int, int], pd.Series]
    df_v2: pd.DataFrame
    v2_metrics: Dict[str, float]
    df_v3_geom: pd.DataFrame
    s_v3_pop: pd.Series
    v3_metrics: Dict[str, Any]
    df_v4_props: pd.DataFrame
    df_v4_summary: pd.DataFrame
    df_v5: pd.DataFrame
    s_salarios_nominal_chained: pd.Series


def compute_chained_gdp_index(gdp_growth_series: pd.Series, base_year: int = 1980) -> pd.Series:
    """Computes the chained geometric index Base `base_year` = 100: I_t = I_{t-1} * (1 + g_t/100)."""
    sub = gdp_growth_series.loc[base_year:].dropna()
    vals = [100.0]
    for g in sub.iloc[1:].values:
        vals.append(vals[-1] * (1.0 + float(g) / 100.0))
    return pd.Series(vals, index=sub.index, name=f"gdp_index_{base_year}")


def build_visual1_macro_series(
    gdp_growth: pd.DataFrame,
    public_expenditure: pd.DataFrame,
    inflation_rate: pd.DataFrame,
    start_year: int = 1980,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], pd.Series]]:
    """Prepares annual series and per-regime exponential trend lines for Visual 1."""
    max_yr = int(gdp_growth.index.max())
    df_v1 = (
        gdp_growth.loc[start_year:max_yr]
        .merge(public_expenditure, left_index=True, right_index=True, how="left")
        .merge(inflation_rate, left_index=True, right_index=True, how="left")
    )
    df_v1["gdp_index_1980"] = compute_chained_gdp_index(df_v1["gdpGrowth"], base_year=start_year)

    regime_windows = [(1980, 1991), (1991, 2011), (2011, 2023), (2023, max_yr)]
    trends: Dict[Tuple[int, int], pd.Series] = {}
    for yr_start, yr_end in regime_windows:
        sub = df_v1.loc[yr_start:yr_end, "gdp_index_1980"].dropna()
        if len(sub) >= 2:
            z = np.polyfit(sub.index.astype(float), np.log(sub.values), 1)
            x_vals = sub.index.astype(float)
            y_fit = np.exp(np.poly1d(z)(x_vals))
            trends[(yr_start, yr_end)] = pd.Series(y_fit, index=sub.index)

    return df_v1, trends


def calc_cagr_yr(df: pd.DataFrame, col: str, y0: int, y1: int) -> float:
    """Computes the compound annual growth rate (CAGR %) between the annual means of y0 and y1."""
    v0 = float(df.loc[f"{y0}-01-01":f"{y0}-12-01", col].mean())
    v1 = float(df.loc[f"{y1}-01-01":f"{y1}-12-01", col].mean())
    return ((v1 / v0) ** (1.0 / (y1 - y0)) - 1.0) * 100.0


def build_visual2_elasticity_series(
    n_jobs_per_province: pd.DataFrame,
    monthly_gdp: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Prepares monthly Base Jan-2009=100 series, correlations, and elasticity regression for Visual 2."""
    gdp_jobs = n_jobs_per_province.merge(monthly_gdp, left_index=True, right_index=True, how="inner")
    gdp_jobs["jobs_idx"] = (gdp_jobs["total"] / gdp_jobs["total"].iloc[0]) * 100.0
    gdp_jobs["emae_orig_idx"] = (gdp_jobs["gdp"] / gdp_jobs["gdp"].iloc[0]) * 100.0
    gdp_jobs["emae_des_idx"] = (gdp_jobs["des"] / gdp_jobs["des"].iloc[0]) * 100.0
    gdp_jobs["emae_tc_idx"] = (gdp_jobs["tc"] / gdp_jobs["tc"].iloc[0]) * 100.0

    corr_orig = round(float(np.corrcoef(gdp_jobs["total"], gdp_jobs["gdp"])[0, 1]) * 100.0, 2)
    corr_des = round(float(np.corrcoef(gdp_jobs["total"], gdp_jobs["des"])[0, 1]) * 100.0, 2)
    corr_tc = round(float(np.corrcoef(gdp_jobs["total"], gdp_jobs["tc"])[0, 1]) * 100.0, 2)

    # Latest full 12-month closed calendar year for comparable annual CAGR calculation
    year_counts = gdp_jobs.groupby(gdp_jobs.index.year)["total"].count()
    last_full_yr = int(year_counts[year_counts == 12].index.max())

    cagr_jobs_1 = calc_cagr_yr(gdp_jobs, "total", 2009, 2011)
    cagr_gdp_1 = calc_cagr_yr(gdp_jobs, "des", 2009, 2011)
    cagr_jobs_2 = calc_cagr_yr(gdp_jobs, "total", 2012, last_full_yr)
    cagr_gdp_2 = calc_cagr_yr(gdp_jobs, "des", 2012, last_full_yr)

    z_all = np.polyfit(gdp_jobs["emae_des_idx"], gdp_jobs["jobs_idx"], 1)
    beta_slope = float(z_all[0])
    beta_intercept = float(z_all[1])

    metrics = {
        "corr_orig": corr_orig,
        "corr_des": corr_des,
        "corr_tc": corr_tc,
        "cagr_jobs_2009_2011": round(cagr_jobs_1, 2),
        "cagr_gdp_2009_2011": round(cagr_gdp_1, 2),
        "cagr_jobs_2012_2025": round(cagr_jobs_2, 2),
        "cagr_gdp_2012_2025": round(cagr_gdp_2, 2),
        "beta_slope": round(beta_slope, 4),
        "beta_intercept": round(beta_intercept, 4),
        "last_full_year": float(last_full_yr),
    }
    return gdp_jobs, metrics


def build_visual3_modalities_growth(
    n_jobs_per_type: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Computes exact cumulative relative geometric growth (X_t / X_0 - 1) and intercensal population projection."""
    annual_pop_growth = (POPULATION_CENSO_2022 / POPULATION_CENSO_2010) ** (1.0 / (2022 - 2010)) - 1.0
    months_idx = np.arange(len(n_jobs_per_type), dtype=float)
    pop_series = pd.Series(
        ((1.0 + annual_pop_growth) ** (months_idx / 12.0) - 1.0) * 100.0,
        index=n_jobs_per_type.index,
        name="population_growth_pct",
    )

    geom_df = (n_jobs_per_type.div(n_jobs_per_type.iloc[0]) - 1.0) * 100.0
    private_end = float(geom_df["Empleo privado"].iloc[-1])
    pop_end = float(pop_series.iloc[-1])
    demographic_gap_pp = round(private_end - pop_end, 1)

    metrics = {
        "annual_pop_growth_pct": round(annual_pop_growth * 100.0, 4),
        "cumulative_pop_growth_pct": round(pop_end, 2),
        "private_cumulative_growth_pct": round(private_end, 2),
        "demographic_gap_pp": demographic_gap_pp,
        "latest_growth_by_modality": {
            col: round(float(geom_df[col].iloc[-1]), 2) for col in geom_df.columns
        },
    }
    return geom_df, pop_series, metrics


def build_visual4_structural_shift(
    n_jobs_per_type: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes percentage share by labor modality and net structural shift in percentage points."""
    cols_mod = [
        "Empleo privado",
        "Empleo público",
        "Monotributo",
        "Empleo en casas particulares",
        "Autónomos",
        "Monotributo Social",
    ]
    props = n_jobs_per_type[cols_mod].div(n_jobs_per_type["Total"], axis=0) * 100.0
    p_2012 = props.loc["2012-01-01"]
    p_2022 = props.loc["2022-03-01"]
    p_latest = props.iloc[-1]
    diff_pp = p_latest - p_2012

    summary_df = pd.DataFrame(
        {
            "share_2012_01": p_2012.values,
            "share_2022_03": p_2022.values,
            "share_latest": p_latest.values,
            "diff_pp": diff_pp.values,
        },
        index=cols_mod,
    )
    return props, summary_df


def splice_salaries_series(
    salaries_until_2015: pd.DataFrame,
    salaries_since_2015: pd.DataFrame,
    col: str = "sectorPrivado",
) -> pd.Series:
    """Deduplicated geometric splice on 2015-10-01 between historical (Base 2012) and current (Base Oct-2016) series."""
    s_hist = salaries_until_2015[col].dropna()
    s_curr = salaries_since_2015[col].dropna()

    overlap_date = pd.Timestamp("2015-10-01")
    if overlap_date not in s_hist.index or overlap_date not in s_curr.index:
        raise ValueError("Splice date 2015-10-01 must exist in both wage series.")

    anchor_hist = float(s_hist.loc[overlap_date])
    anchor_curr = float(s_curr.loc[overlap_date])
    s_curr_after = s_curr.loc[s_curr.index > overlap_date]

    scaled_after = anchor_hist * (s_curr_after / anchor_curr)
    chained = pd.concat([s_hist.loc[s_hist.index <= overlap_date], scaled_after])
    chained.name = f"{col}_chained"
    return chained


def build_visual5_real_wages(
    salaries_until_2015: pd.DataFrame,
    salaries_since_2015: pd.DataFrame,
    monthly_gdp: pd.DataFrame,
    inflation_rate: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Computes private formal real wages and annual average real GDP (Base 2012 = 100) + YoY variations."""
    priv_chained = splice_salaries_series(salaries_until_2015, salaries_since_2015, col="sectorPrivado")

    priv_df = pd.DataFrame({"sal_nom": priv_chained})
    priv_df["YEAR"] = priv_df.index.year
    full_yrs = priv_df.groupby("YEAR")["sal_nom"].count()
    full_yrs = full_yrs[full_yrs == 12].index
    sal_nom_yoy = (
        priv_df[priv_df["YEAR"].isin(full_yrs)]
        .groupby("YEAR")["sal_nom"]
        .mean()
        .pct_change()
        .dropna()
    )

    emae_df = monthly_gdp.copy()
    emae_df["YEAR"] = emae_df.index.year
    full_yrs_e = emae_df.groupby("YEAR")["gdp"].count()
    full_yrs_e = full_yrs_e[full_yrs_e == 12].index
    emae_yoy = (
        emae_df[emae_df["YEAR"].isin(full_yrs_e)]
        .groupby("YEAR")["gdp"]
        .mean()
        .pct_change()
        .dropna()
    )

    df_v5 = (
        pd.DataFrame({"sal_nom_yoy": sal_nom_yoy})
        .merge(inflation_rate / 100.0, left_index=True, right_index=True)
        .merge(pd.DataFrame({"gdp_yoy": emae_yoy}), left_index=True, right_index=True)
    )
    max_yr = int(df_v5.index.max())
    df_v5 = df_v5.loc[2005:max_yr].copy()
    df_v5["real_sal_yoy"] = (1.0 + df_v5["sal_nom_yoy"]) / (1.0 + df_v5["inflation"]) - 1.0

    years_all = df_v5.index.tolist()
    sal_idx = [100.0]
    gdp_idx = [100.0]
    for y in years_all[1:]:
        sal_idx.append(sal_idx[-1] * (1.0 + float(df_v5.loc[y, "real_sal_yoy"])))
        gdp_idx.append(gdp_idx[-1] * (1.0 + float(df_v5.loc[y, "gdp_yoy"])))

    df_v5["sal_real_idx"] = sal_idx
    df_v5["gdp_real_idx"] = gdp_idx
    df_v5["sal_real_idx_2012"] = (df_v5["sal_real_idx"] / float(df_v5.loc[2012, "sal_real_idx"])) * 100.0
    df_v5["gdp_real_idx_2012"] = (df_v5["gdp_real_idx"] / float(df_v5.loc[2012, "gdp_real_idx"])) * 100.0

    return df_v5, priv_chained


def run_all_transformations(
    raw_series: Dict[str, pd.DataFrame],
) -> TransformedMacroBundle:
    """Runs the complete deterministic transformation suite across the 8 validated series."""
    df_v1, v1_trends = build_visual1_macro_series(
        gdp_growth=raw_series["wb_gdp_growth"],
        public_expenditure=raw_series["gasto_publico"],
        inflation_rate=raw_series["wb_inflation"],
    )
    df_v2, v2_metrics = build_visual2_elasticity_series(
        n_jobs_per_province=raw_series["sipa_provincias"],
        monthly_gdp=raw_series["indec_emae"],
    )
    df_v3_geom, s_v3_pop, v3_metrics = build_visual3_modalities_growth(
        n_jobs_per_type=raw_series["sipa_modalidades"],
    )
    df_v4_props, df_v4_summary = build_visual4_structural_shift(
        n_jobs_per_type=raw_series["sipa_modalidades"],
    )
    df_v5, s_sal_chained = build_visual5_real_wages(
        salaries_until_2015=raw_series["indec_salarios_historico"],
        salaries_since_2015=raw_series["indec_salarios_vigente"],
        monthly_gdp=raw_series["indec_emae"],
        inflation_rate=raw_series["wb_inflation"],
    )

    return TransformedMacroBundle(
        df_v1=df_v1,
        v1_trends=v1_trends,
        df_v2=df_v2,
        v2_metrics=v2_metrics,
        df_v3_geom=df_v3_geom,
        s_v3_pop=s_v3_pop,
        v3_metrics=v3_metrics,
        df_v4_props=df_v4_props,
        df_v4_summary=df_v4_summary,
        df_v5=df_v5,
        s_salarios_nominal_chained=s_sal_chained,
    )
