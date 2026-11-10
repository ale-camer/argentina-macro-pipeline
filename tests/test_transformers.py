"""Unit tests for geometric compounding, deduplicated wage splicing, elasticity, and structural breaks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.collectors import load_sources_config
from src.collectors.indec import (
    collect_indec_emae,
    collect_indec_salarios_historico,
    collect_indec_salarios_vigente,
)
from src.collectors.ministerio_economia import collect_gasto_publico
from src.collectors.sipa_trabajo import collect_sipa_all
from src.collectors.world_bank import collect_wb_gdp_growth, collect_wb_inflation
from src.models.structural_breaks import evaluate_structural_breaks
from src.transformers.macro_employment import (
    compute_chained_gdp_index,
    run_all_transformations,
    splice_salaries_series,
)


@pytest.fixture(scope="module")
def bundle():
    cfg = load_sources_config()
    df_mod, df_prov = collect_sipa_all(cfg)
    raw_series = {
        "wb_inflation": collect_wb_inflation(cfg),
        "wb_gdp_growth": collect_wb_gdp_growth(cfg),
        "gasto_publico": collect_gasto_publico(cfg),
        "sipa_modalidades": df_mod,
        "sipa_provincias": df_prov,
        "indec_salarios_vigente": collect_indec_salarios_vigente(cfg),
        "indec_salarios_historico": collect_indec_salarios_historico(cfg),
        "indec_emae": collect_indec_emae(cfg),
    }
    return run_all_transformations(raw_series)


def test_geometric_compounding_vs_arithmetic_sum():
    """Verifies that a 50% drop followed by a 50% gain yields 75.0 (geometric) rather than 100.0 (arithmetic)."""
    s_growth = pd.Series([0.0, -50.0, 50.0], index=[2000, 2001, 2002])
    idx = compute_chained_gdp_index(s_growth, base_year=2000)
    assert np.isclose(idx.loc[2000], 100.0)
    assert np.isclose(idx.loc[2001], 50.0)
    assert np.isclose(idx.loc[2002], 75.0)


def test_salaries_splice_deduplication_and_continuity():
    """Verifies that splicing on 2015-10-01 does not duplicate the overlap month and preserves exact growth rates."""
    cfg = load_sources_config()
    s_hist = collect_indec_salarios_historico(cfg)
    s_curr = collect_indec_salarios_vigente(cfg)

    chained = splice_salaries_series(s_hist, s_curr, col="sectorPrivado")
    assert not chained.index.duplicated().any(), "Duplicate dates found after splicing on 2015-10-01"
    expected_idx = pd.date_range(start="2001-10-01", periods=len(chained), freq="MS")
    assert chained.index.equals(expected_idx)

    # Verify that the Nov-2015 / Oct-2015 monthly ratio matches the current series exactly
    ratio_chained = chained.loc["2015-11-01"] / chained.loc["2015-10-01"]
    ratio_curr = s_curr.loc["2015-11-01", "sectorPrivado"] / s_curr.loc["2015-10-01", "sectorPrivado"]
    assert np.isclose(ratio_chained, ratio_curr, rtol=1e-9)


def test_visual2_correlations_and_elasticity(bundle):
    m = bundle.v2_metrics
    assert m["corr_des"] > m["corr_orig"], "Seasonally adjusting EMAE must increase correlation with formal employment"
    assert np.isclose(m["corr_orig"], 51.93, atol=0.1)
    assert np.isclose(m["corr_des"], 76.72, atol=0.1)
    assert np.isclose(m["corr_tc"], 82.30, atol=0.1)
    assert np.isclose(m["beta_slope"], 0.4428, atol=0.02)


def test_visual3_and_visual4_consistency(bundle):
    # At the base month (2012-01-01), cumulative relative growth must be 0.0% across all modalities
    assert np.allclose(bundle.df_v3_geom.iloc[0].values, 0.0)
    # The sum of percentage point shifts (diff_pp) across all 6 modalities must equal 0.0 pp
    total_shift_pp = float(bundle.df_v4_summary["diff_pp"].sum())
    assert abs(total_shift_pp) < 0.05


def test_visual5_base_2012_normalization_and_structural_breaks(bundle):
    df_v5 = bundle.df_v5
    assert np.isclose(float(df_v5.loc[2012, "sal_real_idx_2012"]), 100.0)
    assert np.isclose(float(df_v5.loc[2012, "gdp_real_idx_2012"]), 100.0)

    brk = evaluate_structural_breaks(bundle)
    assert brk.spending_below_critical_40pct is True
    assert len(brk.pelt_emae_breaks) > 0
