"""Integration and deterministic validation tests for historical Golden Anchors and Agent 1 self-healing."""

from __future__ import annotations

import pytest

from src.agents.ingestion_healer_agent import SelfHealingIngestionAgent
from src.collectors import load_sources_config
from src.collectors.indec import (
    collect_indec_emae,
    collect_indec_salarios_historico,
    collect_indec_salarios_vigente,
)
from src.collectors.ministerio_economia import collect_gasto_publico
from src.collectors.sipa_trabajo import collect_sipa_all
from src.collectors.world_bank import collect_wb_gdp_growth, collect_wb_inflation
from src.validators.schema_and_golden import (
    load_golden_config,
    validate_series_dataframe,
)


@pytest.fixture(scope="module")
def sources_cfg():
    return load_sources_config()


@pytest.fixture(scope="module")
def golden_cfg():
    return load_golden_config()


def test_1_wb_inflation_golden_anchors(sources_cfg, golden_cfg):
    df = collect_wb_inflation(sources_cfg)
    res = validate_series_dataframe("wb_inflation", df, golden_cfg)
    assert res.passed, f"Validation failed for wb_inflation: {res.model_dump()}"
    assert abs(float(df.loc[1961, "inflation"]) - 20.310698) < 1e-4
    assert abs(float(df.loc[2017, "inflation"]) - 26.006379) < 1e-4
    assert abs(float(df.loc[2020, "inflation"]) - 40.083088) < 1e-4


def test_2_wb_gdp_growth_golden_anchors(sources_cfg, golden_cfg):
    df = collect_wb_gdp_growth(sources_cfg)
    res = validate_series_dataframe("wb_gdp_growth", df, golden_cfg)
    assert res.passed, f"Validation failed for wb_gdp_growth: {res.model_dump()}"
    assert abs(float(df.loc[1961, "gdpGrowth"]) - 5.427843) < 1e-4
    assert abs(float(df.loc[2011, "gdpGrowth"]) - 6.003952) < 1e-4
    assert abs(float(df.loc[2017, "gdpGrowth"]) - 2.818503) < 1e-4


def test_3_gasto_publico_golden_anchors(sources_cfg, golden_cfg):
    df = collect_gasto_publico(sources_cfg)
    res = validate_series_dataframe("gasto_publico", df, golden_cfg)
    assert res.passed, f"Validation failed for gasto_publico: {res.model_dump()}"
    assert abs(float(df.loc[1980, "pubExpToGDP"]) - 29.037146) < 1e-4
    assert abs(float(df.loc[2011, "pubExpToGDP"]) - 40.006699) < 1e-4


def test_4_sipa_modalidades_and_provincias_golden_anchors(sources_cfg, golden_cfg):
    df_mod, df_prov = collect_sipa_all(sources_cfg)

    res_mod = validate_series_dataframe("sipa_modalidades", df_mod, golden_cfg)
    assert res_mod.passed, f"Validation failed for sipa_modalidades: {res_mod.model_dump()}"
    row_2012 = df_mod.loc["2012-01-01"]
    assert abs(float(row_2012["Empleo privado"]) - 6106.0) < 0.1
    assert abs(float(row_2012["Empleo público"]) - 2543.7) < 0.1
    assert abs(float(row_2012["Total"]) - 10931.3) < 0.1

    res_prov = validate_series_dataframe("sipa_provincias", df_prov, golden_cfg)
    assert res_prov.passed, f"Validation failed for sipa_provincias: {res_prov.model_dump()}"
    assert abs(float(df_prov.loc["2009-01-01", "total"]) - 5659.961) < 1e-3
    assert abs(float(df_prov.loc["2009-02-01", "total"]) - 5647.381) < 1e-3


def test_5_and_6_indec_salarios_golden_anchors(sources_cfg, golden_cfg):
    df_vig = collect_indec_salarios_vigente(sources_cfg)
    res_vig = validate_series_dataframe("indec_salarios_vigente", df_vig, golden_cfg)
    assert res_vig.passed, f"Validation failed for indec_salarios_vigente: {res_vig.model_dump()}"
    assert abs(float(df_vig.loc["2015-10-01", "sectorPrivado"]) - 73.97) < 1e-2
    assert abs(float(df_vig.loc["2016-10-01", "sectorPrivado"]) - 100.0) < 1e-2

    df_hist = collect_indec_salarios_historico(sources_cfg)
    res_hist = validate_series_dataframe("indec_salarios_historico", df_hist, golden_cfg)
    assert res_hist.passed, f"Validation failed for indec_salarios_historico: {res_hist.model_dump()}"
    assert abs(float(df_hist.loc["2001-10-01", "sectorPrivado"]) - 14.22) < 1e-2
    assert abs(float(df_hist.loc["2001-10-01", "sectorPublico"]) - 27.99) < 1e-2


def test_7_indec_emae_golden_anchors(sources_cfg, golden_cfg):
    df_emae = collect_indec_emae(sources_cfg)
    res_emae = validate_series_dataframe("indec_emae", df_emae, golden_cfg)
    assert res_emae.passed, f"Validation failed for indec_emae: {res_emae.model_dump()}"
    row_2004 = df_emae.loc["2004-01-01"]
    assert abs(float(row_2004["gdp"]) - 92.6275) < 1e-3
    assert abs(float(row_2004["des"]) - 98.5292) < 1e-3
    assert abs(float(row_2004["tc"]) - 96.0499) < 1e-3


def test_self_healing_agent_recovers_from_shifted_offset(sources_cfg, golden_cfg):
    """Verifies that if the row offset is miscalibrated, Agent 1 discovers the true offset and passes the Golden Test."""
    healer = SelfHealingIngestionAgent(golden_config=golden_cfg)
    broken_cfg = {"row_offset": 11, "orig_col_idx": 2, "des_col_idx": 4, "tc_col_idx": 6}

    df_healed, heal_res = healer.attempt_heal(
        source_id="indec_emae",
        raw_path="data/raw_cache/indec_emae.xls",
        current_parser_config=broken_cfg,
        failure_reason="Simulated row offset shift",
        collector_fn=lambda ov: collect_indec_emae(sources_cfg, parser_override=ov),
    )
    assert heal_res.healed is True
    assert df_healed is not None
    assert heal_res.validation_result is not None and heal_res.validation_result.passed is True
    assert heal_res.proposed_override["row_offset"] == 4
