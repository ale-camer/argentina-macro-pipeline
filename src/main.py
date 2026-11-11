"""Main monthly pipeline orchestrator for `argentina-macro-pipeline`."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.agents.ingestion_healer_agent import SelfHealingIngestionAgent
from src.agents.visual_and_regime_agent import VisualQAAndRegimeAgent
from src.collectors import (
    load_last_known_good,
    load_sources_config,
    resolve_raw_file,
    save_last_known_good,
)
from src.collectors.gcp_lake import sync_to_gcp_lake
from src.collectors.indec import (
    collect_indec_emae,
    collect_indec_salarios_historico,
    collect_indec_salarios_vigente,
)
from src.collectors.ministerio_economia import collect_gasto_publico
from src.collectors.sipa_trabajo import (
    parse_sipa_modalidades,
    parse_sipa_provincias,
)
from src.collectors.world_bank import collect_wb_gdp_growth, collect_wb_inflation
from src.models.forecaster import generate_macro_forecasts
from src.models.structural_breaks import evaluate_structural_breaks
from src.monitoring.notifier import (
    PipelineRunReport,
    PipelineStatus,
    SourceTelemetryRow,
    format_markdown_summary,
    publish_run_report,
)
from src.transformers.macro_employment import (
    TransformedMacroBundle,
    run_all_transformations,
)
from src.validators.schema_and_golden import (
    GoldenConfig,
    load_golden_config,
    validate_series_dataframe,
)
from src.visualizers.charts import generate_all_charts


def _get_previous_snapshot_info(
    snapshot_filename: str,
    index_type: str,
    lkg_dir: str = "data/last_known_good",
) -> Tuple[Optional[str], int]:
    try:
        df_prev = load_last_known_good(snapshot_filename, index_type=index_type, lkg_dir=lkg_dir)
        if df_prev.empty:
            return None, 0
        if index_type == "year_int":
            return str(int(df_prev.index[-1])), len(df_prev)
        return pd.Timestamp(df_prev.index[-1]).strftime("%Y-%m-%d"), len(df_prev)
    except Exception:
        return None, 0


def ingest_single_series_with_healing(
    source_id: str,
    snapshot_filename: str,
    raw_source_cfg_key: str,
    collector_fn: Callable[[Optional[Dict[str, Any]]], pd.DataFrame],
    sources_cfg: Dict[str, Any],
    golden_cfg: GoldenConfig,
    healer: SelfHealingIngestionAgent,
    lkg_dir: str = "data/last_known_good",
    raw_cache_dir: str = "data/raw_cache",
    bootstrap_raw_dir: str = "bootstrap/raw_data",
) -> Tuple[pd.DataFrame, SourceTelemetryRow]:
    """Executes Stage 1 for a single series: deterministic collector -> Golden Test -> Agent 1 (on failure) -> LKG fallback."""
    spec = golden_cfg.sources[source_id]
    prev_last, prev_count = _get_previous_snapshot_info(snapshot_filename, spec.index_type, lkg_dir=lkg_dir)

    # 1. 100% deterministic attempt
    failure_reason = ""
    try:
        df = collector_fn(None)
        val_res = validate_series_dataframe(source_id, df, golden_cfg)
        if val_res.passed:
            save_last_known_good(df, snapshot_filename, lkg_dir=lkg_dir)
            new_obs = max(0, val_res.row_count - prev_count) if prev_count > 0 else 0
            return df, SourceTelemetryRow(
                source_id=source_id,
                previous_last_period=prev_last or val_res.last_period,
                current_last_period=val_res.last_period,
                row_count=val_res.row_count,
                new_observations_added=new_obs,
                golden_test_passed=True,
                ingestion_mode="deterministic",
            )
        failure_reason = f"Golden/Schema errors: {val_res.schema_errors} | {[a.error for a in val_res.anchor_checks if not a.passed]}"
    except Exception as exc:
        failure_reason = f"Collector exception: {exc}"

    # 2. Encapsulated invocation of Agent 1 (`SelfHealingIngestionAgent`)
    src_entry = sources_cfg["sources"][raw_source_cfg_key]
    raw_path = resolve_raw_file(src_entry, raw_cache_dir=raw_cache_dir, bootstrap_raw_dir=bootstrap_raw_dir, prefer_network=False)
    df_healed, heal_res = healer.attempt_heal(
        source_id=source_id,
        raw_path=raw_path,
        current_parser_config=src_entry.get("parser_config", {}),
        failure_reason=failure_reason,
        collector_fn=lambda override: collector_fn(override),
    )
    if heal_res.healed and df_healed is not None and heal_res.validation_result is not None:
        save_last_known_good(df_healed, snapshot_filename, lkg_dir=lkg_dir)
        v_res = heal_res.validation_result
        new_obs = max(0, v_res.row_count - prev_count) if prev_count > 0 else 0
        return df_healed, SourceTelemetryRow(
            source_id=source_id,
            previous_last_period=prev_last or v_res.last_period,
            current_last_period=v_res.last_period,
            row_count=v_res.row_count,
            new_observations_added=new_obs,
            golden_test_passed=True,
            ingestion_mode="healed_by_agent",
            notes=f"Healed via {heal_res.mode_used}: {heal_res.proposed_override}",
        )

    # 3. Deterministic fallback to `data/last_known_good/`
    df_lkg = load_last_known_good(snapshot_filename, index_type=spec.index_type, lkg_dir=lkg_dir)
    val_lkg = validate_series_dataframe(source_id, df_lkg, golden_cfg)
    return df_lkg, SourceTelemetryRow(
        source_id=source_id,
        previous_last_period=prev_last,
        current_last_period=val_lkg.last_period,
        row_count=val_lkg.row_count,
        new_observations_added=0,
        golden_test_passed=val_lkg.passed,
        ingestion_mode="fallback_last_known_good",
        notes=f"Fallback LKG used after failure: {failure_reason}",
    )


def build_website_json_contract(
    bundle: TransformedMacroBundle,
    telemetry_rows: List[SourceTelemetryRow],
    forecasts: Dict[str, Any],
    charts_manifest: Dict[str, Dict[str, str]],
    out_path: Path = Path("outputs/macro_argentina_latest.json"),
) -> Path:
    """Exports `outputs/macro_argentina_latest.json` (~26 KB) ready to be consumed by `personal_website`."""
    df_v1 = bundle.df_v1
    df_v2 = bundle.df_v2
    df_v3 = bundle.df_v3_geom
    s_pop = bundle.s_v3_pop
    df_v4_sum = bundle.df_v4_summary
    df_v5 = bundle.df_v5

    payload: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_cutoff": {row.source_id: row.current_last_period for row in telemetry_rows},
        "headline_metrics": {
            "gdp_real_index_1980_base100_latest": round(float(df_v1["gdp_index_1980"].iloc[-1]), 2),
            "public_spending_pct_gdp_latest": round(float(df_v1["pubExpToGDP"].dropna().iloc[-1]), 2),
            "annual_inflation_deflator_latest_pct": round(float(df_v1["inflation"].dropna().iloc[-1]), 2),
            "elasticity_employment_gdp_beta": bundle.v2_metrics["beta_slope"],
            "correlation_jobs_emae_des_pct": bundle.v2_metrics["corr_des"],
            "correlation_jobs_emae_tc_pct": bundle.v2_metrics["corr_tc"],
            "demographic_gap_private_jobs_pp": bundle.v3_metrics["demographic_gap_pp"],
            "real_wage_index_2012_base100_latest": round(float(df_v5["sal_real_idx_2012"].iloc[-1]), 2),
        },
        "figures": {
            k: {
                "png": Path(v["png"]).name,
                "svg": Path(v["svg"]).name,
            }
            for k, v in charts_manifest.items()
        },
        "series": {
            "visual_1_annual_macro": {
                "year": [int(yr) for yr in df_v1.index],
                "gdp_growth_pct": [round(float(v), 3) if pd.notna(v) else None for v in df_v1["gdpGrowth"]],
                "gdp_index_1980": [round(float(v), 2) if pd.notna(v) else None for v in df_v1["gdp_index_1980"]],
                "pub_exp_pct_gdp": [round(float(v), 2) if pd.notna(v) else None for v in df_v1["pubExpToGDP"]],
                "inflation_pct": [round(float(v), 2) if pd.notna(v) else None for v in df_v1["inflation"]],
            },
            "visual_2_monthly_activity_jobs": {
                "period": [dt.strftime("%Y-%m-%d") for dt in df_v2.index],
                "private_jobs_thousands": [round(float(v), 2) for v in df_v2["total"]],
                "jobs_idx_2009": [round(float(v), 2) for v in df_v2["jobs_idx"]],
                "emae_orig_idx_2009": [round(float(v), 2) for v in df_v2["emae_orig_idx"]],
                "emae_des_idx_2009": [round(float(v), 2) for v in df_v2["emae_des_idx"]],
                "emae_tc_idx_2009": [round(float(v), 2) for v in df_v2["emae_tc_idx"]],
            },
            "visual_3_modalities_cumulative_pct": {
                "period": [dt.strftime("%Y-%m-%d") for dt in df_v3.index],
                "empleo_privado_pct": [round(float(v), 2) for v in df_v3["Empleo privado"]],
                "empleo_publico_pct": [round(float(v), 2) for v in df_v3["Empleo público"]],
                "monotributo_pct": [round(float(v), 2) for v in df_v3["Monotributo"]],
                "monotributo_social_pct": [round(float(v), 2) for v in df_v3["Monotributo Social"]],
                "autonomos_pct": [round(float(v), 2) for v in df_v3["Autónomos"]],
                "casas_particulares_pct": [round(float(v), 2) for v in df_v3["Empleo en casas particulares"]],
                "total_pct": [round(float(v), 2) for v in df_v3["Total"]],
                "population_proj_pct": [round(float(s_pop.loc[dt]), 2) for dt in df_v3.index],
            },
            "visual_4_structural_shift_summary": [
                {
                    "modality": str(mod),
                    "share_2012_01_pct": round(float(row["share_2012_01"]), 2),
                    "share_2022_03_pct": round(float(row["share_2022_03"]), 2),
                    "share_latest_pct": round(float(row["share_latest"]), 2),
                    "net_shift_pp": round(float(row["diff_pp"]), 2),
                }
                for mod, row in df_v4_sum.iterrows()
            ],
            "visual_5_real_wages_annual": {
                "year": [int(yr) for yr in df_v5.index],
                "sal_real_idx_2012": [round(float(v), 2) for v in df_v5["sal_real_idx_2012"]],
                "gdp_real_idx_2012": [round(float(v), 2) for v in df_v5["gdp_real_idx_2012"]],
                "real_sal_yoy_pct": [round(float(v) * 100.0, 2) for v in df_v5["real_sal_yoy"]],
                "gdp_yoy_pct": [round(float(v) * 100.0, 2) for v in df_v5["gdp_yoy"]],
            },
        },
        "forecasts_6m_sarimax": forecasts,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return out_path


def run_pipeline(prefer_network: bool = False, sync_lake: bool = False) -> PipelineRunReport:
    """Executes all 4 pipeline stages end-to-end and returns the structured telemetry report."""
    t0 = time.perf_counter()
    sources_cfg = load_sources_config()
    golden_cfg = load_golden_config()
    healer = SelfHealingIngestionAgent(golden_config=golden_cfg)
    visual_agent = VisualQAAndRegimeAgent()

    sipa_raw_path = resolve_raw_file(
        sources_cfg["sources"]["sipa_trabajo_registrado"],
        prefer_network=prefer_network,
    )
    sipa_excel = pd.ExcelFile(sipa_raw_path)

    series_tasks: List[Tuple[str, str, str, Callable[[Optional[Dict[str, Any]]], pd.DataFrame]]] = [
        (
            "wb_inflation",
            "wb_inflation.csv",
            "wb_inflation",
            lambda ov: collect_wb_inflation(sources_cfg, prefer_network=prefer_network, parser_override=ov),
        ),
        (
            "wb_gdp_growth",
            "wb_gdp_growth.csv",
            "wb_gdp_growth",
            lambda ov: collect_wb_gdp_growth(sources_cfg, prefer_network=prefer_network, parser_override=ov),
        ),
        (
            "gasto_publico",
            "gasto_publico.csv",
            "gasto_publico",
            lambda ov: collect_gasto_publico(sources_cfg, prefer_network=prefer_network, parser_override=ov),
        ),
        (
            "sipa_modalidades",
            "sipa_modalidades.csv",
            "sipa_trabajo_registrado",
            lambda ov: parse_sipa_modalidades(
                sipa_excel,
                row_offset=int((ov or {}).get("row_offset", 1)),
            ),
        ),
        (
            "sipa_provincias",
            "sipa_provincias.csv",
            "sipa_trabajo_registrado",
            lambda ov: parse_sipa_provincias(
                sipa_excel,
                row_offset=int((ov or {}).get("row_offset", 1)),
            ),
        ),
        (
            "indec_salarios_vigente",
            "indec_salarios_vigente.csv",
            "indec_salarios_vigente",
            lambda ov: collect_indec_salarios_vigente(sources_cfg, prefer_network=prefer_network, parser_override=ov),
        ),
        (
            "indec_salarios_historico",
            "indec_salarios_historico.csv",
            "indec_salarios_historico",
            lambda ov: collect_indec_salarios_historico(sources_cfg, prefer_network=prefer_network, parser_override=ov),
        ),
        (
            "indec_emae",
            "indec_emae.csv",
            "indec_emae",
            lambda ov: collect_indec_emae(sources_cfg, prefer_network=prefer_network, parser_override=ov),
        ),
    ]

    raw_series: Dict[str, pd.DataFrame] = {}
    telemetry_rows: List[SourceTelemetryRow] = []

    for sid, snap, cfg_key, fn in series_tasks:
        df_s, tel_row = ingest_single_series_with_healing(
            source_id=sid,
            snapshot_filename=snap,
            raw_source_cfg_key=cfg_key,
            collector_fn=fn,
            sources_cfg=sources_cfg,
            golden_cfg=golden_cfg,
            healer=healer,
        )
        raw_series[sid] = df_s
        telemetry_rows.append(tel_row)

    # Stage 2: Mathematical Transformations & Statistical Models (100% Deterministic)
    bundle = run_all_transformations(raw_series)
    break_report = evaluate_structural_breaks(bundle)
    forecasts = generate_macro_forecasts(bundle.df_v2, steps=6)

    # Stage 3: Chart Generation + Agent 2 (`VisualQAAndRegimeAgent`)
    charts_manifest = generate_all_charts(bundle, figures_dir="outputs/figures")
    qa_report = visual_agent.inspect_visuals(charts_manifest)
    pr_proposal = visual_agent.draft_regime_shift_pr_if_needed(break_report, output_dir="outputs")

    # Stage 4: Output Contract + Always-On Monitoring & Alerts
    contract_path = build_website_json_contract(
        bundle=bundle,
        telemetry_rows=telemetry_rows,
        forecasts=forecasts,
        charts_manifest=charts_manifest,
        out_path=Path("outputs/macro_argentina_latest.json"),
    )

    modes = {r.ingestion_mode for r in telemetry_rows}
    if not all(r.golden_test_passed for r in telemetry_rows) or not qa_report.visual_qa_passed:
        overall_status: PipelineStatus = "FAILED"
    elif "fallback_last_known_good" in modes:
        overall_status = "FALLBACK_USED"
    elif "healed_by_agent" in modes:
        overall_status = "HEALED_BY_AGENT"
    elif pr_proposal.pr_opened:
        overall_status = "REGIME_SHIFT_PR_OPENED"
    else:
        overall_status = "OK_DETERMINISTIC"

    outputs_map = {
        "macro_contract_json": str(contract_path),
        "run_report_json": "outputs/run_report_latest.json",
        "figures_dir": "outputs/figures",
    }
    if sync_lake:
        lake_info = sync_to_gcp_lake()
        outputs_map.update(lake_info)

    report = PipelineRunReport(
        run_timestamp_utc=datetime.now(timezone.utc).isoformat(),
        duration_sec=round(time.perf_counter() - t0, 2),
        status=overall_status,
        sources_summary=telemetry_rows,
        visual_qa_summary=f"{qa_report.figures_passed}/{qa_report.figures_checked} PASS",
        visual_qa_report=qa_report,
        structural_breaks=break_report,
        regime_pr_proposal=pr_proposal,
        outputs_generated=outputs_map,
    )
    publish_run_report(report, report_path="outputs/run_report_latest.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deterministic argentina-macro-pipeline engine.")
    parser.add_argument(
        "--network",
        action="store_true",
        help="Download fresh data live from the official sources before validating.",
    )
    parser.add_argument(
        "--sync-lake",
        action="store_true",
        help="Synchronize Bronze, Silver, and Gold layers to GCS and BigQuery in project argentina-macro-pipeline.",
    )
    args = parser.parse_args()
    prefer_net = args.network or os.environ.get("PREFER_NETWORK", "").lower() in ("1", "true", "yes")
    sync_lake = args.sync_lake or os.environ.get("GCP_LAKE_SYNC", "").lower() in ("1", "true", "yes")
    report = run_pipeline(prefer_network=prefer_net, sync_lake=sync_lake)
    print(format_markdown_summary(report))


if __name__ == "__main__":
    main()
