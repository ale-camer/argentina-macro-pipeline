"""Agent 1 (`SelfHealingIngestionAgent`):
Invoked EXCLUSIVELY upon download failures or deterministic Golden Test failures.
Inspects the first 25 rows/columns of the raw file or landing page DOM,
proposes updated offsets (`row_offset`, `col_indices`, etc.), and verifies the proposal
against 100% of the deterministic Golden Tests before accepting it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field

from src.validators.schema_and_golden import (
    GoldenConfig,
    ValidationResult,
    load_golden_config,
    validate_series_dataframe,
)


class HealerResult(BaseModel):
    source_id: str
    healed: bool
    mode_used: str  # "gemini_llm" | "deterministic_grid_search" | "failed"
    proposed_override: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    diagnostics: List[str] = Field(default_factory=list)


class SelfHealingIngestionAgent:
    """Encapsulated self-healing ingestion agent with mandatory deterministic Golden Test verification."""

    def __init__(
        self,
        golden_config: Optional[GoldenConfig] = None,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self.golden_config = golden_config or load_golden_config()
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def inspect_excel_head(self, raw_path: Union[str, Path], sheet_name: Optional[str] = None, max_rows: int = 25) -> str:
        """Extracts a compact preview of the first 25 rows and 12 columns of the raw Excel/CSV for diagnostics."""
        path = Path(raw_path)
        try:
            if path.suffix.lower() == ".csv":
                df_head = pd.read_csv(path, nrows=max_rows, sep=None, engine="python", encoding="latin1")
            else:
                df_head = pd.read_excel(path, sheet_name=sheet_name or 0, nrows=max_rows)
            return df_head.iloc[:max_rows, :12].to_string()
        except Exception as exc:
            return f"Error reading preview of {path}: {exc}"

    def _propose_override_via_llm(
        self,
        source_id: str,
        raw_preview: str,
        current_config: Dict[str, Any],
        failure_reason: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        try:
            from google import genai
            from google.genai import types

            spec = self.golden_config.sources[source_id]
            anchors_desc = [a.model_dump() for a in spec.anchors]
            prompt = (
                f"You are a data engineering agent repairing the parser for source '{source_id}'.\n"
                f"Current failing configuration: {json.dumps(current_config)}\n"
                f"Failure reason: {failure_reason}\n"
                f"Historical benchmarks (Golden Tests) that must match to 0.01 precision: {json.dumps(anchors_desc)}\n"
                f"First 25 rows of the raw file:\n{raw_preview}\n\n"
                "Return ONLY a JSON object with the corrected `parser_config` keys "
                "(e.g., `row_offset`, `years_row_idx`, `values_row_idx`, `start_col_idx`, "
                "`orig_col_idx`, `des_col_idx`, `tc_col_idx`)."
            )
            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            if resp.text:
                parsed = json.loads(resp.text)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            return None
        return None

    def _generate_candidate_overrides(self, source_id: str) -> List[Dict[str, Any]]:
        """Generates candidate row/column offset shifts for deterministic self-healing when no API key is set."""
        candidates: List[Dict[str, Any]] = []
        if source_id in ("wb_inflation", "wb_gdp_growth"):
            for skip in range(0, 10):
                candidates.append({"skiprows": skip})
        elif source_id == "gasto_publico":
            for y_row in range(0, 8):
                for v_row in range(y_row + 1, 12):
                    for s_col in (1, 2, 3):
                        candidates.append(
                            {"years_row_idx": y_row, "values_row_idx": v_row, "start_col_idx": s_col}
                        )
        elif source_id in ("sipa_modalidades", "sipa_provincias", "indec_salarios_historico"):
            for r_off in range(0, 15):
                candidates.append({"row_offset": r_off})
        elif source_id == "indec_emae":
            for r_off in range(0, 12):
                candidates.append({"row_offset": r_off, "orig_col_idx": 2, "des_col_idx": 4, "tc_col_idx": 6})
                candidates.append({"row_offset": r_off, "orig_col_idx": 1, "des_col_idx": 3, "tc_col_idx": 5})
        return candidates

    def attempt_heal(
        self,
        source_id: str,
        raw_path: Union[str, Path],
        current_parser_config: Dict[str, Any],
        failure_reason: str,
        collector_fn: Callable[[Dict[str, Any]], pd.DataFrame],
    ) -> Tuple[Optional[pd.DataFrame], HealerResult]:
        """Deduces updated offsets and validates every candidate against `validate_series_dataframe`.
        Only accepts a configuration if it passes 100% of the deterministic Golden Tests.
        """
        diagnostics: List[str] = [f"Starting self-healing for '{source_id}'. Initial failure: {failure_reason}"]
        preview = self.inspect_excel_head(raw_path, sheet_name=current_parser_config.get("sheet_name"))

        # 1. Try Gemini LLM if GEMINI_API_KEY is configured
        llm_override = self._propose_override_via_llm(
            source_id=source_id,
            raw_preview=preview,
            current_config=current_parser_config,
            failure_reason=failure_reason,
        )
        if llm_override:
            diagnostics.append(f"Candidate proposed by Gemini LLM: {llm_override}")
            try:
                df_cand = collector_fn(llm_override)
                val_res = validate_series_dataframe(source_id, df_cand, self.golden_config)
                if val_res.passed:
                    diagnostics.append("Deterministic Golden Test PASSED 100% with LLM proposal.")
                    return df_cand, HealerResult(
                        source_id=source_id,
                        healed=True,
                        mode_used="gemini_llm",
                        proposed_override=llm_override,
                        validation_result=val_res,
                        diagnostics=diagnostics,
                    )
            except Exception as exc:
                diagnostics.append(f"LLM proposal failed during parse/validation: {exc}")

        # 2. Deterministic offset grid search fallback validated against Golden Anchors
        for cand in self._generate_candidate_overrides(source_id):
            try:
                df_cand = collector_fn(cand)
                val_res = validate_series_dataframe(source_id, df_cand, self.golden_config)
                if val_res.passed:
                    diagnostics.append(f"Deterministic Golden Test PASSED 100% with offset {cand}.")
                    return df_cand, HealerResult(
                        source_id=source_id,
                        healed=True,
                        mode_used="deterministic_grid_search",
                        proposed_override=cand,
                        validation_result=val_res,
                        diagnostics=diagnostics,
                    )
            except Exception:
                continue

        diagnostics.append(f"No candidate offset passed the Golden Test for '{source_id}'. Aborting patch.")
        return None, HealerResult(
            source_id=source_id,
            healed=False,
            mode_used="failed",
            proposed_override={},
            validation_result=None,
            diagnostics=diagnostics,
        )
