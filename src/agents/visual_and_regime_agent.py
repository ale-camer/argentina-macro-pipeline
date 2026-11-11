"""Agent 2 (`VisualQAAndRegimeAgent`):
Strictly encapsulated Regime Shift Evaluator & PR Assistant:
Evaluates PELT structural break telemetry (`structural_breaks.py`) over the transformed JSON contract series.
ONLY when `break_report.regime_shift_detected = True`, drafts a proposed update for `config/regimes.yaml`
and a Pull Request body (`outputs/regime_shift_pr_proposal.md`) for human review, using Gemini (`gemini-3.8-flash`)
with a 100% deterministic fallback when `GEMINI_API_KEY` is not set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel

from src.models.structural_breaks import StructuralBreakReport


class RegimePRProposal(BaseModel):
    pr_opened: bool
    mode_used: str = "no_break_detected"  # "no_break_detected" | "gemini_llm" | "deterministic_fallback"
    pr_title: Optional[str] = None
    pr_markdown_path: Optional[str] = None
    proposed_regime_yaml_snippet: Optional[str] = None


class VisualQAAndRegimeAgent:
    """Encapsulated agent focused exclusively on evaluating PELT structural breaks and drafting regime-shift PR proposals."""

    def __init__(self, model_name: str = "gemini-3.8-flash") -> None:
        self.model_name = os.environ.get("GEMINI_MODEL", model_name)
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def draft_regime_shift_pr_if_needed(
        self,
        break_report: StructuralBreakReport,
        output_dir: Union[str, Path] = "outputs",
    ) -> RegimePRProposal:
        """When `break_report.regime_shift_detected` is True, writes a Markdown PR proposal for human review."""
        if not break_report.regime_shift_detected:
            return RegimePRProposal(pr_opened=False, mode_used="no_break_detected")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pr_path = out_dir / "regime_shift_pr_proposal.md"

        yaml_snippet = (
            "  - id: \"new_post_adjustment_regime\"\n"
            f"    name: \"Post-{break_report.evaluated_up_to_date[:4]}: Newly Detected Regime\"\n"
            f"    start_year: {break_report.evaluated_up_to_date[:4]}\n"
            f"    # 24m EMAE slope: {break_report.latest_24m_emae_cagr_pct:+.2f}%/yr | "
            f"Jobs slope: {break_report.latest_24m_jobs_cagr_pct:+.2f}%/yr\n"
        )

        mode_used = "deterministic_fallback"
        llm_narrative = ""
        if self.api_key:
            synthesized = self._synthesize_break_with_gemini(break_report)
            if synthesized:
                mode_used = "gemini_llm"
                if synthesized.get("yaml_snippet"):
                    yaml_snippet = str(synthesized["yaml_snippet"]).rstrip() + "\n"
                if synthesized.get("executive_summary"):
                    llm_narrative = (
                        "\n## Agentic Regime Synthesis (`gemini-3.8-flash`)\n"
                        f"{synthesized['executive_summary']}\n"
                    )

        pr_title = f"[Regime Shift] Structural break detected as of {break_report.evaluated_up_to_date}"
        pr_body = (
            f"# {pr_title}\n\n"
            "## Statistical Detector Summary (`structural_breaks.py`)\n"
            f"- **Evaluation cutoff date:** `{break_report.evaluated_up_to_date}`\n"
            f"- **24-month annualized slope (Seasonally Adjusted EMAE):** `{break_report.latest_24m_emae_cagr_pct:+.2f}%/yr`\n"
            f"- **24-month annualized slope (SIPA Private Employment):** `{break_report.latest_24m_jobs_cagr_pct:+.2f}%/yr`\n"
            f"- **Latest Public Spending / GDP:** `{break_report.latest_pub_exp_gdp_pct:.2f}%` "
            f"(Below 40% critical threshold: `{break_report.spending_below_critical_40pct}`)\n"
            f"- **PELT EMAE break dates:** `{break_report.pelt_emae_breaks}`\n"
            f"- **PELT Private Employment break dates:** `{break_report.pelt_jobs_breaks}`\n\n"
            "## Trigger Reasons\n"
            + "\n".join(f"- {r}" for r in break_report.trigger_reasons)
            + "\n"
            + llm_narrative
            + "\n## Proposed Snippet for `config/regimes.yaml` (Requires Human Approval)\n"
            f"```yaml\n{yaml_snippet}```\n"
        )
        pr_path.write_text(pr_body, encoding="utf-8")
        return RegimePRProposal(
            pr_opened=True,
            mode_used=mode_used,
            pr_title=pr_title,
            pr_markdown_path=str(pr_path),
            proposed_regime_yaml_snippet=yaml_snippet,
        )

    def _synthesize_break_with_gemini(self, break_report: StructuralBreakReport) -> Optional[dict]:
        """Requests a concise quantitative interpretation and YAML snippet from Gemini upon PELT break detection."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            prompt = (
                "You are a quantitative macroeconomic regime analyst for Argentina.\n"
                "A deterministic PELT structural break detector flagged a regime shift with the following telemetry:\n"
                f"{json.dumps(break_report.model_dump(), indent=2)}\n\n"
                "Return a JSON object with keys:\n"
                '- "executive_summary": a concise 2-sentence quantitative summary of the break.\n'
                '- "yaml_snippet": a YAML list item (`  - id: ...`, `    name: ...`, `    start_year: ...`) for `config/regimes.yaml`.\n'
            )
            resp = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            if resp.text:
                data = json.loads(resp.text)
                if isinstance(data, dict):
                    return data
        except Exception:
            return None
        return None
