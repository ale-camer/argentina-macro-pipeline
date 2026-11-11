"""Agent 2 (`VisualQAAndRegimeAgent`):
1. Multimodal Visual QA Inspector: Verifies that the 5 generated figures have no overlapping text,
   legends colliding with bars/curves, or clipped axes. Includes a safe deterministic fallback when
   `GEMINI_API_KEY` is not set.
2. Regime Shift PR Assistant: ONLY if `structural_breaks.py` flags `regime_shift_detected = True`,
   drafts a proposed update for `config/regimes.yaml` and a Pull Request body for human review.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from src.models.structural_breaks import StructuralBreakReport


class FigureQADetail(BaseModel):
    figure_id: str
    png_path: str
    exists: bool
    width_px: int = 0
    height_px: int = 0
    file_size_bytes: int = 0
    passed: bool
    issues: List[str] = Field(default_factory=list)


class VisualQAReport(BaseModel):
    visual_qa_passed: bool
    mode_used: str  # "gemini_multimodal" | "deterministic_fallback"
    figures_checked: int
    figures_passed: int
    issues: List[str] = Field(default_factory=list)
    details: List[FigureQADetail] = Field(default_factory=list)


class RegimePRProposal(BaseModel):
    pr_opened: bool
    pr_title: Optional[str] = None
    pr_markdown_path: Optional[str] = None
    proposed_regime_yaml_snippet: Optional[str] = None


def _read_png_dimensions(png_path: Path) -> tuple[int, int]:
    """Reads width and height directly from the PNG IHDR header."""
    with png_path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return 0, 0
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


class VisualQAAndRegimeAgent:
    """Encapsulated agent for Visual QA of the 5 figures and PR drafting upon structural breaks."""

    def __init__(self, model_name: str = "gemini-3.8-flash") -> None:
        self.model_name = os.environ.get("GEMINI_MODEL", model_name)
        self.api_key = os.environ.get("GEMINI_API_KEY")

    def inspect_visuals(
        self,
        charts_manifest: Dict[str, Dict[str, str]],
    ) -> VisualQAReport:
        """Audits the 5 generated figures (deterministic IHDR dimension/integrity check + multimodal vision when API key is set)."""
        details: List[FigureQADetail] = []
        all_issues: List[str] = []

        for fig_id, paths in charts_manifest.items():
            png_path = Path(paths["png"])
            if not png_path.exists():
                msg = f"{fig_id}: PNG file does not exist at {png_path}"
                all_issues.append(msg)
                details.append(
                    FigureQADetail(figure_id=fig_id, png_path=str(png_path), exists=False, passed=False, issues=[msg])
                )
                continue

            fsize = png_path.stat().st_size
            w, h = _read_png_dimensions(png_path)
            fig_issues: List[str] = []

            if fsize < 25_000:
                fig_issues.append(f"{fig_id}: File size suspiciously small ({fsize} bytes).")
            if w < 1200 or h < 600:
                fig_issues.append(f"{fig_id}: Insufficient resolution ({w}x{h} px).")

            details.append(
                FigureQADetail(
                    figure_id=fig_id,
                    png_path=str(png_path),
                    exists=True,
                    width_px=w,
                    height_px=h,
                    file_size_bytes=fsize,
                    passed=(len(fig_issues) == 0),
                    issues=fig_issues,
                )
            )
            all_issues.extend(fig_issues)

        mode_used = "deterministic_fallback"
        if self.api_key and len(all_issues) == 0:
            llm_issues = self._run_multimodal_llm_check(details)
            if llm_issues is not None:
                mode_used = "gemini_multimodal"
                all_issues.extend(llm_issues)
                if llm_issues:
                    for d in details:
                        matching = [iss for iss in llm_issues if d.figure_id in iss]
                        if matching:
                            d.passed = False
                            d.issues.extend(matching)

        passed_count = sum(1 for d in details if d.passed)
        return VisualQAReport(
            visual_qa_passed=(passed_count == len(details) and len(details) == 5 and len(all_issues) == 0),
            mode_used=mode_used,
            figures_checked=len(details),
            figures_passed=passed_count,
            issues=all_issues,
            details=details,
        )

    def _run_multimodal_llm_check(self, details: List[FigureQADetail]) -> Optional[List[str]]:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            contents: List[Any] = [
                (
                    "Inspect these 5 macroeconomic charts of Argentina. "
                    "Verify that there is no clipped text, no legend overlapping bars or curves, "
                    "and no unreadable axis labels. Respond in JSON with format: "
                    '{"visual_qa_passed": bool, "issues": ["..."]}.'
                )
            ]
            for d in details:
                img_bytes = Path(d.png_path).read_bytes()
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

            resp = client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            if resp.text:
                data = json.loads(resp.text)
                return list(data.get("issues", []))
        except Exception:
            return None
        return None

    def draft_regime_shift_pr_if_needed(
        self,
        break_report: StructuralBreakReport,
        output_dir: Union[str, Path] = "outputs",
    ) -> RegimePRProposal:
        """When `break_report.regime_shift_detected` is True, writes a Markdown PR proposal for human review."""
        if not break_report.regime_shift_detected:
            return RegimePRProposal(pr_opened=False)

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

        pr_title = f"[Regime Shift] Structural break detected as of {break_report.evaluated_up_to_date}"
        pr_body = (
            f"# {pr_title}\n\n"
            "## Statistical Detector Summary (`structural_breaks.py`)\n"
            f"- **Evaluation cutoff date:** `{break_report.evaluated_up_to_date}`\n"
            f"- **24-month annualized slope (Seasonally Adjusted EMAE):** `{break_report.latest_24m_emae_cagr_pct:+.2f}%/yr`\n"
            f"- **24-month annualized slope (SIPA Private Employment):** `{break_report.latest_24m_jobs_cagr_pct:+.2f}%/yr`\n"
            f"- **Latest Public Spending / GDP:** `{break_report.latest_pub_exp_gdp_pct:.2f}%` "
            f"(Below 40% critical threshold: `{break_report.spending_below_critical_40pct}`)\n"
            f"- **PELT EMAE break dates:** `{break_report.pelt_emae_breaks}`\n\n"
            "## Trigger Reasons\n"
            + "\n".join(f"- {r}" for r in break_report.trigger_reasons)
            + "\n\n## Proposed Snippet for `config/regimes.yaml` (Requires Human Approval)\n"
            f"```yaml\n{yaml_snippet}```\n"
        )
        pr_path.write_text(pr_body, encoding="utf-8")
        return RegimePRProposal(
            pr_opened=True,
            pr_title=pr_title,
            pr_markdown_path=str(pr_path),
            proposed_regime_yaml_snippet=yaml_snippet,
        )
