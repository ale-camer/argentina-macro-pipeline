"""Always-On monitoring and alerting system (`notifier.py`):
Generates `outputs/run_report_latest.json` and dispatches structured notifications (Telegram / SMTP Email / GitHub Actions Summary)
on every pipeline execution (success or failure).
"""

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from src.agents.visual_and_regime_agent import RegimePRProposal, VisualQAReport
from src.models.structural_breaks import StructuralBreakReport


PipelineStatus = Literal[
    "OK_DETERMINISTIC",
    "HEALED_BY_AGENT",
    "REGIME_SHIFT_PR_OPENED",
    "FALLBACK_USED",
    "FAILED",
]


class SourceTelemetryRow(BaseModel):
    source_id: str
    previous_last_period: Optional[str] = None
    current_last_period: Optional[str] = None
    row_count: int = 0
    new_observations_added: int = 0
    golden_test_passed: bool = False
    ingestion_mode: str = "deterministic"  # "deterministic" | "healed_by_agent" | "fallback_last_known_good" | "failed"
    notes: Optional[str] = None


class PipelineRunReport(BaseModel):
    run_timestamp_utc: str
    status: PipelineStatus
    sources_summary: List[SourceTelemetryRow] = Field(default_factory=list)
    visual_qa_summary: str
    visual_qa_report: VisualQAReport
    structural_breaks: StructuralBreakReport
    regime_pr_proposal: RegimePRProposal
    outputs_generated: Dict[str, str] = Field(default_factory=dict)
    notifications_sent: List[str] = Field(default_factory=list)


def format_markdown_summary(report: PipelineRunReport) -> str:
    """Formats a Markdown summary table covering all sources, Golden Tests, and Visual QA."""
    lines = [
        f"## Argentina Macro Pipeline — Run Report (`{report.status}`)",
        f"- **Timestamp (UTC):** `{report.run_timestamp_utc}`",
        f"- **Visual QA:** `{report.visual_qa_summary}` (`{report.visual_qa_report.mode_used}`)",
        f"- **Regime Shift Detected:** `{report.structural_breaks.regime_shift_detected}`",
        "",
        "| Source | Previous Cutoff | Current Cutoff | Rows | New Obs. | Golden Test | Ingestion Mode |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in report.sources_summary:
        gt_badge = "PASS" if row.golden_test_passed else "FAIL"
        lines.append(
            f"| `{row.source_id}` | {row.previous_last_period or 'N/A'} | "
            f"{row.current_last_period or 'N/A'} | {row.row_count} | "
            f"+{row.new_observations_added} | **{gt_badge}** | `{row.ingestion_mode}` |"
        )
    return "\n".join(lines)


def send_telegram_alert(markdown_text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"chat_id": chat_id, "text": markdown_text})
            resp.raise_for_status()
        return True
    except Exception:
        return False


def send_email_alert(subject: str, body: str) -> bool:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    email_to = os.environ.get("ALERT_EMAIL_TO")
    if not (smtp_host and smtp_user and smtp_pass and email_to):
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = email_to
        msg.set_content(body)
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True
    except Exception:
        return False


def write_github_step_summary(markdown_text: str) -> bool:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return False
    try:
        with Path(summary_path).open("a", encoding="utf-8") as f:
            f.write(markdown_text + "\n")
        return True
    except Exception:
        return False


def publish_run_report(
    report: PipelineRunReport,
    report_path: Union[str, Path] = "outputs/run_report_latest.json",
) -> Path:
    """Writes `outputs/run_report_latest.json` and triggers all configured notification channels."""
    md_summary = format_markdown_summary(report)
    channels: List[str] = ["local_json_report"]

    if write_github_step_summary(md_summary):
        channels.append("github_step_summary")
    if send_telegram_alert(md_summary):
        channels.append("telegram")
    if send_email_alert(f"[Argentina Macro Pipeline] Status: {report.status}", md_summary):
        channels.append("email_smtp")

    report.notifications_sent = channels
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
