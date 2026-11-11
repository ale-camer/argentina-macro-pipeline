"""Always-On monitoring and alerting system (`notifier.py`):
Generates `outputs/run_report_latest.json` and dispatches structured notifications (WhatsApp via TextMeBot/CallMeBot, Telegram, SMTP Email, GitHub Actions Summary)
on every pipeline execution (success or failure).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from src.agents.visual_and_regime_agent import RegimePRProposal, VisualQAReport
from src.models.structural_breaks import StructuralBreakReport

logger = logging.getLogger(__name__)


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
    duration_sec: float = 0.0
    status: PipelineStatus
    sources_summary: List[SourceTelemetryRow] = Field(default_factory=list)
    visual_qa_summary: str
    visual_qa_report: VisualQAReport
    structural_breaks: StructuralBreakReport
    regime_pr_proposal: RegimePRProposal
    outputs_generated: Dict[str, str] = Field(default_factory=dict)
    notifications_sent: List[str] = Field(default_factory=list)


def _detect_trigger_label() -> str:
    ev = os.getenv("GITHUB_EVENT_NAME", "").strip().lower()
    if ev == "schedule":
        return "Cron"
    if ev:
        return "Manual (GHA)"
    return "Manual"


def format_markdown_summary(report: PipelineRunReport) -> str:
    """Formats a Markdown summary table covering all sources, Golden Tests, and Visual QA."""
    lines = [
        f"## Argentina Macro Pipeline — Run Report (`{report.status}`)",
        f"- **Timestamp (UTC):** `{report.run_timestamp_utc}` (`{report.duration_sec:.1f}s`)",
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


def format_whatsapp_summary(report: PipelineRunReport) -> str:
    """Formats a 2-tier (Executive Header + Diagnostic Detail) WhatsApp alert with *[ARG-MACRO]*."""
    try:
        dt = datetime.fromisoformat(report.run_timestamp_utc.replace("Z", "+00:00"))
        ts_utc = dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    trigger = _detect_trigger_label()
    sources_ok = sum(
        1
        for r in report.sources_summary
        if r.golden_test_passed and r.ingestion_mode in ("deterministic", "healed_by_agent")
    )
    sources_total = len(report.sources_summary)
    total_rows = sum(r.row_count for r in report.sources_summary)
    delta_rows = sum(r.new_observations_added for r in report.sources_summary)
    fig_qa = f"{report.visual_qa_report.figures_passed}/{report.visual_qa_report.figures_checked}"

    years = [
        int(r.current_last_period[:4])
        for r in report.sources_summary
        if r.current_last_period and len(r.current_last_period) >= 4 and r.current_last_period[:4].isdigit()
    ]
    max_year = max(years) if years else 2026
    year_range = f"1961–{max_year}"

    contract_path = Path(report.outputs_generated.get("macro_contract_json", "outputs/macro_argentina_latest.json"))
    contract_kb = (contract_path.stat().st_size / 1024.0) if contract_path.exists() else 26.0

    success = report.status in ("OK_DETERMINISTIC", "HEALED_BY_AGENT", "REGIME_SHIFT_PR_OPENED")
    if success:
        sign = "+" if delta_rows >= 0 else ""
        quiebre_str = "SÍ" if report.structural_breaks.regime_shift_detected else "NO"
        return (
            "🟢 *[ARG-MACRO]* · Ingesta Mensual OK\n"
            f"_{ts_utc} · {report.duration_sec:.1f}s · {trigger}_\n\n"
            "*Detalle de control:*\n"
            f"> Fuentes: `{sources_ok}/{sources_total}` · Figuras QA: `{fig_qa}`\n"
            f"> Filas: `{total_rows:,}` (`{sign}{delta_rows:,}` vs base) · Quiebre: `{quiebre_str}`\n"
            f"> Rango: `{year_range}` · Contrato: `{contract_kb:.1f} KB`"
        )

    failed_sources = [
        f"{r.source_id} ({r.ingestion_mode})"
        for r in report.sources_summary
        if not r.golden_test_passed or r.ingestion_mode not in ("deterministic", "healed_by_agent")
    ]
    error_detail = (
        f"Fuentes afectadas: {', '.join(failed_sources)}"
        if failed_sources
        else "Fallo en validación visual o de esquema"
    )
    return (
        "🔴 *[ARG-MACRO]* · Fallo en Ingesta\n"
        f"_{ts_utc} · {report.duration_sec:.1f}s · {trigger}_\n\n"
        "*Diagnóstico:*\n"
        f"> Fuentes OK: `{sources_ok}/{sources_total}` · Figuras QA: `{fig_qa}`\n"
        f"> Error: {error_detail}\n"
        "> Snapshot previo preservado sin cambios"
    )


def send_whatsapp_notification(message: str) -> bool:
    """
    Send `message` via TextMeBot or CallMeBot using the standardized credentials:
      - WHATSAPP_PHONE: WhatsApp Group JID (`120363...@g.us`) OR phone number (`+4177...`)
      - WHATSAPP_APIKEY: TextMeBot alphanumeric key or CallMeBot numeric key
    Returns True if sent, False if skipped or failed.
    """
    recipient = os.getenv("WHATSAPP_PHONE", "").strip()
    if not recipient:
        logger.info("WHATSAPP_PHONE not configured; skipping WhatsApp dispatch.")
        return False

    api_key = (
        os.getenv("WHATSAPP_APIKEY", "").strip()
        or os.getenv("WHATSAPP_CALLMEBOT_APIKEY", "").strip()
    )
    if not api_key:
        logger.warning("WHATSAPP_PHONE is set, but WHATSAPP_APIKEY was not found.")
        return False

    encoded_msg = urllib.parse.quote(message)

    if not api_key.isdigit():
        if "@g.us" in recipient:
            formatted_recipient = recipient
        else:
            clean_phone = recipient.lstrip("+").replace(" ", "").replace("-", "")
            formatted_recipient = urllib.parse.quote(f"+{clean_phone}")
        url = (
            f"https://api.textmebot.com/send.php?"
            f"recipient={formatted_recipient}&apikey={urllib.parse.quote(api_key)}"
            f"&text={encoded_msg}&json=yes"
        )
        provider_name = "TextMeBot"
    else:
        clean_phone = recipient.lstrip("+").replace(" ", "").replace("-", "")
        url = (
            f"https://api.callmebot.com/whatsapp.php?"
            f"phone={clean_phone}&text={encoded_msg}&apikey={api_key}"
        )
        provider_name = "CallMeBot"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="ignore").strip()
            if provider_name == "TextMeBot":
                try:
                    ok = json.loads(body).get("status") == "success"
                except Exception:
                    ok = "Success" in body or "success" in body
            else:
                ok = 200 <= resp.status < 300
            logger.info("WhatsApp notification via %s: HTTP %s (ok=%s)", provider_name, resp.status, ok)
            return ok
    except Exception as exc:
        logger.error("Failed to send WhatsApp via %s: %s", provider_name, exc)
        return False


# Backward-compatible alias
send_whatsapp_callmebot_alert = send_whatsapp_notification


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
    smtp_port = int(os.environ.get("SMTP_PORT") or "587")
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
    if send_whatsapp_notification(format_whatsapp_summary(report)):
        channels.append("whatsapp")
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
