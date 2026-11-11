"""Shared utilities for deterministic collectors (HTTP download, regex auto-discovery, SHA-256 hashing, and snapshots)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import urljoin

import httpx
import pandas as pd
import yaml
from bs4 import BeautifulSoup


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def load_sources_config(config_path: Union[str, Path] = "config/sources.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def update_raw_cache_manifest(
    raw_cache_dir: Union[str, Path],
    filename: str,
    sha256_hash: str,
    source_url: str,
) -> None:
    cache_dir = Path(raw_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    manifest[filename] = {
        "sha256": sha256_hash,
        "source_url": source_url,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_url_from_landing_page(
    landing_page_url: str,
    discovery_regex: str,
    timeout: float = 20.0,
) -> Optional[str]:
    """Fetches the official landing page and extracts the file link via regex and BeautifulSoup."""
    with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout) as client:
        resp = client.get(landing_page_url)
        resp.raise_for_status()
        html = resp.text

    pattern = re.compile(discovery_regex, re.IGNORECASE)
    # 1. Search directly in full HTML text
    match = pattern.search(html)
    if match:
        return match.group(0)

    # 2. Search across <a> href attributes resolving relative URLs
    soup = BeautifulSoup(html, "html.parser")
    for a_tag in soup.find_all("a", href=True):
        raw_href = a_tag["href"].strip()
        if raw_href.startswith("../") and "/ftp/" in raw_href:
            idx_ftp = raw_href.find("/ftp/")
            full_url = urljoin(landing_page_url, raw_href[idx_ftp:])
        else:
            full_url = urljoin(landing_page_url, raw_href)
        if pattern.search(full_url):
            return full_url

    return None


def _is_valid_downloaded_payload(content: bytes, filename: str) -> bool:
    """Ensures a downloaded binary file (.xls, .xlsx, .zip) is not an HTML error page."""
    if not content:
        return False
    head = content[:256].lstrip().lower()
    if filename.endswith((".xls", ".xlsx", ".zip")) and (
        head.startswith(b"<!doctype html") or head.startswith(b"<html")
    ):
        return False
    return True


def resolve_raw_file(
    source_cfg: Dict[str, Any],
    raw_cache_dir: Union[str, Path] = "data/raw_cache",
    bootstrap_raw_dir: Union[str, Path] = "bootstrap/raw_data",
    prefer_network: bool = False,
    timeout: float = 25.0,
) -> Path:
    """Resolves the raw data file:
    - When `prefer_network=True`, attempts auto-discovery + direct HTTP download and updates `data/raw_cache/`.
    - When offline or `prefer_network=False`, uses `data/raw_cache/` or initializes from `bootstrap/raw_data/`.
    """
    cache_dir = Path(raw_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_filename = source_cfg["local_raw_filename"]
    cached_path = cache_dir / local_filename
    bootstrap_path = Path(bootstrap_raw_dir) / local_filename

    if not cached_path.exists() and bootstrap_path.exists():
        shutil.copy2(bootstrap_path, cached_path)
        sha = compute_sha256(cached_path)
        update_raw_cache_manifest(
            cache_dir,
            local_filename,
            sha,
            source_cfg.get("direct_url", "bootstrap/raw_data"),
        )

    if prefer_network or not cached_path.exists():
        candidate_urls: list[str] = []
        landing_url = source_cfg.get("landing_page_url")
        regex_str = source_cfg.get("discovery_regex")
        if landing_url and regex_str:
            try:
                discovered = discover_url_from_landing_page(landing_url, regex_str, timeout=timeout)
                if discovered:
                    candidate_urls.append(discovered)
            except Exception:
                pass

        direct_url = source_cfg.get("direct_url")
        if direct_url and direct_url not in candidate_urls:
            candidate_urls.append(direct_url)

        for target_url in candidate_urls:
            try:
                with httpx.Client(headers=DEFAULT_HEADERS, follow_redirects=True, timeout=timeout) as client:
                    resp = client.get(target_url)
                    resp.raise_for_status()
                    if _is_valid_downloaded_payload(resp.content, local_filename):
                        cached_path.write_bytes(resp.content)
                        sha = compute_sha256(cached_path)
                        update_raw_cache_manifest(cache_dir, local_filename, sha, target_url)
                        return cached_path
            except Exception:
                continue

    if cached_path.exists():
        return cached_path

    raise FileNotFoundError(f"Raw file for '{source_cfg.get('id')}' not found at {cached_path}")


def save_last_known_good(
    df: pd.DataFrame,
    snapshot_filename: str,
    lkg_dir: Union[str, Path] = "data/last_known_good",
) -> Path:
    out_dir = Path(lkg_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / snapshot_filename
    df_out = df.copy()
    df_out.index.name = "period"
    df_out.to_csv(out_path, encoding="utf-8")
    return out_path


def load_last_known_good(
    snapshot_filename: str,
    index_type: str = "datetime_ms",
    lkg_dir: Union[str, Path] = "data/last_known_good",
) -> pd.DataFrame:
    path = Path(lkg_dir) / snapshot_filename
    if not path.exists():
        raise FileNotFoundError(f"Snapshot last_known_good does not exist: {path}")
    df = pd.read_csv(path, index_col="period")
    if index_type == "year_int":
        df.index = df.index.astype(int)
    elif index_type == "datetime_ms":
        df.index = pd.to_datetime(df.index)
        df.index.freq = "MS"
    return df
