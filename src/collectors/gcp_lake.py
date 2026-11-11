"""Medallion Data Lake synchronization for Google Cloud Storage (Bronze/Silver/Gold) and BigQuery."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


DEFAULT_GCP_PROJECT = "argentina-macro-pipeline"
DEFAULT_GCS_BUCKET = "gs://argentina-macro-lake-31037835126"
DEFAULT_BQ_DATASET = "macro_lake"
DEFAULT_LOCATION = "us-central1"


def sync_to_gcp_lake(
    project_id: Optional[str] = None,
    bucket_uri: Optional[str] = None,
    bq_dataset: Optional[str] = None,
    raw_cache_dir: str = "data/raw_cache",
    lkg_dir: str = "data/last_known_good",
    outputs_dir: str = "outputs",
) -> Dict[str, str]:
    """Synchronizes local Bronze, Silver, and Gold layers to the private GCS bucket and BigQuery dataset in `us-central1`."""
    proj = project_id or os.environ.get("GCP_PROJECT_ID", DEFAULT_GCP_PROJECT)
    bucket = (bucket_uri or os.environ.get("GCS_LAKE_BUCKET", DEFAULT_GCS_BUCKET)).rstrip("/")
    dataset = bq_dataset or os.environ.get("BQ_DATASET", DEFAULT_BQ_DATASET)
    partition_ym = datetime.now(timezone.utc).strftime("%Y-%m")

    bronze_uri = f"{bucket}/bronze/{partition_ym}/"
    silver_uri = f"{bucket}/silver/latest/"
    gold_uri = f"{bucket}/gold/latest/"

    subprocess.run(
        ["gcloud", "storage", "cp", "-r", f"{raw_cache_dir}/*", bronze_uri, f"--project={proj}", "--quiet"],
        check=True,
    )
    subprocess.run(
        ["gcloud", "storage", "cp", f"{lkg_dir}/*.csv", silver_uri, f"--project={proj}", "--quiet"],
        check=True,
    )
    subprocess.run(
        [
            "gcloud", "storage", "cp",
            f"{outputs_dir}/macro_argentina_latest.json",
            f"{outputs_dir}/run_report_latest.json",
            gold_uri,
            f"--project={proj}",
            "--quiet",
        ],
        check=True,
    )

    for csv_path in sorted(Path(lkg_dir).glob("*.csv")):
        table_name = f"silver_{csv_path.stem}"
        subprocess.run(
            [
                "bq",
                f"--location={DEFAULT_LOCATION}",
                f"--project_id={proj}",
                "load",
                "--replace",
                "--autodetect",
                "--source_format=CSV",
                f"{dataset}.{table_name}",
                str(csv_path),
            ],
            check=True,
        )

    return {
        "gcp_project": proj,
        "bronze_gcs_uri": bronze_uri,
        "silver_gcs_uri": silver_uri,
        "gold_gcs_uri": gold_uri,
        "bigquery_dataset": f"{proj}:{dataset}",
    }

