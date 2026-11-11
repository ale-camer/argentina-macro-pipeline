# `argentina-macro-pipeline`

Deterministic data engineering, quantitative macroeconomic analysis, and publication-grade visualization engine for Argentina (`1961–Present`), designed to power the `#analisis` section of `personal_website`.

## Core Architectural Principle

> **"100% deterministic core; LLM/Agent usage strictly bounded and encapsulated to the two brittle edges of the workflow (self-healing ingestion upon official layout changes, and multimodal visual QA / regime-shift PR drafting upon statistically detected structural breaks), with mandatory always-on telemetry and alerts on every execution."**

---

## 1. Data Lifecycle

The pipeline runs automatically on the **25th of each month at 09:00 UTC** via GitHub Actions (`.github/workflows/monthly_update.yml`) or on-demand via `workflow_dispatch`, executing 4 strict stages:

1. **Stage 1 — Deterministic Ingestion, Golden Tests & Self-Healing (`src/collectors/`, `src/validators/`, `src/agents/ingestion_healer_agent.py`):**
   - Collectors download official raw files from the **7 primary sources** (World Bank, Ministry of Economy, Ministry of Human Capital / SIPA, and INDEC) and record their cryptographic `SHA-256` hash in `data/raw_cache/manifest.json`.
   - The validator (`schema_and_golden.py`) verifies numeric types, absence of `NaN` values in closed periods, strict temporal continuity (`YS` / `MS`), and exact match (within `0.01`) against immutable historical **Golden Anchors** (`config/golden_anchors.json`).
   - Verified series update the clean snapshots in `data/last_known_good/*.csv`.
   - **Agent 1 Boundary (`SelfHealingIngestionAgent`):** Invoked **only** if a download or Golden Test fails (e.g., an agency inserts extra header rows or shifts columns in Excel). It proposes updated parser offsets and **only accepts the patch if it passes 100% of the deterministic Golden Tests**; otherwise, the patch is discarded and `data/last_known_good/` is used as a safe production fallback.
2. **Stage 2 — Mathematical Transformations & Statistical Models (`src/transformers/`, `src/models/`):**
   - **Exact Geometric Compounding:** Chained Real GDP Index (`Base 1980 = 100`), cumulative relative growth by SIPA labor modality ($\frac{X_t}{X_0} - 1$), and intercensal demographic gap (`2010 Census` vs. `2022 Census`).
   - **Seasonal Adjustment & Employment-Output Elasticity:** Monthly alignment between SIPA Private Formal Employment and EMAE (`Base Jan-2009 = 100`), showing correlation rising from `51.93%` (raw seasonal series) to `76.72%` (seasonally adjusted) and `82.30%` (trend-cycle).
   - **Deduplicated Wage Splicing (`2015-10-01`):** Continuous geometric chaining between the historical (`Base 2012`) and current (`Base Oct-2016=100`) INDEC wage series, deflated via the exact Fisher equation ($\frac{1 + w_t}{1 + \pi_t} - 1$) into `Base 2012 = 100`.
   - **Structural Breaks & Forecasting:** Deterministic change-point detection via `ruptures` (`PELT`) + 24-month rolling log-linear slopes (`structural_breaks.py`) and 6-month `SARIMAX` projections with 95% confidence intervals (`forecaster.py`).
3. **Stage 3 — Collision-Free Visualizations & Agentic QA (`src/visualizers/charts.py`, `src/agents/visual_and_regime_agent.py`):**
   - Generates all 5 figures in high resolution (`PNG` + `SVG`) under `outputs/figures/` using dynamically reserved upper header bands and external legends (`bbox_to_anchor`).
   - **Agent 2 Boundary (`VisualQAAndRegimeAgent`):** Audits all 5 generated charts (`5/5 PASS`) and, **only** if `structural_breaks.py` flags `regime_shift_detected = True`, drafts a human-review Pull Request proposal in `outputs/regime_shift_pr_proposal.md`.
4. **Stage 4 — Web Contract & Always-On Monitoring (`outputs/`, `src/monitoring/notifier.py`):**
   - Exports `outputs/macro_argentina_latest.json` (`~26 KB`) and `outputs/run_report_latest.json`, dispatching alerts via WhatsApp (TextMeBot / CallMeBot), GitHub Step Summary, Telegram Bot API, and/or SMTP Email.

---

## 2. Repository Structure

```text
argentina-macro-pipeline/
├── pyproject.toml                       # Project dependencies, CLI scripts, and pytest configuration
├── .env.example                         # Environment variable template (network, Gemini, alerts)
├── README.md                            # Technical documentation and web integration guide
├── config/
│   ├── sources.yaml                     # Canonical URLs, auto-discovery landing pages, and parser offsets
│   ├── golden_anchors.json              # Immutable historical benchmarks ("Golden Tests")
│   └── regimes.yaml                     # Macroeconomic regime definitions and PELT break thresholds
├── data/
│   ├── last_known_good/                 # Clean CSV snapshots (tracked in Git as safe fallback)
│   └── raw_cache/                       # Downloaded raw files + SHA-256 manifest.json
├── src/
│   ├── main.py                          # Main 4-stage monthly pipeline orchestrator
│   ├── collectors/                      # 100% Deterministic: HTTP download + regex auto-discovery
│   ├── validators/                      # 100% Deterministic: Pydantic schemas + Golden Anchor checks
│   ├── transformers/                    # 100% Deterministic: geometric compounding, splicing, elasticities
│   ├── models/                          # 100% Deterministic: PELT structural breaks + SARIMAX forecaster
│   ├── visualizers/                     # 100% Deterministic: 5 zero-overlap figures (PNG + SVG)
│   ├── agents/                          # Strictly encapsulated agentic layer (with deterministic fallback)
│   └── monitoring/                      # Structured telemetry and Always-On alerts (WhatsApp / Telegram / SMTP / GitHub)
├── outputs/
│   ├── macro_argentina_latest.json      # Compact JSON contract (~26 KB) ready for personal_website
│   ├── run_report_latest.json           # Latest execution telemetry (per-source status, Golden Tests, QA)
│   └── figures/                         # 5 publication-grade charts in high-res PNG and vector SVG
├── tests/
│   ├── test_golden_anchors.py           # Verifies parsers reproduce historical benchmarks to 0.01 precision
│   └── test_transformers.py             # Unit tests for geometric compounding, wage splicing, and elasticity
└── .github/
    └── workflows/
        └── monthly_update.yml           # Monthly cron (25th) + automated snapshot & output commits
```

---

## 3. Local Setup & Execution

```bash
# 1. Create virtual environment and install in editable mode
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2. Run the deterministic test suite (Golden Anchors + Transformers)
.venv/bin/pytest -v

# 3. Run the pipeline using verified local cache
.venv/bin/python3 -m src.main

# 4. Run the pipeline downloading live from the 7 official sources (.gob.ar and World Bank)
PREFER_NETWORK=true .venv/bin/python3 -m src.main --network
```

---

## 4. Direct Consumption from `personal_website` (CORS-Free via GitHub Raw)

Because `.github/workflows/monthly_update.yml` commits updated `outputs/macro_argentina_latest.json` and `outputs/figures/*` directly to the `main` branch, the React frontend in `personal_website` can fetch both the JSON data contract and vector figures directly from `raw.githubusercontent.com` (which provides native `Access-Control-Allow-Origin: *` headers and global CDN caching):

### Production Static Endpoints

- **Macro Data & Metrics JSON Contract (`~26 KB`):**
  ```text
  https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/macro_argentina_latest.json
  ```
- **Latest Run Telemetry Report:**
  ```text
  https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/run_report_latest.json
  ```
- **High-Resolution Figures (`SVG` vector recommended for web / `PNG`):**
  - Visual 1 (GDP, Public Spending & Inflation 1980–Present): `https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/figures/macro_improved_2026_1.svg`
  - Visual 2 (Private Formal Jobs vs. EMAE & Elasticity): `https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/figures/macro_improved_2026_2.svg`
  - Visual 3 (Cumulative Geometric Growth by SIPA Modality): `https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/figures/macro_improved_2026_3.svg`
  - Visual 4 (Modality Share & Net Structural Shift `pp`): `https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/figures/macro_improved_2026_4_table.svg`
  - Visual 5 (Real Private Wages vs. Real GDP Base 2012=100): `https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs/figures/macro_improved_2026_4.svg`

### React Integration Snippet (`personal_website`)

```tsx
const PIPELINE_BASE_URL =
  "https://raw.githubusercontent.com/ale-camer/argentina-macro-pipeline/main/outputs";

export async function fetchMacroArgentinaData() {
  const res = await fetch(`${PIPELINE_BASE_URL}/macro_argentina_latest.json`, {
    cache: "no-cache",
  });
  if (!res.ok) {
    throw new Error(`Failed to load macro contract: ${res.status}`);
  }
  const data = await res.json();
  return {
    ...data,
    figureUrls: Object.fromEntries(
      Object.entries(data.figures).map(([key, files]: [string, any]) => [
        key,
        {
          svg: `${PIPELINE_BASE_URL}/figures/${files.svg}`,
          png: `${PIPELINE_BASE_URL}/figures/${files.png}`,
        },
      ])
    ),
  };
}
```
