"""Generación 100% determinística de las 5 figuras con zonas reservadas anti-colisión (PNG + SVG)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.transformers.macro_employment import TransformedMacroBundle


SPANISH_MONTH_ABBR = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 120) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_visual_1(bundle: TransformedMacroBundle, out_dir: Path) -> Dict[str, str]:
    """Visual 1 (1980–Presente): Índice Geométrico Real del PBI (Base 1980=100), Gasto Público/PBI e Inflación Log."""
    df_v1 = bundle.df_v1
    max_yr = int(df_v1.index.max())

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 12.2), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.22},
    )
    fig.patch.set_facecolor("#ffffff")

    regimes = [
        (1980, 1991, "#fee2e2", "1980–1990: Hiperinflación\n• CAGR PBI: -1.15%/año\n• Gasto: 29.6% | Infl. med.: 316%"),
        (1991, 2011, "#dcfce7", "1991–2011: Expansión\n• CAGR PBI: +3.76%/año\n• Gasto: 31.3% | Infl. med.: 10.6%"),
        (2011, 2023.5, "#fef3c7", "2011–2023: Estanflación\n• CAGR PBI: -0.08%/año\n• Gasto: 44.1% | Infl. med.: 41.1%"),
        (2023.5, max_yr + 0.5, "#e0f2fe", f"2024–{max_yr}: Ajuste Fiscal\n• Gasto: 41.6% → 35.6%\n• Infl.: 208%→39% | PBI +4.4%"),
    ]

    for x0, x1, col, _ in regimes:
        ax_top.axvspan(x0, x1, color=col, alpha=0.35, zorder=0)
        ax_bot.axvspan(x0, x1, color=col, alpha=0.35, zorder=0)
        if x0 in [1991, 2011, 2023.5]:
            ax_top.axvline(x0, color="#334155", linestyle="--", linewidth=1.5, alpha=0.8)
            ax_bot.axvline(x0, color="#334155", linestyle="--", linewidth=1.5, alpha=0.8)

    ax_top.bar(
        df_v1.index,
        df_v1["gdp_index_1980"],
        color="#3b82f6",
        alpha=0.45,
        width=0.75,
        label="PBI Real Encadenado Geométrico (Base 1980 = 100)",
        zorder=2,
    )

    for (yr_start, yr_end), y_fit in bundle.v1_trends.items():
        lbl = "Tendencia Exponencial (CAGR por régimen)" if yr_start == 1980 else None
        ax_top.plot(y_fit.index, y_fit.values, color="#1e3a8a", linewidth=3.5, linestyle="--", label=lbl, zorder=4)

    # Banda superior reservada dinámicamente por encima de todas las barras para evitar cualquier colisión
    max_bar = float(df_v1["gdp_index_1980"].max())
    band_sep = max(220.0, max_bar + 9.0)
    card_y = band_sep + 30.0
    top_ylim = band_sep + 62.0

    ax_top.axhline(band_sep, color="#cbd5e1", linestyle="-", linewidth=0.8, alpha=0.7)
    ax_top.text(
        1985.2, card_y, regimes[0][3], ha="center", va="center", fontsize=9.8, fontweight="bold", color="#7f1d1d",
        bbox=dict(facecolor="white", edgecolor="#fca5a5", alpha=0.96, boxstyle="round,pad=0.4"),
    )
    ax_top.text(
        2000.5, card_y, regimes[1][3], ha="center", va="center", fontsize=9.8, fontweight="bold", color="#14532d",
        bbox=dict(facecolor="white", edgecolor="#86efac", alpha=0.96, boxstyle="round,pad=0.4"),
    )
    ax_top.text(
        2014.2, card_y, regimes[2][3], ha="center", va="center", fontsize=9.6, fontweight="bold", color="#78350f",
        bbox=dict(facecolor="white", edgecolor="#fde68a", alpha=0.96, boxstyle="round,pad=0.4"),
    )
    ax_top.text(
        2023.2, card_y, regimes[3][3], ha="center", va="center", fontsize=9.4, fontweight="bold", color="#0c4a6e",
        bbox=dict(facecolor="white", edgecolor="#7dd3fc", alpha=0.96, boxstyle="round,pad=0.4"),
    )

    ax_top.set_ylim(70, top_ylim)
    ax_top.set_yticks([75, 100, 125, 150, 175, 200])
    ax_top.set_ylabel("PBI Real (Índice Base 1980 = 100)", fontsize=13, fontweight="bold", color="#1e3a8a")
    ax_top.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2, fontsize=11.5, frameon=True, facecolor="white")
    ax_top.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(
        f"Políticas Económicas, Gasto Público, Inflación y PBI Real en Argentina (1980 – {max_yr})\n"
        "[Mejora: Índice Geométrico de PBI Base 1980=100 + Panel Fiscal/Monetario con Inflación en Escala Logarítmica]",
        fontsize=15.5, fontweight="bold", y=0.98,
    )

    ax_bot_r = ax_bot.twinx()
    ax_bot.plot(
        df_v1.index, df_v1["pubExpToGDP"], color="#dc2626", linewidth=3.5, marker="o", markersize=4,
        label="Gasto Público Consolidado (% PBI)", zorder=4,
    )
    x_max_lim = max_yr + 1.5
    ax_bot.axhspan(
        40, 48,
        xmin=(2011 - 1979) / (x_max_lim - 1979),
        xmax=(2023.5 - 1979) / (x_max_lim - 1979),
        color="#ef4444", alpha=0.15, label="Banda de Gasto Crítico (40% – 48% PBI)",
    )
    ax_bot.axhline(40, color="#dc2626", linestyle=":", linewidth=1.5, alpha=0.7)

    inf_pos = np.maximum(df_v1["inflation"], 0.5)
    ax_bot_r.plot(
        df_v1.index, inf_pos, color="#16a34a", linewidth=3.0, linestyle="-", marker="s", markersize=3.5,
        alpha=0.85, label="Inflación Anual (Escala Log, sin truncar)", zorder=3,
    )
    ax_bot_r.set_yscale("log")
    ax_bot_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}%"))
    ax_bot_r.set_ylim(0.3, 6000)

    ax_bot.set_ylim(23, 52)
    ax_bot.set_ylabel("Gasto Público Consolidado (% PBI)", fontsize=13, fontweight="bold", color="#b91c1c")
    ax_bot_r.set_ylabel("Inflación Anual (%, Escala Log)", fontsize=13, fontweight="bold", color="#15803d")
    ax_bot.set_xlabel("Año", fontsize=13, fontweight="bold")
    ax_bot.grid(axis="y", linestyle=":", alpha=0.5)

    h1, lb1 = ax_bot.get_legend_handles_labels()
    h2, lb2 = ax_bot_r.get_legend_handles_labels()
    ax_bot.legend(
        h1 + h2, lb1 + lb2, bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=3,
        fontsize=11.2, frameon=True, facecolor="white",
    )
    plt.xlim(1979, x_max_lim)
    return _save_figure(fig, out_dir, "macro_improved_2026_1")


def plot_visual_2(bundle: TransformedMacroBundle, out_dir: Path) -> Dict[str, str]:
    """Visual 2 (01/2009–Presente): Empleo Privado Registrado vs. EMAE Desestacionalizado + Elasticidad."""
    gdp_jobs = bundle.df_v2
    m = bundle.v2_metrics
    last_dt = gdp_jobs.index[-1]
    last_mm_yyyy = last_dt.strftime("%m/%Y")
    last_full_yr = int(m["last_full_year"])

    fig, (ax_ts, ax_sc) = plt.subplots(
        1, 2, figsize=(19, 8.2), gridspec_kw={"width_ratios": [1.85, 1.0], "wspace": 0.18},
    )
    fig.patch.set_facecolor("#ffffff")

    ax_ts.plot(
        gdp_jobs.index, gdp_jobs["emae_orig_idx"], color="#93c5fd", linewidth=1.0, alpha=0.45,
        label=f"EMAE Serie Original Con Estacionalidad (Corr: {m['corr_orig']}%)",
    )
    ax_ts.plot(
        gdp_jobs.index, gdp_jobs["emae_des_idx"], color="#1d4ed8", linewidth=2.5, alpha=0.9,
        label=f"EMAE Desestacionalizado (Corr: {m['corr_des']}% | Tend.-Ciclo: {m['corr_tc']}%)",
    )
    ax_ts.plot(
        gdp_jobs.index, gdp_jobs["jobs_idx"], color="#dc2626", linewidth=2.8,
        label="Empleo Privado Registrado SIPA (Base Ene-2009 = 100)",
    )

    ax_ts.axvline(pd.to_datetime("2012-01-01"), color="#334155", linestyle="--", linewidth=1.5)
    ax_ts.axvspan(pd.to_datetime("2009-01-01"), pd.to_datetime("2011-12-31"), color="#dcfce7", alpha=0.25)
    ax_ts.axvspan(pd.to_datetime("2012-01-01"), last_dt, color="#fef3c7", alpha=0.2)

    max_curve = float(max(gdp_jobs["emae_orig_idx"].max(), gdp_jobs["emae_des_idx"].max(), gdp_jobs["jobs_idx"].max()))
    band_sep = max(143.5, max_curve + 1.2)
    badge_y = band_sep + 6.3
    top_ylim = band_sep + 12.5

    ax_ts.axhline(band_sep, color="#cbd5e1", linestyle="-", linewidth=0.8, alpha=0.7)
    ax_ts.text(
        pd.to_datetime("2010-06-01"), badge_y,
        f"2009–2011\nEmpleo: +{m['cagr_jobs_2009_2011']:.2f}%/a\nPBI: +{m['cagr_gdp_2009_2011']:.2f}%/a",
        ha="center", va="center", fontsize=9.2, fontweight="bold", color="#14532d",
        bbox=dict(facecolor="white", edgecolor="#86efac", alpha=0.96, boxstyle="round,pad=0.35"),
    )
    ax_ts.text(
        pd.to_datetime("2019-06-01"), badge_y,
        f"2012–{last_full_yr} (Estancamiento Estructural)\n"
        f"CAGR Empleo Privado: +{m['cagr_jobs_2012_2025']:.2f}%/año  |  CAGR PBI Desest.: +{m['cagr_gdp_2012_2025']:.2f}%/año",
        ha="center", va="center", fontsize=10.2, fontweight="bold", color="#78350f",
        bbox=dict(facecolor="white", edgecolor="#fde68a", alpha=0.96, boxstyle="round,pad=0.4"),
    )

    ax_ts.set_ylim(82, top_ylim)
    ax_ts.set_yticks([90, 100, 110, 120, 130, 140])
    ax_ts.set_ylabel("Índice Base Ene-2009 = 100 (Escala Unificada)", fontsize=12.5, fontweight="bold")
    ax_ts.set_title(f"A. Evolución Conjunta sin Ruido Estacional (01/2009 – {last_mm_yyyy})", fontsize=13.5, fontweight="bold", pad=68)
    ax_ts.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=1, fontsize=10.0, frameon=True, facecolor="white")
    ax_ts.grid(True, linestyle=":", alpha=0.5)

    sub1 = gdp_jobs.loc["2009-01-01":"2011-12-01"]
    sub2 = gdp_jobs.loc["2012-01-01":]
    ax_sc.scatter(sub1["emae_des_idx"], sub1["jobs_idx"], color="#16a34a", alpha=0.75, s=36, label="2009 – 2011 (Expansión)")
    ax_sc.scatter(sub2["emae_des_idx"], sub2["jobs_idx"], color="#d97706", alpha=0.65, s=32, label=f"2012 – {last_dt.year} (Estancamiento)")

    x_line = np.linspace(float(gdp_jobs["emae_des_idx"].min()), float(gdp_jobs["emae_des_idx"].max()), 100)
    y_line = m["beta_slope"] * x_line + m["beta_intercept"]
    ax_sc.plot(
        x_line, y_line, color="#dc2626", linestyle="--", linewidth=2.5,
        label=f"Regresión Lineal (Pendiente β = {m['beta_slope']:.2f})",
    )

    ax_sc.set_ylim(93, 115)
    ax_sc.set_xlabel("EMAE Desestacionalizado (Base Ene-2009 = 100)", fontsize=12, fontweight="bold")
    ax_sc.set_ylabel("Empleo Privado (Base Ene-2009 = 100)", fontsize=12, fontweight="bold")
    ax_sc.set_title(f"B. Elasticidad Empleo-Producto (R = {m['corr_des']}%)", fontsize=13.5, fontweight="bold", pad=68)
    ax_sc.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=1, fontsize=10.0, frameon=True, facecolor="white")
    ax_sc.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Empleo Privado Registrado vs. Actividad Económica (EMAE) en Argentina (01/2009 – {last_mm_yyyy})\n"
        f"[Mejora: Al desestacionalizar el EMAE, la correlación sube de {m['corr_orig']}% a {m['corr_des']}% (y {m['corr_tc']}% en Tendencia-Ciclo)]",
        fontsize=15, fontweight="bold", y=1.10,
    )
    return _save_figure(fig, out_dir, "macro_improved_2026_2")


def plot_visual_3(bundle: TransformedMacroBundle, out_dir: Path) -> Dict[str, str]:
    """Visual 3 (01/2012–Presente): Crecimiento relativo acumulado geométrico exacto por modalidad SIPA."""
    geom_df = bundle.df_v3_geom
    pop_series = bundle.s_v3_pop
    gap_pp = bundle.v3_metrics["demographic_gap_pp"]
    last_mm_yyyy = geom_df.index[-1].strftime("%m/%Y")

    fig, ax = plt.subplots(figsize=(18, 9.5))
    fig.patch.set_facecolor("#ffffff")

    palette = {
        "Monotributo": ("#7c3aed", 3.2),
        "Empleo público": ("#ea580c", 3.0),
        "Monotributo Social": ("#92400e", 2.5),
        "Total": ("#0f172a", 3.2),
        "Empleo en casas particulares": ("#16a34a", 2.2),
        "Empleo privado": ("#2563eb", 3.5),
        "Autónomos": ("#dc2626", 2.2),
    }

    sorted_cols = geom_df.iloc[-1].sort_values(ascending=False).index.tolist()
    for col in sorted_cols:
        c_hex, lw = palette[col]
        final_val = float(geom_df[col].iloc[-1])
        ls = "--" if col == "Total" else "-"
        ax.plot(geom_df.index, geom_df[col], color=c_hex, linewidth=lw, linestyle=ls, label=f"{col}: {final_val:+.2f}%")

    ax.plot(
        pop_series.index, pop_series.values, color="#475569", linewidth=2.5, linestyle=":",
        label=f"Población Argentina (Proy. Censo): +{pop_series.iloc[-1]:.2f}%",
    )
    ax.fill_between(
        geom_df.index, geom_df["Empleo privado"], pop_series.values,
        color="#ef4444", alpha=0.08, label=f"Brecha Demográfica vs. Empleo Privado ({gap_pp:+.1f} pp)",
    )

    ax.axhline(0, color="#0f172a", linewidth=1.2, linestyle="-", alpha=0.7)
    ax.axvline(pd.to_datetime("2024-11-01"), color="#64748b", linestyle="--", linewidth=1.2, alpha=0.7)

    ax.annotate(
        "Pico Monotributo Social (+292% real en 2024)\ny caída por reempadronamiento en 2025",
        xy=(pd.to_datetime("2024-06-01"), 290),
        xytext=(pd.to_datetime("2017-06-01"), 235),
        fontsize=11, fontweight="bold", color="#78350f",
        bbox=dict(facecolor="#fef3c7", edgecolor="#f59e0b", alpha=0.95, boxstyle="round,pad=0.45"),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color="#b45309", lw=2.0),
    )

    ax.set_ylim(-20, 315)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    ax.set_ylabel("Crecimiento Acumulado Punta a Punta (%, Geométrico Real)", fontsize=13.5, fontweight="bold")
    ax.set_title(
        f"Crecimiento Relativo Acumulado Real por Modalidad de Empleo en Argentina (01/2012 – {last_mm_yyyy})\n"
        "[Mejora: Variación Compuesta Geométrica Exacta (X_t / X_0 - 1) en lugar de suma aritmética .cumsum()]",
        fontsize=15, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper left", fontsize=11.5, frameon=True, facecolor="white")
    ax.grid(True, linestyle=":", alpha=0.5)

    return _save_figure(fig, out_dir, "macro_improved_2026_3")


def plot_visual_4(bundle: TransformedMacroBundle, out_dir: Path) -> Dict[str, str]:
    """Visual 4 (01/2012 vs 03/2022 vs Último mes): Participación porcentual y desplazamiento neto (pp)."""
    summary = bundle.df_v4_summary
    last_dt = bundle.df_v4_props.index[-1]
    last_mm_yyyy = last_dt.strftime("%m/%Y")
    last_label = f"{SPANISH_MONTH_ABBR[last_dt.month]}-{last_dt.year} (Actual)"

    cols_mod = summary.index.tolist()
    p_2012 = summary["share_2012_01"].values
    p_2022 = summary["share_2022_03"].values
    p_latest = summary["share_latest"].values
    diff_pp = summary["diff_pp"].values

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(18, 7.2), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.08},
    )
    fig.patch.set_facecolor("#ffffff")

    y_pos = np.arange(len(cols_mod))
    bar_h = 0.26
    ax_l.barh(y_pos - bar_h, p_2012, height=bar_h, color="#94a3b8", label="Ene-2012 (Inicio serie)")
    ax_l.barh(y_pos, p_2022, height=bar_h, color="#3b82f6", label="Mar-2022 (Corte original)")
    ax_l.barh(y_pos + bar_h, p_latest, height=bar_h, color="#0f172a", label=last_label)

    for i, (v12, v22, v26) in enumerate(zip(p_2012, p_2022, p_latest)):
        ax_l.text(v26 + 0.7, i + bar_h, f"{v26:.2f}%", va="center", fontsize=10, fontweight="bold", color="#0f172a")
        ax_l.text(v12 + 0.7, i - bar_h, f"{v12:.2f}%", va="center", fontsize=9.5, color="#475569")

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(cols_mod, fontsize=11.5, fontweight="bold")
    ax_l.invert_yaxis()
    ax_l.set_xlim(0, 66)
    ax_l.set_xlabel("Participación sobre el Total de Empleo Registrado (%)", fontsize=12, fontweight="bold")
    ax_l.set_title(
        f"A. Participación por Modalidad (01/2012 vs. 03/2022 vs. {last_mm_yyyy})",
        fontsize=13.5, fontweight="bold", pad=36,
    )
    ax_l.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=3, fontsize=10.5, frameon=True, facecolor="white")
    ax_l.grid(axis="x", linestyle=":", alpha=0.5)

    colors_diff = ["#dc2626" if v < 0 else "#16a34a" for v in diff_pp]
    ax_r.barh(y_pos, diff_pp, height=0.55, color=colors_diff, alpha=0.85)
    ax_r.axvline(0, color="#0f172a", linewidth=1.2)
    for i, v in enumerate(diff_pp):
        offset = 0.35 if v >= 0 else -0.35
        ha = "left" if v >= 0 else "right"
        ax_r.text(v + offset, i, f"{v:+.2f} pp", va="center", ha=ha, fontsize=11, fontweight="bold", color=colors_diff[i])

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([])
    ax_r.invert_yaxis()
    ax_r.set_xlim(-11.8, 9.2)
    ax_r.set_xlabel(f"Cambio Estructural en Puntos Porcentuales (2012 → {last_dt.year})", fontsize=12, fontweight="bold")
    ax_r.set_title("B. Desplazamiento Neto de la Matriz Laboral (pp)", fontsize=13.5, fontweight="bold", pad=36)
    ax_r.grid(axis="x", linestyle=":", alpha=0.5)

    return _save_figure(fig, out_dir, "macro_improved_2026_4_table")


def plot_visual_5(bundle: TransformedMacroBundle, out_dir: Path) -> Dict[str, str]:
    """Visual 5 (2005–Presente): Salario Real Privado Registrado y PBI en Base 2012=100 + Variaciones % YoY."""
    df_v5 = bundle.df_v5
    max_yr = int(df_v5.index.max())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 11.5), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.24},
    )
    fig.patch.set_facecolor("#ffffff")

    ax1.plot(
        df_v5.index, df_v5["sal_real_idx_2012"], color="#1d4ed8", linewidth=3.5, marker="o", markersize=6,
        label="Salario Real Privado Registrado (Índice Geométrico Base 2012 = 100)",
    )
    ax1.plot(
        df_v5.index, df_v5["gdp_real_idx_2012"], color="#16a34a", linewidth=3.2, marker="s", markersize=5.5,
        linestyle="--", label="PBI Real Promedio Anual - EMAE (Índice Base 2012 = 100)",
    )
    ax1.axhline(100, color="#dc2626", linestyle=":", linewidth=1.8, label="Nivel Base 2012 = 100")
    ax1.fill_between(
        df_v5.index, 100, df_v5["sal_real_idx_2012"],
        where=(df_v5["sal_real_idx_2012"] < 100),
        color="#ef4444", alpha=0.12, interpolate=True,
        label="Pérdida de Poder Adquisitivo vs. Pico 2012–2013",
    )

    ax1.set_ylim(68, 109)
    ax1.set_ylabel("Índice Real (Base 2012 = 100)", fontsize=13, fontweight="bold")
    ax1.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2, fontsize=11, frameon=True, facecolor="white")
    ax1.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle(
        f"Evolución del Salario Real Privado Registrado y PBI en Argentina (2005 – {max_yr})\n"
        "[Mejora: Empalme 2015-10 deduplicado + Índice Compuesto Geométrico Base 2012=100 + Variaciones Anuales]",
        fontsize=15, fontweight="bold", y=1.01,
    )

    w = 0.36
    x_yrs = df_v5.index.values
    ax2.bar(
        x_yrs - w / 2, df_v5["real_sal_yoy"] * 100.0, width=w, color="#2563eb", alpha=0.85,
        label="Variación Anual Salario Real (% YoY)",
    )
    ax2.bar(
        x_yrs + w / 2, df_v5["gdp_yoy"] * 100.0, width=w, color="#16a34a", alpha=0.75,
        label="Variación Anual PBI - EMAE (% YoY)",
    )
    ax2.axhline(0, color="#0f172a", linewidth=1.2)
    ax2.set_ylim(-13, 13)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
    ax2.set_ylabel("Variación Interanual (% YoY)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Año", fontsize=13, fontweight="bold")
    ax2.set_xticks(x_yrs)
    ax2.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center", ncol=2, fontsize=11.5, frameon=True, facecolor="white")
    ax2.grid(True, linestyle=":", alpha=0.5)

    return _save_figure(fig, out_dir, "macro_improved_2026_4")


def generate_all_charts(
    bundle: TransformedMacroBundle,
    figures_dir: Union[str, Path] = "outputs/figures",
) -> Dict[str, Dict[str, str]]:
    """Genera las 5 visualizaciones principales en PNG y SVG en `outputs/figures/`."""
    out_path = Path(figures_dir)
    return {
        "visual_1": plot_visual_1(bundle, out_path),
        "visual_2": plot_visual_2(bundle, out_path),
        "visual_3": plot_visual_3(bundle, out_path),
        "visual_4": plot_visual_4(bundle, out_path),
        "visual_5": plot_visual_5(bundle, out_path),
    }

