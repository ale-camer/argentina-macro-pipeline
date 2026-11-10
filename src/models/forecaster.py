"""Short-term deterministic statistical forecasting (SARIMAX) for economic activity and employment."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_monthly_series(
    series: pd.Series,
    steps: int = 6,
    order: tuple[int, int, int] = (1, 1, 1),
) -> List[Dict[str, Any]]:
    """Fits a deterministic SARIMAX model and projects `steps` months ahead with 95% confidence intervals."""
    clean = series.dropna().astype(float)
    if len(clean) < 24:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            clean,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps)
        mean_forecast = pred.predicted_mean
        conf_int = pred.conf_int(alpha=0.05)

    future_dates = pd.date_range(
        start=clean.index[-1] + pd.DateOffset(months=1),
        periods=steps,
        freq="MS",
    )

    output: List[Dict[str, Any]] = []
    for i, dt in enumerate(future_dates):
        val = float(mean_forecast.iloc[i])
        low = float(conf_int.iloc[i, 0])
        high = float(conf_int.iloc[i, 1])
        output.append(
            {
                "period": dt.strftime("%Y-%m-%d"),
                "point_forecast": round(val, 2),
                "ci95_low": round(low, 2),
                "ci95_high": round(high, 2),
            }
        )
    return output


def generate_macro_forecasts(
    df_v2: pd.DataFrame,
    steps: int = 6,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generates 6-month deterministic SARIMAX projections for Seasonally Adjusted EMAE and Private Employment."""
    return {
        "emae_des_idx_base2009": forecast_monthly_series(df_v2["emae_des_idx"], steps=steps),
        "private_jobs_idx_base2009": forecast_monthly_series(df_v2["jobs_idx"], steps=steps),
    }
