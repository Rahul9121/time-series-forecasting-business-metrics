from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def detect_seasonality(series: pd.Series, period: int = 52) -> dict[str, Any]:
    clean_series = series.dropna().astype(float)
    output: dict[str, Any] = {
        "is_seasonal": False,
        "period": period,
        "autocorr_at_period": None,
        "seasonal_strength": None,
        "decomposition": None,
        "notes": "",
    }

    if len(clean_series) < period * 2:
        output["notes"] = (
            f"Need at least {period * 2} observations for robust weekly seasonality decomposition; "
            f"found {len(clean_series)}."
        )
        return output

    acf_at_period = float(clean_series.autocorr(lag=period))
    decomposition = seasonal_decompose(clean_series, model="additive", period=period, extrapolate_trend="freq")

    residual_var = float(np.nanvar(decomposition.resid))
    seasonal_var = float(np.nanvar(decomposition.seasonal))
    denom = residual_var + seasonal_var
    seasonal_strength = 0.0 if denom == 0 else max(0.0, 1 - residual_var / denom)

    is_seasonal = abs(acf_at_period) >= 0.30 or seasonal_strength >= 0.20

    output.update(
        {
            "is_seasonal": is_seasonal,
            "autocorr_at_period": acf_at_period,
            "seasonal_strength": seasonal_strength,
            "decomposition": decomposition,
            "notes": "Seasonality flagged using autocorrelation and decomposition strength thresholds.",
        }
    )
    return output

