from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_LAGS = (1, 2, 3, 4, 8, 12, 26, 52)
DEFAULT_ROLLING_WINDOWS = (4, 12)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-8
    adjusted_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(mean_absolute_percentage_error(adjusted_true, y_pred) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": _safe_mape(y_true, y_pred),
    }


def _clip_prediction_series(series: pd.Series, clip_negative: bool) -> pd.Series:
    if not clip_negative:
        return series
    return series.clip(lower=0.0)


def make_ml_features(
    series: pd.Series,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    y = series.astype(float)
    frame = pd.DataFrame({"y": y})

    for lag in lags:
        frame[f"lag_{lag}"] = y.shift(lag)

    for window in rolling_windows:
        frame[f"roll_mean_{window}"] = y.shift(1).rolling(window=window).mean()
        frame[f"roll_std_{window}"] = y.shift(1).rolling(window=window).std()

    iso_week = frame.index.isocalendar().week.astype(int)
    frame["weekofyear"] = iso_week.values
    frame["month"] = frame.index.month
    frame["quarter"] = frame.index.quarter
    frame["year"] = frame.index.year
    frame["sin_week"] = np.sin(2 * np.pi * frame["weekofyear"] / 52.0)
    frame["cos_week"] = np.cos(2 * np.pi * frame["weekofyear"] / 52.0)

    return frame.dropna()


def _build_single_feature_row(
    history: list[float],
    next_date: pd.Timestamp,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> dict[str, float]:
    row: dict[str, float] = {}
    history_array = np.array(history, dtype=float)
    fallback = float(np.nanmean(history_array))

    for lag in lags:
        row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else fallback

    for window in rolling_windows:
        if len(history) >= window:
            window_values = history_array[-window:]
            row[f"roll_mean_{window}"] = float(np.nanmean(window_values))
            row[f"roll_std_{window}"] = float(np.nanstd(window_values))
        else:
            row[f"roll_mean_{window}"] = fallback
            row[f"roll_std_{window}"] = 0.0

    iso_week = int(next_date.isocalendar().week)
    row["weekofyear"] = iso_week
    row["month"] = int(next_date.month)
    row["quarter"] = int(next_date.quarter)
    row["year"] = int(next_date.year)
    row["sin_week"] = float(np.sin(2 * np.pi * iso_week / 52.0))
    row["cos_week"] = float(np.cos(2 * np.pi * iso_week / 52.0))

    return row


def recursive_ml_forecast(
    model: RandomForestRegressor,
    series: pd.Series,
    horizon_weeks: int,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    feature_columns: list[str] | None = None,
    clip_negative: bool = True,
) -> pd.Series:
    history = series.astype(float).tolist()
    start_date = series.index[-1]
    future_index = pd.date_range(
        start=start_date + pd.Timedelta(weeks=1),
        periods=horizon_weeks,
        freq="W-FRI",
    )

    predictions: list[float] = []
    for date in future_index:
        features = _build_single_feature_row(
            history=history,
            next_date=date,
            lags=lags,
            rolling_windows=rolling_windows,
        )
        x_next = pd.DataFrame([features])

        if feature_columns is not None:
            for feature in feature_columns:
                if feature not in x_next.columns:
                    x_next[feature] = 0.0
            x_next = x_next[feature_columns]

        y_next = float(model.predict(x_next)[0])
        if clip_negative:
            y_next = max(0.0, y_next)
        predictions.append(y_next)
        history.append(y_next)

    return pd.Series(predictions, index=future_index, name="ml_forecast")


@dataclass
class ModelRun:
    metrics: dict[str, float]
    holdout_predictions: pd.Series
    future_predictions: pd.Series
    details: dict[str, Any]


def run_arima(
    series: pd.Series,
    holdout_weeks: int,
    future_horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 52),
    clip_negative: bool = True,
) -> ModelRun:
    if len(series) <= holdout_weeks + 10:
        raise ValueError("Series is too short for ARIMA validation split.")

    train_series = series.iloc[:-holdout_weeks]
    test_series = series.iloc[-holdout_weeks:]

    used_seasonal_order = seasonal_order
    arima_warning = None

    try:
        fitted = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    except Exception:
        used_seasonal_order = (0, 0, 0, 0)
        arima_warning = "Seasonal ARIMA fit failed; switched to non-seasonal ARIMA."
        fitted = SARIMAX(
            train_series,
            order=order,
            seasonal_order=used_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    holdout_pred = fitted.forecast(steps=holdout_weeks)
    holdout_pred.index = test_series.index
    holdout_pred = _clip_prediction_series(holdout_pred, clip_negative=clip_negative)
    metrics = regression_metrics(test_series.values, holdout_pred.values)

    full_fit = SARIMAX(
        series,
        order=order,
        seasonal_order=used_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(weeks=1),
        periods=future_horizon,
        freq="W-FRI",
    )
    future_pred = full_fit.forecast(steps=future_horizon)
    future_pred.index = future_index
    future_pred = _clip_prediction_series(future_pred, clip_negative=clip_negative)
    future_pred.name = "arima_forecast"

    return ModelRun(
        metrics=metrics,
        holdout_predictions=holdout_pred.rename("arima_holdout"),
        future_predictions=future_pred,
        details={
            "order": order,
            "seasonal_order": used_seasonal_order,
            "warning": arima_warning,
            "clip_negative": clip_negative,
        },
    )


def run_ml(
    series: pd.Series,
    holdout_weeks: int,
    future_horizon: int,
    n_estimators: int = 300,
    random_state: int = 42,
    clip_negative: bool = True,
) -> ModelRun:
    feature_df = make_ml_features(series)
    if feature_df.empty:
        raise ValueError("Not enough historical points to create ML lag features.")
    if len(feature_df) <= holdout_weeks + 5:
        raise ValueError("Not enough engineered rows for ML validation split.")

    train_df = feature_df.iloc[:-holdout_weeks]
    test_df = feature_df.iloc[-holdout_weeks:]

    x_train = train_df.drop(columns=["y"])
    y_train = train_df["y"]
    x_test = test_df.drop(columns=["y"])
    y_test = test_df["y"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    holdout_values = model.predict(x_test)
    if clip_negative:
        holdout_values = np.clip(holdout_values, a_min=0.0, a_max=None)
    holdout_pred = pd.Series(holdout_values, index=y_test.index, name="ml_holdout")
    metrics = regression_metrics(y_test.values, holdout_values)

    full_df = make_ml_features(series)
    x_full = full_df.drop(columns=["y"])
    y_full = full_df["y"]

    full_model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    full_model.fit(x_full, y_full)
    future_pred = recursive_ml_forecast(
        model=full_model,
        series=series,
        horizon_weeks=future_horizon,
        feature_columns=x_full.columns.tolist(),
        clip_negative=clip_negative,
    )

    return ModelRun(
        metrics=metrics,
        holdout_predictions=holdout_pred,
        future_predictions=future_pred,
        details={
            "n_estimators": n_estimators,
            "clip_negative": clip_negative,
        },
    )


def compare_models(
    series: pd.Series,
    holdout_weeks: int,
    future_horizon: int,
    arima_order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 52),
    n_estimators: int = 300,
    clip_negative: bool = True,
) -> dict[str, Any]:
    arima_run = run_arima(
        series=series,
        holdout_weeks=holdout_weeks,
        future_horizon=future_horizon,
        order=arima_order,
        seasonal_order=seasonal_order,
        clip_negative=clip_negative,
    )
    ml_run = run_ml(
        series=series,
        holdout_weeks=holdout_weeks,
        future_horizon=future_horizon,
        n_estimators=n_estimators,
        clip_negative=clip_negative,
    )

    ensemble_holdout_values = (
        arima_run.holdout_predictions.values + ml_run.holdout_predictions.values
    ) / 2.0
    ensemble_holdout = pd.Series(
        ensemble_holdout_values,
        index=arima_run.holdout_predictions.index,
        name="ensemble_holdout",
    )
    holdout_actual = series.iloc[-holdout_weeks:]
    ensemble_metrics = regression_metrics(holdout_actual.values, ensemble_holdout.values)

    ensemble_future_values = (
        arima_run.future_predictions.values + ml_run.future_predictions.values
    ) / 2.0
    ensemble_future = pd.Series(
        ensemble_future_values,
        index=arima_run.future_predictions.index,
        name="ensemble_forecast",
    )
    ensemble_run = ModelRun(
        metrics=ensemble_metrics,
        holdout_predictions=ensemble_holdout,
        future_predictions=ensemble_future,
        details={
            "strategy": "Average of ARIMA and ML forecasts",
            "clip_negative": clip_negative,
        },
    )

    metrics_df = pd.DataFrame(
        [
            {"Model": "ARIMA", **arima_run.metrics},
            {"Model": "ML (RandomForest)", **ml_run.metrics},
            {"Model": "Ensemble (ARIMA + ML)", **ensemble_run.metrics},
        ]
    ).sort_values("MAE", ascending=True)

    return {
        "arima": arima_run,
        "ml": ml_run,
        "ensemble": ensemble_run,
        "metrics_table": metrics_df.reset_index(drop=True),
    }

