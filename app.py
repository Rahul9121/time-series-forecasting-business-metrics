from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from forecasting.data import (  # noqa: E402
    DATASET_NAME,
    FIELD_DESCRIPTIONS,
    KAGGLE_URL,
    PUBLIC_MIRROR_BASE_URL,
    build_sales_series,
    dataset_summary,
    load_walmart_data,
    selector_options,
)
from forecasting.models import compare_models  # noqa: E402
from forecasting.seasonality import detect_seasonality  # noqa: E402

LIVE_APP_URL = "https://time-series-forecasting-business-metrics-by-rahulega.streamlit.app/"

st.set_page_config(
    page_title="Business Metrics Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        border: 1px solid rgba(120,120,120,0.25);
        border-radius: 12px;
        padding: 14px 16px;
        background: rgba(120,120,120,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Time Series Forecasting for Business Metrics")
st.caption("Built by **rahul ega**")
st.markdown(
    f"""
    <div class="hero-box">
    Interactive dashboard for business forecasting with seasonality insights and model benchmarking.<br>
    Live App: <a href="{LIVE_APP_URL}" target="_blank">{LIVE_APP_URL}</a>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    return load_walmart_data(base_dir=ROOT_DIR, force_refresh=force_refresh)


with st.sidebar:
    st.header("⚙️ Controls")
    force_refresh_data = st.button("🔄 Refresh Dataset from Source")

try:
    data = get_data(force_refresh=force_refresh_data)
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

train_df = data["train"]
features_df = data["features"]
stores_df = data["stores"]
merged_df = data["merged"]
summary = dataset_summary(train_df=train_df, features_df=features_df, stores_df=stores_df)
options = selector_options(merged_df)

with st.sidebar:
    st.subheader("Series Selection")
    aggregation_level = st.selectbox(
        "Aggregation Level",
        options=["All Stores + All Departments", "Store", "Department", "Store + Department"],
        index=0,
    )
    selected_store = None
    selected_dept = None
    if aggregation_level in {"Store", "Store + Department"}:
        selected_store = st.selectbox("Store", options=options["stores"])
    if aggregation_level in {"Department", "Store + Department"}:
        selected_dept = st.selectbox("Department", options=options["depts"])

series = build_sales_series(
    df=merged_df,
    aggregation_level=aggregation_level,
    store=selected_store,
    dept=selected_dept,
)
series_length = len(series)

holdout_max = min(26, max(2, series_length - 12))
holdout_min = 6
if holdout_max < holdout_min:
    holdout_min = holdout_max
holdout_default = min(12, holdout_max)

seasonal_period_max = min(60, max(4, (series_length // 2) - 1))
seasonal_period_default = min(52, seasonal_period_max)

with st.sidebar:
    st.subheader("Forecast Settings")
    forecast_horizon = st.slider("Future Forecast Horizon (weeks)", min_value=4, max_value=52, value=16, step=1)
    holdout_weeks = st.slider(
        "Validation Holdout Window (weeks)",
        min_value=holdout_min,
        max_value=holdout_max,
        value=holdout_default,
        step=1,
    )

    st.subheader("ARIMA Parameters")
    arima_p = st.slider("ARIMA p", min_value=0, max_value=3, value=1, step=1)
    arima_d = st.slider("ARIMA d", min_value=0, max_value=2, value=1, step=1)
    arima_q = st.slider("ARIMA q", min_value=0, max_value=3, value=1, step=1)
    use_seasonal = st.checkbox("Use Seasonal ARIMA", value=True)
    seasonal_period = st.slider(
        "Seasonal Period",
        min_value=4,
        max_value=seasonal_period_max,
        value=seasonal_period_default,
        step=1,
    )

    st.subheader("ML Parameters")
    n_estimators = st.slider("RandomForest Trees", min_value=100, max_value=1000, value=350, step=50)
    clip_negative = st.checkbox("Clip Negative Predictions to 0", value=True)

    st.subheader("Output")
    preferred_output = st.selectbox(
        "Preferred Forecast Output",
        options=["Best holdout model", "ARIMA", "ML (RandomForest)", "Ensemble (ARIMA + ML)"],
        index=0,
    )
    run_forecast = st.button("Run Forecast & Model Comparison", type="primary")

if "latest_forecast" not in st.session_state:
    st.session_state.latest_forecast = None
if "latest_forecast_context" not in st.session_state:
    st.session_state.latest_forecast_context = None
if "latest_forecast_error" not in st.session_state:
    st.session_state.latest_forecast_error = None

if run_forecast:
    seasonal_order = (1, 1, 1, seasonal_period) if use_seasonal else (0, 0, 0, 0)
    with st.spinner("Training ARIMA, ML, and ensemble models..."):
        try:
            results = compare_models(
                series=series,
                holdout_weeks=holdout_weeks,
                future_horizon=forecast_horizon,
                arima_order=(arima_p, arima_d, arima_q),
                seasonal_order=seasonal_order,
                n_estimators=n_estimators,
                clip_negative=clip_negative,
            )
            st.session_state.latest_forecast = results
            st.session_state.latest_forecast_error = None
            st.session_state.latest_forecast_context = {
                "series_name": series.name,
                "aggregation": aggregation_level,
                "holdout_weeks": holdout_weeks,
                "forecast_horizon": forecast_horizon,
            }
        except Exception as exc:
            st.session_state.latest_forecast = None
            st.session_state.latest_forecast_error = str(exc)

overview_tab, forecast_tab, dataset_tab, about_tab = st.tabs(
    ["📊 Overview", "🤖 Forecast Results", "🗂 Dataset", "ℹ️ About"]
)

with overview_tab:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Observations", f"{series_length}")
    metric_col2.metric("Average Weekly Sales", f"{series.mean():,.0f}")
    metric_col3.metric("Latest Weekly Sales", f"{series.iloc[-1]:,.0f}")
    metric_col4.metric("Series Frequency", "Weekly")

    if series_length < 70:
        st.warning("Short history detected. ML lag features and seasonal modeling may be less stable.")

    st.subheader("Historical Series")
    st.line_chart(series.rename("Weekly_Sales"), height=320)

    seasonality = detect_seasonality(series=series, period=seasonal_period)
    st.subheader("Seasonality Insights")
    season_col1, season_col2, season_col3 = st.columns(3)
    season_col1.metric("Seasonality Flag", "Yes" if seasonality["is_seasonal"] else "No")
    season_col2.metric(
        f"Autocorr @ lag {seasonality['period']}",
        "N/A" if seasonality["autocorr_at_period"] is None else f"{seasonality['autocorr_at_period']:.3f}",
    )
    season_col3.metric(
        "Seasonal Strength",
        "N/A" if seasonality["seasonal_strength"] is None else f"{seasonality['seasonal_strength']:.3f}",
    )

    if seasonality["notes"]:
        st.info(seasonality["notes"])

    if seasonality["decomposition"] is not None:
        with st.expander("Show Seasonal Decomposition Plot"):
            fig = seasonality["decomposition"].plot()
            fig.set_size_inches(12, 8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

with forecast_tab:
    st.subheader("Model Comparison")
    if st.session_state.latest_forecast_error:
        st.error(f"Model run failed: {st.session_state.latest_forecast_error}")
    elif st.session_state.latest_forecast is None:
        st.info("Click **Run Forecast & Model Comparison** in the sidebar to generate results.")
    else:
        results = st.session_state.latest_forecast
        context = st.session_state.latest_forecast_context or {}
        st.caption(
            f"Results for: {context.get('series_name', 'selected series')} | "
            f"Holdout: {context.get('holdout_weeks', holdout_weeks)} weeks | "
            f"Horizon: {context.get('forecast_horizon', forecast_horizon)} weeks"
        )

        metrics_table = results["metrics_table"].copy()
        st.dataframe(
            metrics_table.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE_%": "{:,.2f}%"}),
            use_container_width=True,
        )
        best_model = metrics_table.iloc[0]["Model"]
        st.success(f"Best model on holdout window: {best_model}")

        arima_run = results["arima"]
        ml_run = results["ml"]
        ensemble_run = results["ensemble"]

        if arima_run.details.get("warning"):
            st.warning(arima_run.details["warning"])

        model_series_map = {
            "ARIMA": arima_run.future_predictions,
            "ML (RandomForest)": ml_run.future_predictions,
            "Ensemble (ARIMA + ML)": ensemble_run.future_predictions,
        }
        chosen_model = best_model if preferred_output == "Best holdout model" else preferred_output
        chosen_forecast = model_series_map[chosen_model].rename("Chosen Output Forecast")

        st.subheader("Holdout Performance View")
        holdout_actual = series.iloc[-holdout_weeks:].rename("Actual")
        holdout_plot = pd.concat(
            [
                holdout_actual,
                arima_run.holdout_predictions.rename("ARIMA"),
                ml_run.holdout_predictions.rename("ML"),
                ensemble_run.holdout_predictions.rename("Ensemble"),
            ],
            axis=1,
        )
        st.line_chart(holdout_plot, height=320)

        st.subheader("Future Forecast View")
        history_tail = series.iloc[-40:].rename("Historical (recent)")
        forecast_plot = pd.concat(
            [
                history_tail,
                arima_run.future_predictions.rename("ARIMA Forecast"),
                ml_run.future_predictions.rename("ML Forecast"),
                ensemble_run.future_predictions.rename("Ensemble Forecast"),
                chosen_forecast,
            ],
            axis=1,
        )
        st.line_chart(forecast_plot, height=350)
        st.info(f"Preferred output currently selected: **{chosen_model}**")

        forecast_download_df = pd.concat(
            [
                arima_run.future_predictions.rename("arima_forecast"),
                ml_run.future_predictions.rename("ml_forecast"),
                ensemble_run.future_predictions.rename("ensemble_forecast"),
                chosen_forecast.rename("chosen_output_forecast"),
            ],
            axis=1,
        ).reset_index(names="Date")
        st.download_button(
            label="Download Forecast CSV",
            data=forecast_download_df.to_csv(index=False),
            file_name="future_forecast_comparison.csv",
            mime="text/csv",
        )

with dataset_tab:
    st.subheader("Dataset Information")
    st.markdown(
        f"""
**Dataset Used:** {DATASET_NAME}  
**Original Source:** {KAGGLE_URL}  
**Public Mirror Used by App:** {PUBLIC_MIRROR_BASE_URL}
"""
    )
    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("Train Rows", f"{summary['train_rows']:,}")
    top_mid.metric("Stores", summary["n_stores"])
    top_right.metric("Departments", summary["n_depts"])
    st.markdown(
        f"**Date Range:** {summary['date_min']} → {summary['date_max']}  \n"
        f"**Features Rows:** {summary['features_rows']:,} | **Stores Rows:** {summary['stores_rows']:,}"
    )

    with st.expander("Dataset Fields Included"):
        for field in FIELD_DESCRIPTIONS:
            st.write(f"- {field}")

    with st.expander("Preview: train.csv"):
        st.dataframe(train_df.head(20), use_container_width=True)
    with st.expander("Preview: features.csv"):
        st.dataframe(features_df.head(20), use_container_width=True)
    with st.expander("Preview: stores.csv"):
        st.dataframe(stores_df.head(20), use_container_width=True)

with about_tab:
    st.subheader("About This Project")
    st.markdown(
        f"""
This project forecasts business metrics (sales/demand trends) using weekly Walmart sales data.

### What it does
- Detects seasonality in the selected series.
- Compares **ARIMA** and **ML (RandomForest)** models.
- Adds an **Ensemble forecast** (average of ARIMA + ML) for a more stable option.
- Exports future forecasts as CSV.

### Why this is useful
The app supports planning in demand estimation, inventory readiness, and revenue trend tracking.

### Live dashboard
{LIVE_APP_URL}
"""
    )
    st.markdown("Dashboard author: **rahul ega**")

