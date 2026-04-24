from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DATASET_NAME = "Walmart Store Sales Forecasting (Kaggle)"
KAGGLE_URL = "https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data"
PUBLIC_MIRROR_BASE_URL = "https://raw.githubusercontent.com/gagandeepsinghkhanuja/Walmart-Sales-Forecasting/master"

DATA_URLS = {
    "train": f"{PUBLIC_MIRROR_BASE_URL}/train.csv",
    "features": f"{PUBLIC_MIRROR_BASE_URL}/features.csv",
    "stores": f"{PUBLIC_MIRROR_BASE_URL}/stores.csv",
}

FIELD_DESCRIPTIONS = [
    "Store: store number",
    "Dept: department number",
    "Date: weekly timestamp",
    "Weekly_Sales: target metric",
    "IsHoliday: holiday flag",
    "Temperature/Fuel_Price/CPI/Unemployment: macro and store context",
    "MarkDown1..5: promotional markdown variables",
    "Type/Size: store metadata",
]


def ensure_data_dir(base_dir: Path | None = None) -> Path:
    root = base_dir if base_dir is not None else Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_if_needed(url: str, destination: Path, force_refresh: bool = False) -> None:
    if destination.exists() and not force_refresh:
        return
    df = pd.read_csv(url)
    df.to_csv(destination, index=False)


def load_walmart_data(base_dir: Path | None = None, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    data_dir = ensure_data_dir(base_dir=base_dir)
    frames: dict[str, pd.DataFrame] = {}

    for name, url in DATA_URLS.items():
        local_file = data_dir / f"{name}.csv"
        _download_if_needed(url=url, destination=local_file, force_refresh=force_refresh)
        frames[name] = pd.read_csv(local_file)

    train = frames["train"].copy()
    features = frames["features"].copy()
    stores = frames["stores"].copy()

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    merged = (
        train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
        .merge(stores, on="Store", how="left")
        .sort_values("Date")
    )

    return {
        "train": train,
        "features": features,
        "stores": stores,
        "merged": merged,
    }


def selector_options(df: pd.DataFrame) -> dict[str, list[int]]:
    stores = sorted(df["Store"].dropna().astype(int).unique().tolist())
    depts = sorted(df["Dept"].dropna().astype(int).unique().tolist())
    return {"stores": stores, "depts": depts}


def build_sales_series(
    df: pd.DataFrame,
    aggregation_level: str,
    store: int | None = None,
    dept: int | None = None,
) -> pd.Series:
    work_df = df.copy()
    work_df["Date"] = pd.to_datetime(work_df["Date"])

    if aggregation_level == "All Stores + All Departments":
        grouped = work_df.groupby("Date", as_index=False)["Weekly_Sales"].sum()
        label = "Total Weekly Sales (All Stores + All Departments)"
    elif aggregation_level == "Store":
        if store is None:
            raise ValueError("Store must be selected for Store aggregation.")
        grouped = (
            work_df.loc[work_df["Store"] == store]
            .groupby("Date", as_index=False)["Weekly_Sales"]
            .sum()
        )
        label = f"Weekly Sales | Store {store}"
    elif aggregation_level == "Department":
        if dept is None:
            raise ValueError("Department must be selected for Department aggregation.")
        grouped = (
            work_df.loc[work_df["Dept"] == dept]
            .groupby("Date", as_index=False)["Weekly_Sales"]
            .sum()
        )
        label = f"Weekly Sales | Department {dept}"
    elif aggregation_level == "Store + Department":
        if store is None or dept is None:
            raise ValueError("Store and Department must be selected for Store + Department aggregation.")
        grouped = (
            work_df.loc[(work_df["Store"] == store) & (work_df["Dept"] == dept)]
            .groupby("Date", as_index=False)["Weekly_Sales"]
            .sum()
        )
        label = f"Weekly Sales | Store {store} + Department {dept}"
    else:
        raise ValueError(f"Unsupported aggregation level: {aggregation_level}")

    series = (
        grouped.sort_values("Date")
        .set_index("Date")["Weekly_Sales"]
        .asfreq("W-FRI")
        .interpolate(method="linear")
    )
    series.name = label
    return series


def dataset_summary(train_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "train_rows": int(train_df.shape[0]),
        "features_rows": int(features_df.shape[0]),
        "stores_rows": int(stores_df.shape[0]),
        "date_min": str(train_df["Date"].min().date()),
        "date_max": str(train_df["Date"].max().date()),
        "n_stores": int(train_df["Store"].nunique()),
        "n_depts": int(train_df["Dept"].nunique()),
    }

