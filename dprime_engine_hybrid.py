"""Experimental D-PRIME hybrid forecast engine.

This module leaves the stable ``dprime_engine.py`` untouched.  It reproduces
the notebook routing rule:

* Erratic -> XGBoost with demand, calendar, and lagged PMI features
* Lumpy -> Random Forest with demand-history features
* Intermittent, Smooth, No Demand -> TSB

Currency is intentionally not included: the supplied currency workbook is a
single weekly snapshot rather than a historical time series suitable for
validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise ImportError(
        "D-PRIME Hybrid membutuhkan package xgboost. "
        "Jalankan: python -m pip install xgboost"
    ) from exc

from dprime_engine import (
    DPrimeConfig,
    run_dprime,
    tsb_forecast,
    validate_weekly_inputs,
)


INTERNAL_FEATURES = [
    "Demand_Lag1",
    "Demand_Lag2",
    "Demand_Lag3",
    "Rolling_Mean_3",
    "Rolling_Std_3",
    "Active_Months_3",
]

CALENDAR_FEATURES = INTERNAL_FEATURES + [
    "Month_Sin",
    "Month_Cos",
]

EVENT_NAMES = [
    "New Year", "Chinese New Year", "Ramadan", "Eid al-Fitr",
    "Eid al-Adha", "Mawlid", "Independence Day", "Christmas",
]
EVENT_FEATURES = [
    f"Event_{event.replace(' ', '_').replace('-', '_')}_{position}"
    for event in EVENT_NAMES
    for position in ["Pre", "During", "Post"]
]
CALENDAR_FEATURES = CALENDAR_FEATURES + EVENT_FEATURES

EXTERNAL_FEATURES = CALENDAR_FEATURES + ["PMI_Lag1", "Seasonal_Index"]

ROUTING_RULE = {
    "Intermittent": "TSB",
    "Lumpy": "Random Forest",
    "Erratic": "XGBoost",
    "Smooth": "TSB",
    "No Demand": "TSB",
}


@dataclass(frozen=True)
class HybridConfig(DPrimeConfig):
    """Configuration for the experimental hybrid engine."""

    latest_year: int = 2026
    rf_n_estimators: int = 300
    rf_max_depth: int = 5
    rf_min_samples_leaf: int = 1
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 2
    xgb_learning_rate: float = 0.03
    random_state: int = 42


def default_pmi_data() -> pd.DataFrame:
    """Return the PMI series embedded in the analysis notebook."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2025-07-01", "2025-08-01", "2025-09-01",
                    "2025-10-01", "2025-11-01", "2025-12-01",
                    "2026-01-01", "2026-02-01", "2026-03-01",
                    "2026-04-01", "2026-05-01", "2026-06-01",
                    "2026-07-01",
                ]
            ),
            "PMI_Value": [
                49.2, 51.5, 50.4, 51.2, 53.3, 51.2, 52.6,
                53.8, 50.1, 49.1, 50.0, 46.9, 50.2,
            ],
        }
    )


def default_event_calendar() -> pd.DataFrame:
    """Seed calendar; future dates can be merged from the Streamlit app."""
    rows = [
        ("New Year", "2024-01-01"), ("Chinese New Year", "2024-02-10"),
        ("Ramadan", "2024-03-12"), ("Eid al-Fitr", "2024-04-10"),
        ("Eid al-Adha", "2024-06-17"), ("Independence Day", "2024-08-17"),
        ("Mawlid", "2024-09-16"), ("Christmas", "2024-12-25"),
        ("New Year", "2025-01-01"), ("Chinese New Year", "2025-01-29"),
        ("Ramadan", "2025-03-01"), ("Eid al-Fitr", "2025-03-31"),
        ("Eid al-Adha", "2025-06-06"), ("Independence Day", "2025-08-17"),
        ("Mawlid", "2025-09-05"), ("Christmas", "2025-12-25"),
        ("New Year", "2026-01-01"), ("Chinese New Year", "2026-02-17"),
        ("Ramadan", "2026-02-18"), ("Eid al-Fitr", "2026-03-21"),
        ("Eid al-Adha", "2026-05-27"), ("Independence Day", "2026-08-17"),
        ("Mawlid", "2026-08-26"), ("Christmas", "2026-12-25"),
    ]
    return pd.DataFrame(rows, columns=["Event", "Event_Date"]).assign(
        Event_Date=lambda frame: pd.to_datetime(frame["Event_Date"])
    )


def _normalize_pn(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _period_columns(frame: pd.DataFrame) -> list[object]:
    columns = [column for column in frame if str(column).strip().isdigit()]
    return sorted(columns, key=lambda column: int(str(column).strip()))


def _period_date(month: int, latest_month: int, latest_year: int) -> pd.Timestamp:
    year = latest_year if month <= latest_month else latest_year - 1
    return pd.Timestamp(year=year, month=month, day=1)


def _classify_patterns(values: pd.DataFrame) -> pd.Series:
    active = values.gt(0).sum(axis=1)
    adi = len(values.columns) / active.replace(0, np.nan)
    nonzero = values.where(values.gt(0))
    cv2 = (nonzero.std(axis=1, ddof=1) / nonzero.mean(axis=1)).pow(2)
    cv2 = cv2.where(active.ne(1), 0.0)
    return pd.Series(
        np.select(
            [
                adi.isna() | cv2.isna(),
                (adi <= 1.32) & (cv2 <= 0.49),
                (adi <= 1.32) & (cv2 > 0.49),
                (adi > 1.32) & (cv2 <= 0.49),
            ],
            ["No Demand", "Smooth", "Erratic", "Intermittent"],
            default="Lumpy",
        ),
        index=values.index,
        dtype="string",
    )


def _prepare_pmi(pmi_data: pd.DataFrame | None) -> pd.DataFrame:
    pmi = default_pmi_data() if pmi_data is None else pmi_data.copy()
    required = {"Date", "PMI_Value"}
    missing = required.difference(pmi.columns)
    if missing:
        raise ValueError(f"PMI data: missing columns {sorted(missing)}")
    pmi["Date"] = pd.to_datetime(pmi["Date"]).dt.to_period("M").dt.to_timestamp()
    pmi["PMI_Value"] = pd.to_numeric(pmi["PMI_Value"], errors="coerce")
    if pmi["Date"].duplicated().any():
        raise ValueError("PMI data: one row per month is required")
    pmi = pmi.sort_values("Date").dropna(subset=["Date", "PMI_Value"])
    # Attach month t's published PMI to target month t+1, preventing leakage.
    pmi["Date"] = pmi["Date"] + pd.DateOffset(months=1)
    return pmi.rename(columns={"PMI_Value": "PMI_Lag1"})[
        ["Date", "PMI_Lag1"]
    ]


def _prepare_seasonality(seasonal_data: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize compact monthly trend history used for causal seasonal indices."""
    if seasonal_data is None or seasonal_data.empty:
        raise ValueError(
            "Historical seasonality belum tersedia. Upload Trend Parts terlebih dahulu."
        )
    seasonal = seasonal_data.copy()
    required = {"Date", "Seasonal_Qty"}
    missing = required.difference(seasonal.columns)
    if missing:
        raise ValueError(
            "Historical seasonality membutuhkan kolom Date dan Seasonal_Qty"
        )
    seasonal["Date"] = (
        pd.to_datetime(seasonal["Date"], errors="coerce")
        .dt.to_period("M").dt.to_timestamp()
    )
    seasonal["Seasonal_Qty"] = pd.to_numeric(
        seasonal["Seasonal_Qty"], errors="coerce"
    )
    seasonal = seasonal.dropna().groupby("Date", as_index=False)[
        "Seasonal_Qty"
    ].sum().sort_values("Date")
    return seasonal


def _seasonal_indices(
    dates: pd.Series, seasonal_data: pd.DataFrame | None
) -> pd.DataFrame:
    """Create a leakage-safe month index using only trend history before each date."""
    seasonal = _prepare_seasonality(seasonal_data)
    rows = []
    for date in pd.Series(dates.unique()).sort_values():
        history = seasonal[seasonal["Date"].lt(date)]
        overall = history["Seasonal_Qty"].mean()
        same_month = history.loc[
            history["Date"].dt.month.eq(date.month), "Seasonal_Qty"
        ]
        index = same_month.mean() / overall if overall and not same_month.empty else 1.0
        rows.append({"Date": date, "Seasonal_Index": float(index)})
    return pd.DataFrame(rows)


def _event_features(
    dates: pd.Series, event_data: pd.DataFrame | None
) -> pd.DataFrame:
    events = default_event_calendar() if event_data is None else event_data.copy()
    required = {"Event", "Event_Date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Event Calendar membutuhkan kolom Event dan Event_Date")
    events["Event"] = events["Event"].astype("string").str.strip()
    events["Event_Date"] = pd.to_datetime(
        events["Event_Date"], errors="coerce"
    ).dt.to_period("M")
    events = events.dropna(subset=["Event", "Event_Date"])

    unique_dates = pd.DataFrame({"Date": pd.Series(dates.unique()).sort_values()})
    unique_dates["_period"] = unique_dates["Date"].dt.to_period("M")
    for event in EVENT_NAMES:
        slug = event.replace(" ", "_").replace("-", "_")
        event_periods = set(events.loc[events["Event"].eq(event), "Event_Date"])
        unique_dates[f"Event_{slug}_Pre"] = unique_dates["_period"].map(
            lambda period: int(period + 1 in event_periods)
        )
        unique_dates[f"Event_{slug}_During"] = unique_dates["_period"].map(
            lambda period: int(period in event_periods)
        )
        unique_dates[f"Event_{slug}_Post"] = unique_dates["_period"].map(
            lambda period: int(period - 1 in event_periods)
        )
    return unique_dates.drop(columns="_period")


def prepare_model_data(
    inputs: Mapping[str, pd.DataFrame],
    config: HybridConfig | None = None,
    pmi_data: pd.DataFrame | None = None,
    seasonal_data: pd.DataFrame | None = None,
    event_data: pd.DataFrame | None = None,
    *,
    include_next_month: bool = False,
) -> pd.DataFrame:
    """Create the long-form feature table used by RF and XGBoost."""
    config = config or HybridConfig()
    copied = {name: frame.copy() for name, frame in inputs.items()}
    for frame in copied.values():
        frame.columns = [
            column.strip() if isinstance(column, str) else column
            for column in frame.columns
        ]
    validate_weekly_inputs(copied)

    usage = copied["usage"].rename(
        columns={
            "Part Number": "PN",
            "Description": "Part Description",
            "RANK Freq Call": "Rank",
        }
    )
    watch = copied["watchlist"].rename(columns={"Part Number": "PN"})
    usage["PN"] = _normalize_pn(usage["PN"])
    watch["PN"] = _normalize_pn(watch["PN"])
    usage = usage.drop_duplicates("PN", keep="first")
    scope = watch[["PN"]].dropna().drop_duplicates().merge(
        usage, on="PN", how="left"
    )
    mad = scope.get("MAD", pd.Series(np.nan, index=scope.index))
    scope = scope[scope["Rank"].notna() & mad.notna()].copy()

    period_columns = _period_columns(scope)
    if len(period_columns) != 12:
        raise ValueError(
            "Hybrid modeling requires 12 monthly Usage columns numbered 1..12"
        )
    values = (
        scope[period_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    scope["Demand Pattern"] = _classify_patterns(values)

    demand = scope[
        ["PN", "Part Description", "Rank", "Demand Pattern"]
        + period_columns
    ].melt(
        id_vars=["PN", "Part Description", "Rank", "Demand Pattern"],
        value_vars=period_columns,
        var_name="Period",
        value_name="Demand",
    )
    demand = demand.rename(columns={"PN": "Part Number"})
    demand["Period"] = demand["Period"].astype(int)
    demand["Demand"] = pd.to_numeric(demand["Demand"], errors="coerce").fillna(0)
    demand["Date"] = demand["Period"].map(
        lambda month: _period_date(
            month, config.latest_period, config.latest_year
        )
    )

    if include_next_month:
        target_date = pd.Timestamp(
            year=config.latest_year,
            month=config.latest_period,
            day=1,
        ) + pd.DateOffset(months=1)
        target = scope[
            ["PN", "Part Description", "Rank", "Demand Pattern"]
        ].rename(columns={"PN": "Part Number"})
        target["Period"] = target_date.month
        target["Demand"] = np.nan
        target["Date"] = target_date
        demand = pd.concat([demand, target], ignore_index=True)

    demand = demand.sort_values(["Part Number", "Date"]).reset_index(drop=True)
    grouped = demand.groupby("Part Number", sort=False)["Demand"]
    demand["Demand_Lag1"] = grouped.shift(1)
    demand["Demand_Lag2"] = grouped.shift(2)
    demand["Demand_Lag3"] = grouped.shift(3)
    demand["Rolling_Mean_3"] = grouped.transform(
        lambda series: series.shift(1).rolling(3).mean()
    )
    demand["Rolling_Std_3"] = grouped.transform(
        lambda series: series.shift(1).rolling(3).std()
    )
    demand["Active_Months_3"] = grouped.transform(
        lambda series: series.shift(1).rolling(3).apply(
            lambda values_: (values_ > 0).sum(), raw=True
        )
    )
    demand["Month_Sin"] = np.sin(2 * np.pi * demand["Date"].dt.month / 12)
    demand["Month_Cos"] = np.cos(2 * np.pi * demand["Date"].dt.month / 12)
    demand = demand.merge(
        _event_features(demand["Date"], event_data), on="Date", how="left"
    )
    demand = demand.merge(_prepare_pmi(pmi_data), on="Date", how="left")
    demand = demand.merge(
        _seasonal_indices(demand["Date"], seasonal_data),
        on="Date",
        how="left",
    )
    return demand


def _fit_predict_ml(
    train: pd.DataFrame,
    target: pd.DataFrame,
    config: HybridConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rf = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_leaf=config.rf_min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )
    rf.fit(train[INTERNAL_FEATURES], train["Demand"])
    rf_prediction = np.maximum(rf.predict(target[INTERNAL_FEATURES]), 0)

    # Use the same XGBoost configuration for the internal-only counterfactual.
    # The difference from the external model therefore comes from the feature
    # set, not from a different algorithm or hyperparameter configuration.
    xgb_internal = XGBRegressor(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=-1,
    )
    xgb_internal.fit(train[INTERNAL_FEATURES], train["Demand"])
    xgb_internal_prediction = np.maximum(
        xgb_internal.predict(target[INTERNAL_FEATURES]), 0
    )

    if train[EXTERNAL_FEATURES].isna().any().any():
        raise ValueError("Hybrid training data contains missing calendar/PMI features")
    if target[EXTERNAL_FEATURES].isna().any().any():
        dates = target.loc[
            target[EXTERNAL_FEATURES].isna().any(axis=1), "Date"
        ].dt.strftime("%Y-%m").unique()
        raise ValueError(
            "PMI bulan sebelumnya belum tersedia untuk target: "
            + ", ".join(dates)
        )
    xgb = XGBRegressor(
        n_estimators=config.xgb_n_estimators,
        max_depth=config.xgb_max_depth,
        learning_rate=config.xgb_learning_rate,
        objective="reg:squarederror",
        random_state=config.random_state,
        n_jobs=-1,
    )
    xgb.fit(train[EXTERNAL_FEATURES], train["Demand"])
    xgb_prediction = np.maximum(xgb.predict(target[EXTERNAL_FEATURES]), 0)
    return rf_prediction, xgb_internal_prediction, xgb_prediction


def _tsb_predictions(
    demand: pd.DataFrame,
    target_date: pd.Timestamp,
    config: HybridConfig,
) -> pd.DataFrame:
    rows = []
    target = demand[demand["Date"].eq(target_date)]
    for part_number in target["Part Number"]:
        history = demand[
            demand["Part Number"].eq(part_number)
            & demand["Date"].lt(target_date)
        ].sort_values("Date")["Demand"]
        rows.append(
            {
                "Part Number": part_number,
                "TSB_Prediction": tsb_forecast(
                    history.to_numpy(), config.tsb_alpha, config.tsb_beta
                ),
            }
        )
    return pd.DataFrame(rows)


def forecast_hybrid_for_month(
    model_data: pd.DataFrame,
    target_date: pd.Timestamp | str,
    config: HybridConfig | None = None,
) -> pd.DataFrame:
    """Fit expanding-history models and predict one target month."""
    config = config or HybridConfig()
    target_date = pd.Timestamp(target_date).to_period("M").to_timestamp()
    eligible = model_data.dropna(subset=INTERNAL_FEATURES).copy()
    train = eligible[
        eligible["Date"].lt(target_date) & eligible["Demand"].notna()
    ]
    target = eligible[eligible["Date"].eq(target_date)].copy()
    if train.empty or target.empty:
        raise ValueError(f"Insufficient modeling rows for target {target_date:%Y-%m}")

    rf_prediction, xgb_internal_prediction, xgb_prediction = _fit_predict_ml(
        train, target, config
    )
    output = target[
        ["Part Number", "Demand Pattern", "Demand"]
    ].rename(columns={"Demand": "Actual"})
    output["RF_Prediction"] = rf_prediction
    output["Internal_XGB_Prediction"] = xgb_internal_prediction
    output["XGB_Prediction"] = xgb_prediction
    output = output.merge(
        _tsb_predictions(model_data, target_date, config),
        on="Part Number",
        how="left",
        validate="one_to_one",
    )
    output["Selected_Model"] = output["Demand Pattern"].map(ROUTING_RULE)
    output["Hybrid_Prediction"] = np.select(
        [
            output["Selected_Model"].eq("TSB"),
            output["Selected_Model"].eq("Random Forest"),
            output["Selected_Model"].eq("XGBoost"),
        ],
        [
            output["TSB_Prediction"],
            output["RF_Prediction"],
            output["XGB_Prediction"],
        ],
        default=np.nan,
    )
    external_applied = output["Selected_Model"].eq("XGBoost")
    output["Internal_Forecast"] = np.where(
        external_applied,
        output["Internal_XGB_Prediction"],
        output["Hybrid_Prediction"],
    )
    output["External_Forecast"] = output["Hybrid_Prediction"]
    output["External_Impact_Units"] = (
        output["External_Forecast"] - output["Internal_Forecast"]
    )
    output["External_Impact_Pct"] = np.where(
        output["Internal_Forecast"].gt(0),
        output["External_Impact_Units"] / output["Internal_Forecast"] * 100,
        np.nan,
    )
    output.insert(0, "Date", target_date)
    return output.reset_index(drop=True)


def validate_hybrid(
    inputs: Mapping[str, pd.DataFrame],
    pmi_data: pd.DataFrame | None = None,
    seasonal_data: pd.DataFrame | None = None,
    event_data: pd.DataFrame | None = None,
    config: HybridConfig | None = None,
    validation_months: Sequence[str | pd.Timestamp] = (
        "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01"
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run notebook-equivalent expanding-window validation."""
    config = config or HybridConfig(latest_period=7, latest_year=2026)
    data = prepare_model_data(
        inputs, config, pmi_data, seasonal_data, event_data
    )
    predictions = pd.concat(
        [
            forecast_hybrid_for_month(data, month, config)
            for month in validation_months
        ],
        ignore_index=True,
    )
    actual = predictions["Actual"]
    metrics = []
    for strategy, column in [
        ("Global TSB", "TSB_Prediction"),
        ("Pattern-Based Hybrid", "Hybrid_Prediction"),
    ]:
        metrics.append(
            {
                "Strategy": strategy,
                "MAE": mean_absolute_error(actual, predictions[column]),
                "RMSE": np.sqrt(mean_squared_error(actual, predictions[column])),
                "Rows": len(predictions),
            }
        )
    return predictions, pd.DataFrame(metrics)


def _recalculate_replenishment(
    result: pd.DataFrame,
    config: HybridConfig,
) -> pd.DataFrame:
    result = result.copy()
    months = result["Lead_Time_Days"] / 30
    result["Lead_Time_Demand"] = result["Forecast_Demand"] * months
    priority = result["Rank"].map({"A": 3.5, "B": 2.818, "C": 2.0}).fillna(0)
    lead_std = result["Demand_Std_12M"] * np.sqrt(months)
    base_ss = priority * lead_std
    low_frequency = result["Rank"].isin(["E", "F", "G"]) & result[
        "Demand_Pattern"
    ].isin(["Intermittent", "Lumpy"])
    result["Safety_Stock"] = np.maximum(
        base_ss, lead_std.where(low_frequency, 0)
    )
    result["Min_Stock"] = result["Lead_Time_Demand"] + result["Safety_Stock"]
    result["Max_Stock"] = (
        result["Min_Stock"] + config.review_cycle_months * result["Forecast_Demand"]
    )
    requirement = np.where(
        result["Inventory_Position"] < result["Min_Stock"],
        result["Max_Stock"] - result["Inventory_Position"],
        0,
    )
    result["Suggested_Order"] = np.where(
        result["Inventory_Position"].notna(),
        np.ceil(np.maximum(requirement, 0)),
        np.nan,
    )
    result["Decision"] = np.select(
        [
            result["Total_Stock_On_Hand_All_Plant"].isna(),
            result["Suggested_Order"].gt(0),
        ],
        ["REVIEW REQUIRED", "ORDER"],
        default="NO ORDER",
    )
    result["Decision_Reason"] = np.select(
        [result["Decision"].eq("ORDER"), result["Decision"].eq("NO ORDER")],
        ["Inventory Position Below Min Stock", "Inventory Position At/Above Min Stock"],
        default="Missing Stock Data",
    )
    result["Gap_to_Min"] = result["Inventory_Position"] - result["Min_Stock"]
    result["Gap_to_Max"] = result["Max_Stock"] - result["Inventory_Position"]
    result["Action_Note"] = np.select(
        [result["Decision"].eq("ORDER"), result["Decision"].eq("NO ORDER")],
        [
            "Lakukan replenishment hingga mendekati Max Stock",
            "Belum perlu order karena Inventory Position masih di atas Min Stock",
        ],
        default="Verifikasi Stock On Hand sebelum menentukan keputusan replenishment",
    )
    return result


def run_dprime_hybrid(
    inputs: Mapping[str, pd.DataFrame],
    pmi_data: pd.DataFrame | None = None,
    seasonal_data: pd.DataFrame | None = None,
    event_data: pd.DataFrame | None = None,
    config: HybridConfig | None = None,
) -> pd.DataFrame:
    """Run the experimental hybrid forecast and D-PRIME replenishment logic."""
    config = config or HybridConfig()
    baseline = run_dprime(inputs, config=config)
    model_data = prepare_model_data(
        inputs, config, pmi_data, seasonal_data, event_data,
        include_next_month=True
    )
    target_date = pd.Timestamp(
        year=config.latest_year,
        month=config.latest_period,
        day=1,
    ) + pd.DateOffset(months=1)
    forecast = forecast_hybrid_for_month(model_data, target_date, config)
    selected = forecast[
        [
            "Part Number",
            "Hybrid_Prediction",
            "Selected_Model",
            "Internal_Forecast",
            "External_Forecast",
            "External_Impact_Units",
            "External_Impact_Pct",
        ]
    ].rename(
        columns={
            "Hybrid_Prediction": "Hybrid_Forecast_Demand",
            "Selected_Model": "Forecast_Model",
        }
    )
    output = baseline.merge(
        selected, on="Part Number", how="left", validate="one_to_one"
    )
    if output["Hybrid_Forecast_Demand"].isna().any():
        raise ValueError("Hybrid forecast is missing for one or more Part Numbers")
    output["Forecast_Demand"] = output.pop("Hybrid_Forecast_Demand")
    output = _recalculate_replenishment(output, config)

    ordered = list(baseline.columns)
    forecast_position = ordered.index("Forecast_Demand") + 1
    ordered[forecast_position:forecast_position] = [
        "Forecast_Model",
        "Internal_Forecast",
        "External_Forecast",
        "External_Impact_Units",
        "External_Impact_Pct",
    ]
    return output[ordered].reset_index(drop=True)


def hybrid_summary(result: pd.DataFrame) -> dict[str, int]:
    counts = result["Decision"].value_counts()
    return {
        "total_pn": len(result),
        "order": int(counts.get("ORDER", 0)),
        "no_order": int(counts.get("NO ORDER", 0)),
        "review_required": int(counts.get("REVIEW REQUIRED", 0)),
        "total_suggested_order": int(result["Suggested_Order"].sum()),
    }