"""D-PRIME replenishment engine extracted from the analysis notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np
import pandas as pd


InputFile = str | Path | BinaryIO


@dataclass(frozen=True)
class DPrimeConfig:
    tsb_alpha: float = 0.20
    tsb_beta: float = 0.40
    fallback_lead_time_days: float = 60.0
    review_cycle_months: float = 1.5

    # Dipakai hanya untuk Validation Mode.
    forecast_holdout_period: int | None = 7

    # Bulan actual terakhir untuk Production Mode.
    # 1 = Januari, 2 = Februari, ..., 12 = Desember.
    latest_period: int = 7


FILE_KEYS = ("usage", "stock", "policy", "watchlist", "migo", "backorder", "wrs")


def _read_excel(source: InputFile, *, preferred_sheet: str | None = None, header: int = 0) -> pd.DataFrame:
    if hasattr(source, "seek"):
        source.seek(0)
    book = pd.ExcelFile(source)
    sheet = preferred_sheet if preferred_sheet in book.sheet_names else book.sheet_names[0]
    return pd.read_excel(book, sheet_name=sheet, header=header)


def load_weekly_inputs(
    *, usage_file: InputFile, stock_file: InputFile, policy_file: InputFile,
    watchlist_file: InputFile, migo_file: InputFile, backorder_file: InputFile,
    wrs_file: InputFile,
) -> dict[str, pd.DataFrame]:
    inputs = {
        "usage": _read_excel(usage_file, preferred_sheet="Rank Nas", header=1),
        "stock": _read_excel(stock_file, preferred_sheet="STOCK"),
        "policy": _read_excel(policy_file),
        "watchlist": _read_excel(watchlist_file),
        "migo": _read_excel(migo_file),
        "backorder": _read_excel(backorder_file),
        "wrs": _read_excel(wrs_file),
    }
    for frame in inputs.values():
        frame.columns = [c.strip() if isinstance(c, str) else c for c in frame.columns]
    validate_weekly_inputs(inputs)
    return inputs


def validate_weekly_inputs(inputs: Mapping[str, pd.DataFrame]) -> None:
    required = {
        "usage": ["Part Number", "RANK Freq Call"],
        "stock": ["Material", "Available stock", "Plant"],
        "policy": ["Part Number", "Qty Min", "Qty Max"],
        "watchlist": ["Part Number"],
        "migo": ["Material", "Qty. Supply"],
        "backorder": ["Material", "Qty. Order"],
        "wrs": ["Material", "Qty. Order"],
    }
    errors: list[str] = []
    for name in FILE_KEYS:
        if name not in inputs:
            errors.append(f"missing dataset: {name}")
            continue
        missing = [c for c in required[name] if c not in inputs[name].columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
    if errors:
        raise ValueError("Invalid D-PRIME input:\n- " + "\n- ".join(errors))


def _pn(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _numeric(series: pd.Series, fill: float | None = None) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    return result.fillna(fill) if fill is not None else result


def _period_columns(frame: pd.DataFrame) -> list[object]:
    cols = [c for c in frame.columns if str(c).strip().isdigit()]
    return sorted(cols, key=lambda c: int(str(c).strip()))


def tsb_forecast(demand: np.ndarray, alpha: float = .20, beta: float = .40) -> float:
    values = np.asarray(demand, dtype=float)
    positive = np.where(values > 0)[0]
    if not len(positive):
        return 0.0
    first = int(positive[0])
    size, probability = values[first], 1 / (first + 1)
    for value in values[first + 1:]:
        if value > 0:
            size = alpha * value + (1 - alpha) * size
            probability = beta + (1 - beta) * probability
        else:
            probability = (1 - beta) * probability
    return max(0.0, float(size * probability))


def _demand_features(
    frame: pd.DataFrame,
    period_cols: list[object],
    config: DPrimeConfig,
) -> pd.DataFrame:

    # Mengubah demand menjadi numerik.
    # Missing value diisi 0 dan demand negatif dijadikan 0.
    values = (
        frame[period_cols]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
    )

    # Menghitung jumlah bulan dengan demand lebih dari 0.
    active = (
        values
        .gt(0)
        .sum(axis=1)
    )

    # Average Demand Interval.
    adi = (
        len(period_cols)
        / active.replace(0, np.nan)
    )

    # Demand non-zero untuk perhitungan CV².
    nonzero = values.where(
        values.gt(0)
    )

    cv2 = (
        nonzero.std(
            axis=1,
            ddof=1,
        )
        / nonzero.mean(axis=1)
    ).pow(2)

    # Jika demand hanya aktif satu kali,
    # variasinya dianggap 0.
    cv2 = cv2.where(
        active.ne(1),
        0.0,
    )

    # Klasifikasi pola demand.
    conditions = [
        adi.isna() | cv2.isna(),

        (
            (adi <= 1.32)
            & (cv2 <= 0.49)
        ),

        (
            (adi <= 1.32)
            & (cv2 > 0.49)
        ),

        (
            (adi > 1.32)
            & (cv2 <= 0.49)
        ),
    ]

    pattern_choices = [
        "No Demand",
        "Smooth",
        "Erratic",
        "Intermittent",
    ]

    frame["Demand_Pattern"] = np.select(
        conditions,
        pattern_choices,
        default="Lumpy",
    )

    # =====================================================
    # MENENTUKAN URUTAN HISTORI
    # =====================================================

    if config.forecast_holdout_period is not None:

        # Validation Mode:
        # bulan target tidak dimasukkan ke histori.
        history_cols = [
            column
            for column in period_cols
            if int(str(column).strip())
            != config.forecast_holdout_period
        ]

        # Contoh:
        # holdout Juli (7)
        # histori dimulai dari Agustus (8).
        first_history_period = (
            config.forecast_holdout_period % 12
        ) + 1

    else:

        # Production Mode:
        # seluruh histori 12 bulan digunakan.
        history_cols = period_cols.copy()

        # Contoh:
        # actual terakhir November (11)
        # urutan histori dimulai Desember (12)
        # dan berakhir November (11).
        first_history_period = (
            config.latest_period % 12
        ) + 1

    # Mengurutkan histori secara rolling.
    chronological = sorted(
        history_cols,
        key=lambda column: (
            int(str(column).strip())
            - first_history_period
        ) % 12,
    )

    # =====================================================
    # TSB FORECAST
    # =====================================================

    frame["Forecast_Demand"] = (
        values[chronological]
        .apply(
            lambda row: tsb_forecast(
                row.to_numpy(),
                config.tsb_alpha,
                config.tsb_beta,
            ),
            axis=1,
        )
    )

    # Variabilitas demand menggunakan seluruh histori.
    frame["Demand_Std_12M"] = values.std(
        axis=1,
        ddof=1,
    )

    return frame


def run_dprime(inputs: Mapping[str, pd.DataFrame], config: DPrimeConfig | None = None) -> pd.DataFrame:
    """Calculate D-PRIME from the seven raw weekly DataFrames; no notebook globals required."""
    config = config or DPrimeConfig()
    inputs = {name: df.copy() for name, df in inputs.items()}
    for df in inputs.values():
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    validate_weekly_inputs(inputs)

    usage, stock, policy = inputs["usage"], inputs["stock"], inputs["policy"]
    watch, migo, backorder, wrs = inputs["watchlist"], inputs["migo"], inputs["backorder"], inputs["wrs"]
    usage = usage.rename(columns={"Part Number": "PN", "RANK Freq Call": "Rank", "Description": "Part Description"})
    stock = stock.rename(columns={"Material": "PN", "Available stock": "Available_Stock"})
    policy = policy.rename(columns={"Part Number": "PN", "Qty Min": "Policy_Min", "Qty Max": "Policy_Max"})
    watch = watch.rename(columns={"Part Number": "PN"})
    for df in (usage, stock, policy, watch, migo, backorder, wrs):
        key = "PN" if "PN" in df else "Material"
        df[key] = _pn(df[key])

    period_cols = _period_columns(usage)
    if not period_cols:
        raise ValueError("usage: demand history columns (for example 1..12) were not found")
    usage = usage.drop_duplicates("PN", keep="first")
    scope = watch[["PN"]].dropna().drop_duplicates().merge(usage, on="PN", how="left")
    scope = scope[scope["Rank"].notna() & scope.get("MAD", pd.Series(np.nan, index=scope.index)).notna()].copy()

    stock["Available_Stock"] = _numeric(stock["Available_Stock"])
    stock["Plant"] = stock["Plant"].astype("string").str.strip().str.upper()

    # Keep the plant-level stock composition visible while preserving the
    # original replenishment basis: total available stock across all plants.
    stock_total = (
        stock.groupby("PN", as_index=False)["Available_Stock"]
        .sum(min_count=1)
        .rename(columns={"Available_Stock": "Total_Stock_On_Hand_All_Plant"})
    )
    hdo_stock = (
        stock.loc[stock["Plant"].eq("HDO")]
        .groupby("PN", as_index=False)["Available_Stock"]
        .sum(min_count=1)
        .rename(columns={"Available_Stock": "HDO_Stock"})
    )
    other_stock = (
        stock.loc[stock["Plant"].ne("HDO") & stock["Plant"].notna()]
        .groupby("PN", as_index=False)["Available_Stock"]
        .sum(min_count=1)
        .rename(columns={"Available_Stock": "Other_Stock_Without_HDO"})
    )
    stock_summary = stock_total.merge(hdo_stock, on="PN", how="left").merge(
        other_stock, on="PN", how="left"
    )
    has_stock_data = stock_summary["Total_Stock_On_Hand_All_Plant"].notna()
    stock_summary.loc[has_stock_data, ["HDO_Stock", "Other_Stock_Without_HDO"]] = (
        stock_summary.loc[
            has_stock_data, ["HDO_Stock", "Other_Stock_Without_HDO"]
        ].fillna(0)
    )
    policy_sum = policy[["PN", "Policy_Min", "Policy_Max"]].drop_duplicates("PN")

    def qty_summary(df: pd.DataFrame, value: str, output: str) -> pd.DataFrame:
        temp = df[["Material", value]].rename(columns={"Material": "PN"})
        temp[value] = _numeric(temp[value], 0)
        return temp.groupby("PN", as_index=False)[value].sum().rename(columns={value: output})

    result = (scope.merge(stock_summary, on="PN", how="left")
              .merge(policy_sum, on="PN", how="left")
              .merge(qty_summary(migo, "Qty. Supply", "Incoming_MIGO"), on="PN", how="left")
              .merge(qty_summary(wrs, "Qty. Order", "Incoming_WRS"), on="PN", how="left")
              .merge(qty_summary(backorder, "Qty. Order", "Backorder"), on="PN", how="left"))
    for col in ("Incoming_MIGO", "Incoming_WRS", "Backorder"):
        result[col] = result[col].fillna(0)
    result = _demand_features(result, period_cols, config)

    lead = wrs[["Material", "Lead Time (Days)"]].copy() if "Lead Time (Days)" in wrs else pd.DataFrame(columns=["Material", "Lead Time (Days)"])
    lead["Lead Time (Days)"] = _numeric(lead["Lead Time (Days)"])
    lead = lead[lead["Lead Time (Days)"].between(1, 365)]
    lead = lead.groupby("Material", as_index=False)["Lead Time (Days)"].median().rename(
        columns={"Material": "PN", "Lead Time (Days)": "Lead_Time_Days"})
    result = result.merge(lead, on="PN", how="left")
    result["Lead_Time_Source"] = np.where(result["Lead_Time_Days"].notna(), "WRS Actual", "G-Force 60-Day Fallback")
    result["Lead_Time_Days"] = result["Lead_Time_Days"].fillna(config.fallback_lead_time_days)
    months = result["Lead_Time_Days"] / 30
    result["Lead_Time_Demand"] = result["Forecast_Demand"] * months
    priority = result["Rank"].map({"A": 3.5, "B": 2.818, "C": 2.0}).fillna(0)
    lead_std = result["Demand_Std_12M"] * np.sqrt(months)
    base_ss = priority * lead_std
    low_frequency = result["Rank"].isin(["E", "F", "G"]) & result["Demand_Pattern"].isin(["Intermittent", "Lumpy"])
    result["Safety_Stock"] = np.maximum(base_ss, lead_std.where(low_frequency, 0))
    result["Min_Stock"] = result["Lead_Time_Demand"] + result["Safety_Stock"]
    result["Max_Stock"] = result["Min_Stock"] + config.review_cycle_months * result["Forecast_Demand"]
    result["Inventory_Position"] = result["Total_Stock_On_Hand_All_Plant"] + result["Incoming_MIGO"] + result["Incoming_WRS"] + result["Backorder"]
    requirement = np.where(result["Inventory_Position"] < result["Min_Stock"], result["Max_Stock"] - result["Inventory_Position"], 0)
    result["Suggested_Order"] = np.where(result["Inventory_Position"].notna(), np.ceil(np.maximum(requirement, 0)), np.nan)
    result["Decision"] = np.select(
        [result["Total_Stock_On_Hand_All_Plant"].isna(), result["Suggested_Order"].gt(0)],
        ["REVIEW REQUIRED", "ORDER"], default="NO ORDER")
    result["Decision_Reason"] = np.select(
        [result["Decision"].eq("ORDER"), result["Decision"].eq("NO ORDER")],
        ["Inventory Position Below Min Stock", "Inventory Position At/Above Min Stock"],
        default="Missing Stock Data")
    result["Gap_to_Min"] = result["Inventory_Position"] - result["Min_Stock"]
    result["Gap_to_Max"] = result["Max_Stock"] - result["Inventory_Position"]
    result["Action_Note"] = np.select(
        [result["Decision"].eq("ORDER"), result["Decision"].eq("NO ORDER")],
        ["Lakukan replenishment hingga mendekati Max Stock", "Belum perlu order karena Inventory Position masih di atas Min Stock"],
        default="Verifikasi Stock On Hand sebelum menentukan keputusan replenishment")
    columns = ["PN", "Part Description", "Rank", "Demand_Pattern", "Forecast_Demand", "Lead_Time_Days",
               "Lead_Time_Source", "Lead_Time_Demand", "Demand_Std_12M", "Safety_Stock", "Min_Stock", "Max_Stock",
               "HDO_Stock", "Other_Stock_Without_HDO", "Total_Stock_On_Hand_All_Plant",
               "Incoming_MIGO", "Incoming_WRS", "Backorder", "Inventory_Position", "Suggested_Order",
               "Decision", "Decision_Reason", "Gap_to_Min", "Gap_to_Max", "Action_Note"]
    return result[[c for c in columns if c in result]].rename(columns={"PN": "Part Number"}).reset_index(drop=True)


def regression_summary(result: pd.DataFrame) -> dict[str, int]:
    counts = result["Decision"].value_counts()
    return {"total_pn": len(result), "order": int(counts.get("ORDER", 0)),
            "no_order": int(counts.get("NO ORDER", 0)), "review_required": int(counts.get("REVIEW REQUIRED", 0)),
            "total_suggested_order": int(result["Suggested_Order"].sum())}