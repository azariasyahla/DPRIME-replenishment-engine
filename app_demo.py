from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
import os
import re
import shutil

import pandas as pd
import streamlit as st
import altair as alt

from dprime_engine import (
    DPrimeConfig,
    load_weekly_inputs,
    regression_summary,
    run_dprime,
)
from dprime_engine_hybrid import (
    HybridConfig,
    default_event_calendar,
    default_pmi_data,
    run_dprime_hybrid,
)

STOCK_DISPLAY_NAMES = {
    "HDO_Stock": "HDO Stock",
    "Other_Stock_Without_HDO": "Other Stock (Without HDO)",
    "Total_Stock_On_Hand_All_Plant": "Total Stock On Hand (All Plant)",
}


def load_part_master():
    """Use optional master attributes when available in a local installation."""
    master_path = BASE_DIR / "master_data" / "master_part.xlsx"
    if not master_path.exists():
        return pd.DataFrame({"PN": pd.Series(dtype="string")})
    master = pd.read_excel(master_path, sheet_name="Part Master")
    master.columns = [str(column).strip() for column in master.columns]
    master["PN"] = master["PN"].astype("string").str.strip()
    return master.drop_duplicates("PN", keep="first")


def build_operational_output(master, dprime):
    """Keep every original D-PRIME column and append SAP/EPC attributes."""
    original_dprime_columns = list(dprime.columns)
    # Retain these diagnostic fields inside the engine, but keep the user-facing
    # recommendation table and its Excel export focused on the final forecast.
    hidden_diagnostic_columns = [
        "Forecast_Model",
        "Internal_Forecast",
        "External_Forecast",
        "External_Impact_Units",
        "External_Impact_Pct",
        "Lead_Time_Days",
        "Lead_Time_Source",
    ]
    output = dprime.drop(columns=hidden_diagnostic_columns, errors="ignore").copy()
    output["Part Number"] = output["Part Number"].astype("string").str.strip()

    master_for_merge = master.copy().rename(columns={"PN": "Part Number"})
    master_for_merge["Part Number"] = (
        master_for_merge["Part Number"].astype("string").str.strip()
    )
    output = output.merge(
        master_for_merge,
        on="Part Number",
        how="left",
        validate="one_to_one",
    )

    # Show one description: keep the D-PRIME value, using master data only
    # when that value is missing or blank.
    if "Description" in output.columns:
        if "Part Description" in output.columns:
            primary = output["Part Description"].astype("string")
            primary = primary.mask(primary.str.strip().eq(""))
            master_description = output["Description"].astype("string")
            master_description = master_description.mask(
                master_description.str.strip().eq("")
            )
            output["Part Description"] = primary.fillna(master_description)
        else:
            output["Part Description"] = output["Description"]
        output = output.drop(columns="Description")

    master_columns = [
        "Model Unit", "Qty Needed Per Unit", "Remarks",
        "Besi Baja", "APEX", "FOB",
    ]
    master_columns = [column for column in master_columns if column in output]

    # Part Number + one description, then master attributes and D-PRIME output.
    leading = [
        column for column in ["Part Number", "Part Description"]
        if column in output.columns
    ]
    dprime_remaining = [
        column for column in original_dprime_columns
        if column not in leading and column in output.columns
    ]
    return output[leading + master_columns + dprime_remaining].rename(
        columns=STOCK_DISPLAY_NAMES
    )


def build_gforce_comparison(dprime_result, gforce_file):
    """Compare D-PRIME and G-Force suggested orders by Part Number."""
    try:
        gforce = pd.read_excel(gforce_file, sheet_name="Plan")
    except ValueError as exc:
        raise ValueError(
            "The G-Force file must contain a sheet named 'Plan'."
        ) from exc

    required_columns = {"PN", "Suggested Order"}
    missing_columns = required_columns.difference(gforce.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Required G-Force columns not found: {missing_text}."
        )

    gforce_clean = gforce[["PN", "Suggested Order"]].copy()
    gforce_clean = gforce_clean.rename(
        columns={
            "PN": "Part Number",
            "Suggested Order": "GForce_Suggested_Order",
        }
    )
    gforce_clean["Part Number"] = (
        gforce_clean["Part Number"].astype("string").str.strip()
    )
    gforce_clean["GForce_Suggested_Order"] = pd.to_numeric(
        gforce_clean["GForce_Suggested_Order"],
        errors="coerce",
    ).fillna(0)
    gforce_clean = (
        gforce_clean.dropna(subset=["Part Number"])
        .groupby("Part Number", as_index=False)["GForce_Suggested_Order"]
        .sum()
    )

    dprime_columns = [
        "Part Number", "Part Description", "Forecast_Demand",
        "Inventory_Position", "Suggested_Order", "Decision",
        "Decision_Reason",
    ]
    if "Forecast_Model" in dprime_result.columns:
        dprime_columns.insert(3, "Forecast_Model")
    dprime_clean = dprime_result[dprime_columns].copy()
    dprime_clean["Part Number"] = (
        dprime_clean["Part Number"].astype("string").str.strip()
    )
    dprime_clean = dprime_clean.rename(
        columns={
            "Suggested_Order": "DPRIME_Suggested_Order",
            "Decision": "DPRIME_Decision",
            "Decision_Reason": "DPRIME_Decision_Reason",
        }
    )

    comparison = dprime_clean.merge(
        gforce_clean,
        on="Part Number",
        how="outer",
        validate="one_to_one",
    )
    for column in ["DPRIME_Suggested_Order", "GForce_Suggested_Order"]:
        comparison[column] = pd.to_numeric(
            comparison[column], errors="coerce"
        ).fillna(0)

    comparison["Qty_Difference_DPRIME_minus_GForce"] = (
        comparison["DPRIME_Suggested_Order"]
        - comparison["GForce_Suggested_Order"]
    )
    dprime_order = comparison["DPRIME_Suggested_Order"].gt(0)
    gforce_order = comparison["GForce_Suggested_Order"].gt(0)
    comparison["Comparison_Status"] = "BOTH NO ORDER"
    comparison.loc[dprime_order & gforce_order, "Comparison_Status"] = (
        "BOTH ORDER"
    )
    comparison.loc[dprime_order & ~gforce_order, "Comparison_Status"] = (
        "D-PRIME ONLY"
    )
    comparison.loc[~dprime_order & gforce_order, "Comparison_Status"] = (
        "G-FORCE ONLY"
    )

    status_order = {
        "D-PRIME ONLY": 0,
        "G-FORCE ONLY": 1,
        "BOTH ORDER": 2,
        "BOTH NO ORDER": 3,
    }
    comparison["_status_order"] = comparison["Comparison_Status"].map(
        status_order
    )
    return (
        comparison.sort_values(
            ["_status_order", "Qty_Difference_DPRIME_minus_GForce"],
            ascending=[True, False],
        )
        .drop(columns="_status_order")
        .reset_index(drop=True)
    )
# BASIC CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent

# Simpan data operasional di lokasi permanen milik user Windows. Dengan begitu
# data tidak ikut hilang saat folder aplikasi dipindah atau app.py diganti.
if os.name == "nt" and os.getenv("LOCALAPPDATA"):
    APP_DATA_ROOT = Path(os.environ["LOCALAPPDATA"]) / "DPRIME"
else:
    APP_DATA_ROOT = BASE_DIR

DATA_DIR = APP_DATA_ROOT / "data_input"
ARCHIVE_DIR = APP_DATA_ROOT / "archive"
EXTERNAL_DIR = APP_DATA_ROOT / "external_input"
PMI_FILE = EXTERNAL_DIR / "pmi_history.xlsx"
SEASONAL_FILE = EXTERNAL_DIR / "seasonality_history.xlsx"
CURRENCY_FILE = EXTERNAL_DIR / "currency_history.xlsx"
EVENT_FILE = EXTERNAL_DIR / "event_calendar.xlsx"
MODEL_MONITORING_DIR = APP_DATA_ROOT / "model_monitoring"
FORECAST_HISTORY_FILE = MODEL_MONITORING_DIR / "forecast_history.csv"

# Reference performance from the March-June 2026 validation period.
REFERENCE_MAE = 2.9676
REFERENCE_RMSE = 7.0470


def save_forecast_snapshot(
    result: pd.DataFrame,
    *,
    target_year: int,
    target_month: int,
    engine_name: str,
) -> None:
    """Persist a forecast until actual demand for its target month arrives."""
    required = {"Part Number", "Forecast_Demand"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(
            "Forecast snapshot cannot be saved because these columns are "
            f"missing: {', '.join(sorted(missing))}."
        )

    MODEL_MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = result[["Part Number", "Forecast_Demand"]].copy()
    snapshot["Part Number"] = (
        snapshot["Part Number"].astype("string").str.strip()
    )
    snapshot["Forecast_Demand"] = pd.to_numeric(
        snapshot["Forecast_Demand"], errors="coerce"
    )
    snapshot = snapshot.dropna(subset=["Part Number", "Forecast_Demand"])
    snapshot["Target_Year"] = int(target_year)
    snapshot["Target_Month"] = int(target_month)
    snapshot["Forecast_Engine"] = engine_name
    snapshot["Generated_At"] = datetime.now().isoformat(timespec="seconds")

    if FORECAST_HISTORY_FILE.exists():
        history = pd.read_csv(
            FORECAST_HISTORY_FILE,
            dtype={"Part Number": "string"},
        )
        same_target = (
            pd.to_numeric(history["Target_Year"], errors="coerce").eq(target_year)
            & pd.to_numeric(history["Target_Month"], errors="coerce").eq(target_month)
        )
        history = history.loc[~same_target].copy()
        snapshot = pd.concat([history, snapshot], ignore_index=True)

    snapshot.to_csv(FORECAST_HISTORY_FILE, index=False)


def evaluate_saved_forecast(
    usage: pd.DataFrame,
    *,
    actual_year: int,
    actual_month: int,
) -> dict | None:
    """Compare a stored forecast with newly available actual demand."""
    if not FORECAST_HISTORY_FILE.exists():
        return None

    history = pd.read_csv(
        FORECAST_HISTORY_FILE,
        dtype={"Part Number": "string"},
    )
    selected = history[
        pd.to_numeric(history["Target_Year"], errors="coerce").eq(actual_year)
        & pd.to_numeric(history["Target_Month"], errors="coerce").eq(actual_month)
    ].copy()
    if selected.empty:
        return None

    period_column = next(
        (
            column for column in usage.columns
            if str(column).strip() == str(int(actual_month))
        ),
        None,
    )
    if period_column is None:
        return None

    actual = usage[["Part Number", period_column]].copy()
    actual["Part Number"] = actual["Part Number"].astype("string").str.strip()
    actual["Actual_Demand"] = pd.to_numeric(
        actual[period_column], errors="coerce"
    )
    actual = (
        actual.dropna(subset=["Part Number", "Actual_Demand"])
        .groupby("Part Number", as_index=False)["Actual_Demand"]
        .sum()
    )

    selected["Forecast_Demand"] = pd.to_numeric(
        selected["Forecast_Demand"], errors="coerce"
    )
    matched = selected.merge(actual, on="Part Number", how="inner")
    matched = matched.dropna(subset=["Forecast_Demand", "Actual_Demand"])
    if matched.empty:
        return None

    error = matched["Forecast_Demand"] - matched["Actual_Demand"]
    mae = float(error.abs().mean())
    rmse = float((error.pow(2).mean()) ** 0.5)

    if mae <= REFERENCE_MAE * 1.10 and rmse <= REFERENCE_RMSE * 1.10:
        status = "STABLE"
        message = "Forecast performance remains within the expected range."
    elif mae <= REFERENCE_MAE * 1.25 and rmse <= REFERENCE_RMSE * 1.25:
        status = "MONITOR"
        message = "Forecast error has increased and should be monitored."
    else:
        status = "REVIEW MODEL"
        message = "Forecast error increased significantly; review the model."

    return {
        "year": int(actual_year),
        "month": int(actual_month),
        "mae": mae,
        "rmse": rmse,
        "matched_pn": int(len(matched)),
        "status": status,
        "message": message,
        "details": matched,
    }

DATA_FILES = {
    "usage_file": "usage_data.xlsx",
    "stock_file": "stock_data.xlsx",
    "policy_file": "stock_policy_data.xlsx",
    "watchlist_file": "watchlist_data.xlsx",
    "migo_file": "incoming_migo_data.xlsx",
    "backorder_file": "backorder_data.xlsx",
    "wrs_file": "wrs_data.xlsx",
}

# Expected refresh cycles for the seven internal data sources. The application
# uses each saved file's modification time as its last-update timestamp.
INTERNAL_DATA_INFO = {
    "usage_file": {
        "number": "01",
        "label": "Usage Data",
        "cadence": "Monthly",
        "days": 31,
        "description": (
            "Historical monthly demand by part. Expected columns: Part Number, "
            "Part Description, RANK Freq Call, MAD, and monthly usage columns "
            "numbered 1 through 12."
        ),
    },
    "stock_file": {
        "number": "02",
        "label": "Stock Data",
        "cadence": "Weekly",
        "days": 7,
        "description": (
            "Current available inventory by material and plant. Expected "
            "columns: Material, Available stock, and Plant."
        ),
    },
    "policy_file": {
        "number": "03",
        "label": "Stock Policy",
        "cadence": "Every 2–3 months",
        "days": 90,
        "description": (
            "Approved minimum and maximum stock levels. Expected columns: "
            "Part Number, Qty Min, and Qty Max."
        ),
    },
    "watchlist_file": {
        "number": "04",
        "label": "Watchlist",
        "cadence": "Every 2–3 months",
        "days": 90,
        "description": (
            "The list of parts included in the replenishment analysis. "
            "Expected column: Part Number."
        ),
    },
    "migo_file": {
        "number": "05",
        "label": "Incoming MIGO",
        "cadence": "Weekly",
        "days": 7,
        "description": (
            "Incoming supply recorded in MIGO. Expected columns: Material "
            "and Qty. Supply."
        ),
    },
    "backorder_file": {
        "number": "06",
        "label": "Backorder",
        "cadence": "Weekly",
        "days": 7,
        "description": (
            "Outstanding customer or operational demand not yet fulfilled. "
            "Expected columns: Material and Qty. Order."
        ),
    },
    "wrs_file": {
        "number": "07",
        "label": "WRS",
        "cadence": "Weekly",
        "days": 7,
        "description": (
            "Open WRS supply and lead-time information. Expected columns: "
            "Material and Qty. Order; Lead Time (Days) is optional."
        ),
    },
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

# Seed awal berasal dari seri PMI yang digunakan dan divalidasi di notebook.
# File tidak ditimpa lagi setelah dibuat, sehingga update user tetap tersimpan.
if not PMI_FILE.exists():
    default_pmi_data().to_excel(PMI_FILE, index=False)
if not EVENT_FILE.exists():
    default_event_calendar().to_excel(EVENT_FILE, index=False)


def migrate_project_inputs():
    """Copy existing project-folder inputs into permanent storage once."""
    old_data_dir = BASE_DIR / "data_input"
    if old_data_dir.resolve() == DATA_DIR.resolve() or not old_data_dir.exists():
        return

    for filename in DATA_FILES.values():
        old_file = old_data_dir / filename
        permanent_file = DATA_DIR / filename
        if old_file.exists() and not permanent_file.exists():
            shutil.copy2(old_file, permanent_file)


migrate_project_inputs()


def active_input_paths():
    """Return canonical paths for the seven active weekly inputs."""
    return {
        key: DATA_DIR / filename
        for key, filename in DATA_FILES.items()
    }


def save_input_updates(uploaded_files):
    """Atomically replace only uploaded inputs and archive older versions."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_batch = ARCHIVE_DIR / timestamp
    updated = []

    for key, uploaded_file in uploaded_files.items():
        if uploaded_file is None:
            continue

        destination = DATA_DIR / DATA_FILES[key]
        if destination.exists():
            archive_batch.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, archive_batch / destination.name)

        temporary = destination.with_suffix(".tmp")
        uploaded_file.seek(0)
        temporary.write_bytes(uploaded_file.getvalue())
        temporary.replace(destination)
        updated.append(destination.name)

    return updated


def queue_success(message):
    """Keep a success message visible after Streamlit reruns the page."""
    st.session_state["dprime_success_message"] = message


def show_error(message):
    """Show an unmistakable failure notification without hiding details."""
    st.toast(message, icon="❌")
    st.error(message)


def read_pmi_history(source=PMI_FILE):
    """Read and validate one monthly PMI observation per row."""
    pmi = pd.read_excel(source)
    required = {"Date", "PMI_Value"}
    missing = required.difference(pmi.columns)
    if missing:
        raise ValueError(
            "PMI data must contain Date and PMI_Value columns. "
            f"Missing columns: {', '.join(sorted(missing))}."
        )
    pmi = pmi[["Date", "PMI_Value"]].copy()
    pmi["Date"] = (
        pd.to_datetime(pmi["Date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    pmi["PMI_Value"] = pd.to_numeric(pmi["PMI_Value"], errors="coerce")
    if pmi.isna().any().any():
        raise ValueError("Date and PMI_Value must contain valid values.")
    if pmi["Date"].duplicated().any():
        raise ValueError("PMI data must contain only one value per month.")
    return pmi.sort_values("Date").reset_index(drop=True)


def save_pmi_history(uploaded_file):
    """Validate and merge uploaded PMI periods into active history."""
    pmi = read_pmi_history(uploaded_file)
    return merge_monthly_history(
        PMI_FILE, pmi, ["Date"], ["PMI_Value"]
    )


def merge_monthly_history(existing_file, update, key_columns, value_columns):
    """Merge new periods, update duplicates, and archive the previous file."""
    frames = []
    if existing_file.exists():
        frames.append(pd.read_excel(existing_file))
    frames.append(update)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(key_columns, keep="last")
    merged = merged.sort_values(key_columns).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if existing_file.exists():
        archive_batch = ARCHIVE_DIR / timestamp
        archive_batch.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing_file, archive_batch / existing_file.name)
    temporary = existing_file.with_suffix(".tmp.xlsx")
    merged[key_columns + value_columns].to_excel(temporary, index=False)
    temporary.replace(existing_file)
    return merged[key_columns + value_columns]


def parse_seasonality_upload(uploaded_file):
    """Accept compact history or raw Trend Parts workbook and aggregate monthly."""
    workbook = pd.ExcelFile(uploaded_file)
    first = pd.read_excel(uploaded_file, sheet_name=workbook.sheet_names[0])
    if {"Date", "Seasonal_Qty"}.issubset(first.columns):
        seasonal = first[["Date", "Seasonal_Qty"]].copy()
    elif "Master" in workbook.sheet_names:
        raw = pd.read_excel(
            uploaded_file,
            sheet_name="Master",
            usecols=["Product", "Posting Date", "Qty"],
        )
        raw["Product"] = raw["Product"].astype("string").str.upper().str.strip()
        toyota = raw[raw["Product"].eq("TOYOTA")].copy()
        if toyota.empty:
            raise ValueError("The Master sheet contains no TOYOTA product data.")
        toyota["Date"] = (
            pd.to_datetime(toyota["Posting Date"], errors="coerce")
            .dt.to_period("M").dt.to_timestamp()
        )
        toyota["Seasonal_Qty"] = pd.to_numeric(toyota["Qty"], errors="coerce")
        seasonal = toyota.groupby("Date", as_index=False)["Seasonal_Qty"].sum()
    else:
        raise ValueError(
            "Use a Trend Parts file with a Master sheet, or provide Date "
            "and Seasonal_Qty columns."
        )
    seasonal["Date"] = (
        pd.to_datetime(seasonal["Date"], errors="coerce")
        .dt.to_period("M").dt.to_timestamp()
    )
    seasonal["Seasonal_Qty"] = pd.to_numeric(
        seasonal["Seasonal_Qty"], errors="coerce"
    )
    seasonal = seasonal.dropna()
    if seasonal.empty:
        raise ValueError("Historical seasonality contains no valid rows.")
    return seasonal.groupby("Date", as_index=False)["Seasonal_Qty"].sum()


def parse_currency_upload(uploaded_file):
    """Accept compact currency history or the weekly Finance workbook."""
    first = pd.read_excel(uploaded_file)
    if {"Date", "Currency", "Rate"}.issubset(first.columns):
        currency = first[["Date", "Currency", "Rate"]].copy()
    else:
        raw_header = pd.read_excel(uploaded_file, header=None, nrows=3)
        date_text = " ".join(raw_header.astype(str).fillna("").values.ravel())
        date_match = re.search(r"\d{1,2}\s+[A-Za-z]{3}\s+\d{4}", date_text)
        finance = pd.read_excel(uploaded_file, header=3)
        finance.columns = [str(column).strip() for column in finance.columns]
        curr_col = next((c for c in finance if "Curr" in c), None)
        rate_col = next((c for c in finance if c == "Kurs Acuan"), None)
        if curr_col is None or rate_col is None or date_match is None:
            raise ValueError(
                "Currency data requires Date, Currency, and Rate columns, or "
                "the Finance Reference Rate format with a <3 Month forecast."
            )
        currency = finance[[curr_col, rate_col]].rename(
            columns={curr_col: "Currency", rate_col: "Rate"}
        )
        currency["Date"] = pd.to_datetime(date_match.group(), dayfirst=True)
    currency["Date"] = (
        pd.to_datetime(currency["Date"], errors="coerce")
        .dt.to_period("M").dt.to_timestamp()
    )
    currency["Currency"] = currency["Currency"].astype("string").str.upper().str.strip()
    currency["Rate"] = pd.to_numeric(currency["Rate"], errors="coerce")
    currency = currency.dropna()
    if currency.empty:
        raise ValueError("Currency data contains no valid rows.")
    return currency[["Date", "Currency", "Rate"]]


def parse_event_upload(uploaded_file):
    """Validate an editable calendar of events with moving annual dates."""
    events = pd.read_excel(uploaded_file)
    required = {"Event", "Event_Date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("The event calendar requires Event and Event_Date columns.")
    events = events[["Event", "Event_Date"]].copy()
    events["Event"] = events["Event"].astype("string").str.strip()
    events["Event_Date"] = pd.to_datetime(
        events["Event_Date"], errors="coerce"
    )
    events = events.dropna()
    if events.empty:
        raise ValueError("The event calendar contains no valid rows.")
    return events.drop_duplicates(["Event", "Event_Date"], keep="last")

st.set_page_config(
    page_title="D-PRIME",
    page_icon="📦",
    layout="wide",
)

success_message = st.session_state.pop("dprime_success_message", None)
if success_message:
    st.toast(success_message, icon="✅")
    st.success(success_message)


# CUSTOM CSS

st.markdown(
    """
    <style>
        :root {
            --navy: #18205d;
            --blue: #435991;
            --red: #ef2029;
            --ink: #171b2c;
            --muted: #697089;
        }

        .stApp {
            background: linear-gradient(
                180deg,
                #f4f6fb 0%,
                #fafbfe 44%,
                #f5f7fb 100%
            );
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .hero {
            padding: 18px 22px;
            border-radius: 24px;
            color: white;
            margin-bottom: 0px;
            text-align: center;

            background:
                radial-gradient(
                    circle at 84% 10%,
                    rgba(255, 255, 255, 0.13),
                    transparent 27%
                ),
                linear-gradient(
                    118deg,
                    #17205e,
                    #40568e 58%,
                    #263775
                );

            box-shadow:
                0 18px 42px rgba(24, 32, 93, 0.20);

            border:
                1px solid rgba(255, 255, 255, 0.15);
        }

        .hero-title {
            font-size: 24px;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 8px;
        }

        .dprime-name {
            color: #ffffff;
            font-size: 34px;
            font-weight: 900;
            letter-spacing: 4px;
            margin-bottom: 7px;

            text-shadow:
                0 0 12px rgba(239, 32, 41, 0.75);
        }

        .hero-copy {
            color: #e8ecff;
            font-size: 13px;
            line-height: 1.5;
            max-width: 680px;
            margin: 0 auto;
        }

        .section-title {
            font-size: 21px;
            font-weight: 850;
            color: var(--ink);
            margin: 8px 0 3px;
        }

        .section-copy {
            color: var(--muted);
            font-size: 14px;
            margin-bottom: 16px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 11px 15px;
            margin: 9px 0 18px;
            border-radius: 12px;
            background: white;
            border: 1px solid #e4e7f0;
            color: #646b7e;
            font-size: 13px;
        }

        .dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #21a365;
            box-shadow: 0 0 0 4px #dff5e9;
        }

        [data-testid="stFileUploader"] {
            background: white;
            border: 1px solid #e5e8f1;
            border-radius: 16px;
            padding: 13px 14px 5px;

            box-shadow:
                0 6px 18px rgba(27, 36, 83, 0.045);
        }

        [data-testid="stFileUploader"] label {
            color: #292f48;
            font-weight: 750;
        }

        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e6e8f0;
            border-radius: 17px;
            padding: 16px 18px;

            box-shadow:
                0 7px 20px rgba(27, 36, 83, 0.05);
        }

        [data-testid="stMetricValue"] {
            color: var(--navy);
            font-weight: 850;
        }

        div.stButton > button {
    min-height: 49px;
    border: 2px solid #ef2029;
    border-radius: 13px;
    background: linear-gradient(
        100deg,
        #f5202d,
        #ff414b
    );
    color: #111111 !important;
    font-size: 16px;
    font-weight: 900 !important;
    box-shadow:
        0 9px 21px rgba(239, 32, 41, 0.20);
    transition: all 0.2s ease;
    opacity: 1 !important;
}

div.stButton > button p {
    color: #111111 !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

div.stButton > button:hover {
    background: #ff5961;
    color: #111111 !important;
    border-color: #d81924;
    transform: translateY(-1px);
}

div.stButton > button:hover p {
    color: #111111 !important;
    font-weight: 900 !important;
}

div.stButton > button:active,
div.stButton > button:focus {
    background: #21a365 !important;
    border-color: #188651 !important;
    color: #111111 !important;
}

div.stButton > button:active p,
div.stButton > button:focus p {
    color: #111111 !important;
    font-weight: 900 !important;
}

div.stButton > button:disabled {
    background: #e1e3e8 !important;
    border-color: #b8bdc7 !important;
    color: #111111 !important;
    opacity: 1 !important;
    box-shadow: none;
}

div.stButton > button:disabled p {
    color: #111111 !important;
    font-weight: 900 !important;
    opacity: 1 !important;
}

        div.stDownloadButton > button {
            min-height: 48px;
            border: 1px solid #263775;
            border-radius: 13px;
            font-weight: 800;
            color: white;
            background: #263775;
        }

        .dprime-footer {
            margin-top: 48px;
            padding: 22px 18px;
            text-align: center;
            color: #697089;
            font-size: 13px;
            border-top: 1px solid #dfe3ee;
        }

        .dprime-footer strong {
            color: #18205d;
            font-weight: 850;
        }

        .dprime-footer a {
            display: inline-block;
            margin: 10px 5px 0;
            padding: 7px 13px;
            border: 1px solid #d7dcea;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.82);
            color: #263775 !important;
            font-weight: 750;
            text-decoration: none !important;
            transition: all 0.2s ease;
        }

        .dprime-footer a:hover {
            border-color: #ef2029;
            color: #ef2029 !important;
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# HEADER AND LOGOS
with st.container(border=True):
    logo_col, hero_col, partner_col = st.columns(
        [1.5, 3.6, 1.2],
        vertical_alignment="center",
    )

    with logo_col:
        dprime_logo = BASE_DIR / "assets" / "dprime_logo.png"
        if not dprime_logo.exists():
            dprime_logo = BASE_DIR / "dprime_logo.png"
        if dprime_logo.exists():
            st.image(dprime_logo, width=260)

    with hero_col:
        st.markdown(
            """<div class="hero">
                <div class="hero-title">Weekly Replenishment Dashboard</div>
                <div class="dprime-name">D-PRIME</div>
                <div class="hero-copy">
                    Demand Forecasting-Driven Predictive Replenishment
                    &amp; Inventory Management Engine
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with partner_col:
        partner_logo = BASE_DIR / "assets" / "traknus_logo.png"
        if not partner_logo.exists():
            partner_logo = BASE_DIR / "traknus_logo.png"
        if partner_logo.exists():
            st.image(partner_logo, width=145)


# INTERNAL DATA INPUT

st.markdown(
    '<div class="section-title">01 · Internal Data Input</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-copy">
        These seven operational files are saved locally and reused automatically.
        Upload only the files that have changed, then save the updates before
        running the analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

uploads = {}
stored_paths = active_input_paths()

first_row_keys = list(INTERNAL_DATA_INFO.keys())[:4]
second_row_keys = list(INTERNAL_DATA_INFO.keys())[4:]


def internal_file_status(key):
    """Return a plain-language freshness status for one saved input file."""
    path = stored_paths[key]
    info = INTERNAL_DATA_INFO[key]
    if not path.exists():
        return {
            "state": "missing",
            "label": "FILE REQUIRED",
            "last_updated": "Never",
            "due_text": "Upload the first active file.",
            "warning": f"Upload {info['label']} before running D-PRIME.",
        }

    updated_at = datetime.fromtimestamp(path.stat().st_mtime)
    due_at = updated_at + timedelta(days=info["days"])
    overdue = datetime.now() >= due_at
    return {
        "state": "overdue" if overdue else "current",
        "label": "UPDATE REQUIRED" if overdue else "UP TO DATE",
        "last_updated": updated_at.strftime("%d %b %Y, %H:%M"),
        "due_text": (
            f"Due since {due_at:%d %b %Y}"
            if overdue
            else f"Next update due: {due_at:%d %b %Y}"
        ),
        "warning": (
            f"Upload the latest {info['label']}. The saved file reached its "
            f"{info['cadence'].lower()} refresh date on {due_at:%d %b %Y}."
            if overdue else None
        ),
    }


def render_internal_uploader(key):
    """Render one internal uploader with cadence, columns, and freshness."""
    info = INTERNAL_DATA_INFO[key]
    status = internal_file_status(key)
    color = "#21a365" if status["state"] == "current" else "#d92d20"
    background = "#ecfdf3" if status["state"] == "current" else "#fff1f0"
    st.markdown(
        f"""<div style="border-left:4px solid {color};background:{background};
        border-radius:10px;padding:9px 11px;margin-bottom:8px;font-size:12px;">
        <strong style="color:{color};">● {status['label']}</strong><br>
        <span><strong>Refresh:</strong> {info['cadence']}</span><br>
        <span><strong>Last updated:</strong> {status['last_updated']}</span><br>
        <span>{status['due_text']}</span></div>""",
        unsafe_allow_html=True,
    )
    uploads[key] = st.file_uploader(
        f"{info['number']} · {info['label']}:",
        type=["xlsx"],
        key=key,
        help=(
            f"Purpose and required content: {info['description']} "
            f"Refresh schedule: {info['cadence']}. Saved file: "
            f"{stored_paths[key].name}."
        ),
    )
    if status["warning"]:
        st.warning(status["warning"], icon="⚠️")

first_row_columns = st.columns(4)

for column, key in zip(
    first_row_columns,
    first_row_keys,
):
    with column:
        render_internal_uploader(key)

second_row_columns = st.columns(3)

for column, key in zip(
    second_row_columns,
    second_row_keys,
):
    with column:
        render_internal_uploader(key)



# UPLOAD STATUS

ready = all(path.exists() for path in stored_paths.values())
stored_count = sum(path.exists() for path in stored_paths.values())
pending_count = sum(file is not None for file in uploads.values())

if ready:
    status_text = "All required files are available"
else:
    status_text = "Upload the missing required files"

st.markdown(
    f'<div class="status"><span class="dot"></span><b>{stored_count}/7 active files</b><span> · {pending_count} unsaved update(s) · {status_text}.</span></div>',
    unsafe_allow_html=True,
)
st.caption(f"Local storage location: {DATA_DIR}")

save_clicked = False
if pending_count > 0:
    _, save_column, _ = st.columns([1, 1.5, 1])
    with save_column:
        save_clicked = st.button(
            f"💾 Save {pending_count} Internal Data Update",
            width="stretch",
        )
else:
    st.info(
        "There are no pending internal-data updates. The saved active files "
        "will be reused when D-PRIME runs."
    )

if save_clicked:
    try:
        updated_files = save_input_updates(uploads)
        queue_success(
            "Internal data updated successfully: " + ", ".join(updated_files)
        )
        st.rerun()
    except Exception as exc:
        show_error(f"The internal data could not be saved: {exc}")

month_options = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

COMPARISON_MODE = "Comparison — D-PRIME vs G-Force"
comparison_label = None
download_filename = "DPRIME_Weekly_Output.xlsx"

st.markdown(
    '<div class="section-title">02 · External Intelligence Input</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-copy">Manage historical seasonality, PMI, and '
    'currency data. Every update is saved locally, while the previous version '
    'is retained in the archive.</div>',
    unsafe_allow_html=True,
)


def render_external_status(
    label,
    path,
    cadence,
    latest_period=None,
    maximum_age_days=None,
):
    """Show a consistent freshness indicator for an external data source."""
    if not path.exists():
        state = "missing"
        status_label = "DATA REQUIRED"
        last_updated = "Never"
        detail = "Upload the first active file."
        warning = f"Upload the latest {label} data before running D-PRIME."
    else:
        updated_at = datetime.fromtimestamp(path.stat().st_mtime)
        last_updated = updated_at.strftime("%d %b %Y, %H:%M")
        overdue = False

        if latest_period is not None:
            latest_month = pd.Timestamp(latest_period).to_period("M")
            required_month = (
                pd.Timestamp.today().to_period("M") - 1
            )
            overdue = latest_month < required_month
            detail = f"Latest data period: {latest_month.strftime('%B %Y')}"
            due_text = f"Expected period: {required_month.strftime('%B %Y')}"
        else:
            due_at = updated_at + timedelta(days=maximum_age_days)
            overdue = datetime.now() >= due_at
            detail = f"Next update due: {due_at:%d %b %Y}"
            due_text = f"Update due since: {due_at:%d %b %Y}"

        state = "overdue" if overdue else "current"
        status_label = "UPDATE REQUIRED" if overdue else "UP TO DATE"
        if overdue:
            detail = due_text
            warning = (
                f"Upload the latest {label} data. The current data no longer "
                f"meets its {cadence.lower()} refresh schedule."
            )
        else:
            warning = None

    color = "#21a365" if state == "current" else "#d92d20"
    background = "#ecfdf3" if state == "current" else "#fff1f0"
    st.markdown(
        f"""<div style="border-left:4px solid {color};background:{background};
        border-radius:10px;padding:9px 11px;margin-bottom:8px;font-size:12px;">
        <strong style="color:{color};">● {status_label}</strong><br>
        <span><strong>Refresh:</strong> {cadence}</span><br>
        <span><strong>Last uploaded:</strong> {last_updated}</span><br>
        <span>{detail}</span></div>""",
        unsafe_allow_html=True,
    )
    if warning:
        st.warning(warning, icon="⚠️")


if True:
    pmi_history = read_pmi_history()
    latest_pmi = pmi_history.iloc[-1]
    seasonal_active = (
        pd.read_excel(SEASONAL_FILE) if SEASONAL_FILE.exists() else None
    )
    external_upload_columns = st.columns(3)

    with external_upload_columns[0]:
        with st.container(border=True):
            st.markdown("#### 01 · Historical Seasonality")
            seasonal_latest_period = (
                pd.to_datetime(seasonal_active["Date"]).max()
                if seasonal_active is not None
                and "Date" in seasonal_active.columns
                else None
            )
            render_external_status(
                "Historical Seasonality",
                SEASONAL_FILE,
                "Annually",
                maximum_age_days=365,
            )
            seasonal_upload = st.file_uploader(
                "Upload or drag Trend Parts:",
                type=["xlsx", "xls"],
                key="seasonality_history_update",
                help=(
                    f"Active history: {len(seasonal_active)} months"
                    if seasonal_active is not None
                    else "No active data. Upload the first Trend Parts file."
                ),
            )
            if seasonal_upload is not None and st.button(
                "💾 Save / Merge Seasonality",
                key="save_seasonality_card",
                width="stretch",
            ):
                try:
                    update = parse_seasonality_upload(seasonal_upload)
                    saved = merge_monthly_history(
                        SEASONAL_FILE, update, ["Date"], ["Seasonal_Qty"]
                    )
                    queue_success(
                        f"Seasonality saved successfully: {len(saved)} months."
                    )
                    st.rerun()
                except Exception as exc:
                    show_error(f"Seasonality could not be saved: {exc}")

    with external_upload_columns[1]:
        with st.container(border=True):
            st.markdown("#### 02 · Monthly PMI")
            render_external_status(
                "PMI History",
                PMI_FILE,
                "Monthly",
                latest_period=latest_pmi["Date"],
            )
            pmi_upload = st.file_uploader(
                "Upload or drag PMI History:",
                type=["xlsx"],
                key="pmi_history_update",
                help=(
                    f"Active through {latest_pmi['Date']:%B %Y}: "
                    f"{latest_pmi['PMI_Value']:.1f}"
                ),
            )
            if pmi_upload is not None and st.button(
                "💾 Save / Merge PMI Excel",
                key="save_pmi_card",
                width="stretch",
            ):
                try:
                    saved = save_pmi_history(pmi_upload)
                    queue_success(f"PMI saved successfully: {len(saved)} months.")
                    st.rerun()
                except Exception as exc:
                    show_error(f"PMI could not be saved: {exc}")
            st.caption("Or enter one new monthly value:")
            pmi_manual_date = st.date_input(
                "PMI period:", value=datetime(2026, 8, 1), key="pmi_manual_date"
            )
            pmi_manual_value = st.number_input(
                "PMI value:", min_value=0.0, max_value=100.0,
                value=50.0, step=0.1, key="pmi_manual_value"
            )
            if st.button(
                "➕ Save / Update PMI Manual",
                key="save_manual_pmi",
                width="stretch",
            ):
                try:
                    update = pd.DataFrame({
                        "Date": [pd.Timestamp(pmi_manual_date)],
                        "PMI_Value": [pmi_manual_value],
                    })
                    saved = merge_monthly_history(
                        PMI_FILE, update, ["Date"], ["PMI_Value"]
                    )
                    queue_success(f"PMI saved successfully: {len(saved)} months.")
                    st.rerun()
                except Exception as exc:
                    show_error(f"PMI could not be saved: {exc}")

    with external_upload_columns[2]:
        with st.container(border=True):
            st.markdown("#### 03 · Reference Currency Rate")
            currency_status_data = (
                pd.read_excel(CURRENCY_FILE) if CURRENCY_FILE.exists() else None
            )
            currency_latest_period = (
                pd.to_datetime(currency_status_data["Date"]).max()
                if currency_status_data is not None
                and "Date" in currency_status_data.columns
                else None
            )
            render_external_status(
                "Currency History",
                CURRENCY_FILE,
                "Monthly",
                latest_period=currency_latest_period,
            )
            currency_upload = st.file_uploader(
                "Upload or drag Currency Rate Data:",
                type=["xlsx"],
                key="currency_history_update",
                help=(
                    "Active currency data is available."
                    if CURRENCY_FILE.exists()
                    else "No active currency data."
                ),
            )
            if currency_upload is not None and st.button(
                "💾 Save / Merge Currency Excel",
                key="save_currency_card",
                width="stretch",
            ):
                try:
                    update = parse_currency_upload(currency_upload)
                    saved = merge_monthly_history(
                        CURRENCY_FILE, update, ["Date", "Currency"], ["Rate"]
                    )
                    queue_success(f"Currency data saved successfully: {len(saved)} rows.")
                    st.rerun()
                except Exception as exc:
                    show_error(f"Currency data could not be saved: {exc}")
            st.caption("Or enter one new monthly value:")
            currency_date = st.date_input(
                "Currency period:", value=datetime(2026, 9, 1), key="currency_date"
            )
            currency_code = st.selectbox(
                "Currency:", ["USD", "JPY", "EUR"], key="currency_code"
            )
            currency_value = st.number_input(
                "Forecast rate (<3 months):", min_value=0.0, value=18150.0,
                step=1.0, key="currency_value"
            )
            if st.button(
                "➕ Save / Update Currency Manually",
                key="save_manual_currency",
                width="stretch",
            ):
                try:
                    update = pd.DataFrame({
                        "Date": [pd.Timestamp(currency_date)],
                        "Currency": [currency_code],
                        "Rate": [currency_value],
                    })
                    saved = merge_monthly_history(
                        CURRENCY_FILE, update, ["Date", "Currency"], ["Rate"]
                    )
                    queue_success(f"Currency data saved successfully: {len(saved)} rows.")
                    st.rerun()
                except Exception as exc:
                    show_error(f"Currency data could not be saved: {exc}")

    # EXTERNAL SIGNAL MONITOR
    st.markdown(
        '<div class="section-title">🧠 External Intelligence Center</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-copy">Review the active external signals before '
        'running a forecast. Visuals update automatically after data is saved '
        'or merged.</div>',
        unsafe_allow_html=True,
    )

    seasonal_ready = SEASONAL_FILE.exists()
    pmi_ready = PMI_FILE.exists()
    currency_ready = CURRENCY_FILE.exists()
    currency_period_count = (
        pd.to_datetime(pd.read_excel(CURRENCY_FILE)["Date"])
        .dt.to_period("M").nunique()
        if currency_ready else 0
    )
    readiness_columns = st.columns(3)
    readiness_columns[0].metric(
        "Historical Seasonality",
        "READY" if seasonal_ready else "MISSING",
        "Used by the model" if seasonal_ready else "Upload required",
    )
    readiness_columns[1].metric(
        "PMI History",
        "READY" if pmi_ready else "MISSING",
        "Used by the model" if pmi_ready else "Data required",
    )
    readiness_columns[2].metric(
        "Currency History",
        f"{currency_period_count}/12 MONTHS",
        (
            "Ready for feature validation"
            if currency_period_count >= 12
            else "Collecting validation history"
        ),
    )
    signal_tabs = st.tabs([
        "📈 Seasonal & Event Pulse",
        "🏭 PMI Momentum",
        "💱 Currency Watch",
    ])

    with signal_tabs[0]:
        if seasonal_ready:
            seasonal_chart = pd.read_excel(SEASONAL_FILE)
            seasonal_chart["Date"] = pd.to_datetime(seasonal_chart["Date"])
            event_chart = pd.read_excel(EVENT_FILE)
            event_chart["Event_Date"] = pd.to_datetime(event_chart["Event_Date"])
            event_chart["Month_Date"] = (
                event_chart["Event_Date"].dt.to_period("M").dt.to_timestamp()
            )
            event_points = event_chart.merge(
                seasonal_chart.rename(columns={"Date": "Month_Date"}),
                on="Month_Date",
                how="inner",
            )
            st.caption(
                "Historical part orders and calendar events are shown on one "
                "timeline. Hover over a colored point to view event details."
            )
            trend_line = alt.Chart(seasonal_chart).mark_line(
                point=True, color="#263775", strokeWidth=3
            ).encode(
                x=alt.X("Date:T", title="Period"),
                y=alt.Y("Seasonal_Qty:Q", title="Total Part Orders"),
                tooltip=[
                    alt.Tooltip("Date:T", title="Period", format="%B %Y"),
                    alt.Tooltip("Seasonal_Qty:Q", title="Total quantity", format=","),
                ],
            )
            event_marks = alt.Chart(event_points).mark_point(
                filled=True, size=130, stroke="white", strokeWidth=1
            ).encode(
                x="Month_Date:T",
                y="Seasonal_Qty:Q",
                color=alt.Color("Event:N", title="Calendar Event:"),
                tooltip=[
                    alt.Tooltip("Event:N", title="Event"),
                    alt.Tooltip("Event_Date:T", title="Date", format="%d %B %Y"),
                    alt.Tooltip("Seasonal_Qty:Q", title="Total quantity", format=","),
                ],
            )
            st.altair_chart(
                (trend_line + event_marks).interactive(),
                width="stretch",
            )
        else:
            st.warning(
                "Upload Trend Parts, then select Save / Merge to display the chart."
            )

    with signal_tabs[1]:
        if pmi_ready:
            pmi_chart = read_pmi_history().set_index("Date")
            latest_value = float(pmi_chart["PMI_Value"].iloc[-1])
            prior_value = (
                float(pmi_chart["PMI_Value"].iloc[-2])
                if len(pmi_chart) > 1 else latest_value
            )
            pmi_metric, pmi_chart_column = st.columns([1, 3])
            pmi_metric.metric(
                "Latest PMI",
                f"{latest_value:.1f}",
                f"{latest_value - prior_value:+.1f} vs previous month",
            )
            pmi_chart_column.line_chart(
                pmi_chart[["PMI_Value"]],
                x_label="Period",
                y_label="PMI Index",
                width="stretch",
            )

    with signal_tabs[2]:
        if currency_ready:
            currency_chart = pd.read_excel(CURRENCY_FILE)
            currency_chart["Date"] = pd.to_datetime(currency_chart["Date"])
            available_currencies = sorted(currency_chart["Currency"].unique())
            selected_currency = st.selectbox(
                "Select currency:",
                available_currencies,
                key="external_currency_chart",
            )
            selected_currency_data = currency_chart[
                currency_chart["Currency"].eq(selected_currency)
            ].sort_values("Date").set_index("Date")
            st.line_chart(
                selected_currency_data[["Rate"]],
                x_label="Period",
                y_label=f"{selected_currency} rate",
                width="stretch",
            )
            if currency_period_count < 12:
                st.info(
                    f"History currently covers {currency_period_count}/12 months. "
                    "Currency may be validated as a candidate feature after at "
                    "least 12 monthly snapshots; 18–24 months is preferred."
                )
        else:
            st.info("No currency history is available. This input remains optional.")

st.markdown(
    '<div class="section-title">03 · Model &amp; Analysis Configuration</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-copy">Select the analysis purpose and data period. '
    'D-PRIME will automatically use the appropriate engine.</div>',
    unsafe_allow_html=True,
)

run_mode = st.radio(
    "Select analysis mode:",
    options=[
        "Production",
        "Validation — July 2026",
        COMPARISON_MODE,
    ],
    horizontal=True,
)
use_hybrid = run_mode != "Validation — July 2026"

if use_hybrid:
    st.success(
        "Active model: Hybrid (TSB + Random Forest + XGBoost). Historical "
        "seasonality, calendar events, and PMI are included in the forecast."
    )
    if currency_period_count < 12:
        st.warning(
            f"Currency forecast: COLLECTING ({currency_period_count}/12 months). "
            "The data is saved but is not yet used by the model."
        )
    else:
        st.info(
            f"Currency forecast: READY FOR VALIDATION "
            f"({currency_period_count}/12 months). Currency will not be used "
            "until validation proves that it reduces MAE and RMSE."
        )
else:
    st.info(
        "Active model: TSB Baseline. This mode reproduces the historical "
        "checkpoint of 790 part numbers, 144 orders, and 1,473 units."
    )

if run_mode == "Production":
    month_column, year_column = st.columns(2)
    with month_column:
        selected_month_name = st.selectbox(
            "Latest actual usage month:",
            options=list(month_options.keys()),
            index=6,
        )
    with year_column:
        selected_actual_year = st.number_input(
            "Latest actual usage year:",
            min_value=2020,
            max_value=2100,
            value=2026,
            step=1,
            key="production_actual_year",
        )

    latest_period = month_options[
        selected_month_name
    ]

    next_period = (
        latest_period % 12
    ) + 1

    next_month_name = list(
        month_options.keys()
    )[next_period - 1]

    target_year = int(selected_actual_year) + (1 if latest_period == 12 else 0)
    st.info(
        f"Latest history: {selected_month_name} {int(selected_actual_year)}; "
        f"forecast target: {next_month_name} {target_year}."
    )

    dprime_config = HybridConfig(
        forecast_holdout_period=None,
        latest_period=latest_period,
        latest_year=int(selected_actual_year),
    )

elif run_mode == "Validation — July 2026":
    dprime_config = DPrimeConfig(
        forecast_holdout_period=7,
        latest_period=6,
    )

    st.info(
        "Validation uses history through June to reproduce the July 2026 forecast."
    )

else:
    period_column, year_column, week_column = st.columns(3)
    with period_column:
        selected_month_name = st.selectbox(
            "Latest actual usage month:",
            options=list(month_options.keys()),
            index=6,
            key="comparison_actual_month",
        )
    with year_column:
        selected_actual_year = st.number_input(
            "Latest actual usage year:",
            min_value=2020,
            max_value=2100,
            value=2026,
            step=1,
            key="comparison_actual_year",
        )
    with week_column:
        selected_week = st.selectbox(
            "G-Force week:",
            options=["W1", "W2", "W3", "W4", "W5"],
            index=2,
            key="comparison_week",
        )

    latest_period = month_options[selected_month_name]
    target_period = (latest_period % 12) + 1
    target_month_name = list(month_options.keys())[target_period - 1]
    target_year = int(selected_actual_year) + (1 if latest_period == 12 else 0)
    comparison_label = (
        f"{selected_week} {target_month_name} {target_year}"
    )

    dprime_config = HybridConfig(
        forecast_holdout_period=None,
        latest_period=latest_period,
        latest_year=int(selected_actual_year),
    )
    st.info(
        f"Usage history ends in {selected_month_name} {int(selected_actual_year)}; "
        f"the target is {target_month_name} {target_year}. Confirm that all seven "
        f"active files and the G-Force output use the same {comparison_label} "
        "snapshot."
    )
    gforce_upload = st.file_uploader(
        f"Upload G-Force output for {comparison_label}:",
        type=["xlsx"],
        key="gforce_comparison_file",
        help=(
            "The application reads the Plan sheet and the PN and Suggested Order columns."
        ),
    )
    safe_period_label = comparison_label.replace(" ", "_")
    download_filename = f"DPRIME_Comparison_{safe_period_label}.xlsx"

# RUN BUTTON

left_space, center_button, right_space = st.columns(
    [1, 1.5, 1]
)

with center_button:
    comparison_ready = (
        run_mode != COMPARISON_MODE
        or gforce_upload is not None
    )
    external_ready = (
        not use_hybrid
        or (SEASONAL_FILE.exists() and PMI_FILE.exists() and EVENT_FILE.exists())
    )
    run_clicked = st.button(
        "⚡ Run D-PRIME Analysis",
        type="primary",
        disabled=not ready or not comparison_ready or not external_ready,
        width="stretch",
    )

# RUN D-PRIME ENGINE

if run_clicked:
    try:
        with st.spinner(
            "D-PRIME is calculating replenishment recommendations..."
        ):
            weekly_inputs = load_weekly_inputs(**stored_paths)

            if use_hybrid:
                result = run_dprime_hybrid(
                    weekly_inputs,
                    pmi_data=read_pmi_history(),
                    seasonal_data=pd.read_excel(SEASONAL_FILE),
                    event_data=pd.read_excel(EVENT_FILE),
                    config=dprime_config,
                )
                forecast_engine_name = (
                    "Hybrid — Seasonality + Event Calendar + PMI"
                )
            else:
                result = run_dprime(
                    weekly_inputs,
                    config=dprime_config,
                )
                forecast_engine_name = "TSB Baseline"

            master_result = load_part_master()
            operational_result = build_operational_output(master_result, result)

            model_evaluation = None
            if run_mode == "Production":
                # First evaluate the forecast previously made for the latest
                # actual month, then save the new forecast for next month.
                model_evaluation = evaluate_saved_forecast(
                    weekly_inputs["usage"],
                    actual_year=int(selected_actual_year),
                    actual_month=int(latest_period),
                )
                save_forecast_snapshot(
                    result,
                    target_year=int(target_year),
                    target_month=int(next_period),
                    engine_name=forecast_engine_name,
                )

            comparison_result = None
            if run_mode == COMPARISON_MODE:
                comparison_result = build_gforce_comparison(
                    result,
                    gforce_upload,
                )

            st.session_state["result"] = result
            st.session_state["operational_result"] = operational_result
            st.session_state["comparison_result"] = comparison_result
            st.session_state["comparison_label"] = comparison_label
            st.session_state["download_filename"] = download_filename
            st.session_state["usage_history"] = weekly_inputs["usage"].copy()
            st.session_state["forecast_history_end"] = (
                6 if run_mode == "Validation — July 2026" else latest_period
            )
            st.session_state["forecast_latest_year"] = (
                2026
                if run_mode == "Validation — July 2026"
                else int(selected_actual_year)
            )
            st.session_state["analysis_mode"] = run_mode
            st.session_state["forecast_engine_name"] = forecast_engine_name
            st.session_state["model_evaluation"] = model_evaluation

        run_success_message = (
            "Analysis completed successfully. Replenishment recommendations are ready."
        )
        st.toast(run_success_message, icon="✅")
        st.success(run_success_message)

    except Exception as exc:
        show_error(f"D-PRIME analysis failed: {exc}")


# RESULT DASHBOARD

if (
    "result" in st.session_state
    and st.session_state.get("analysis_mode") == run_mode
):
    result = st.session_state["result"]
    operational_result = st.session_state["operational_result"]

    summary = regression_summary(
        result
    )

    st.caption(
        "Forecast engine: "
        + st.session_state.get("forecast_engine_name", "D-PRIME")
    )

    st.markdown(
        '<div class="section-title">📊 Executive Summary</div>',
        unsafe_allow_html=True,
    )

    metric_items = [
        ("Total Part Number", "total_pn", 0),
        ("Order", "order", 0),
        ("No Order", "no_order", 0),
        ("Review Required", "review_required", 0),
        ("Suggested Units", "total_suggested_order", 0),
        ("Avg Units / Order PN", "average_units_per_order_pn", 2),
    ]

    summary["average_units_per_order_pn"] = (
        summary["total_suggested_order"] / summary["order"]
        if summary["order"] else 0
    )

    metric_columns = st.columns(6)

    for column, (label, key, decimals) in zip(
        metric_columns,
        metric_items,
    ):
        if decimals:
            value = f"{summary[key]:,.{decimals}f}"
            value = value.replace(",", "_").replace(".", ",").replace("_", ".")
        else:
            value = f"{summary[key]:,}".replace(",", ".")

        column.metric(
            label,
            value,
        )

    if run_mode == "Production":
        st.markdown(
            '<div class="section-title">📈 Automatic Model Evaluation</div>',
            unsafe_allow_html=True,
        )
        model_evaluation = st.session_state.get("model_evaluation")

        if model_evaluation is None:
            st.info(
                "The current forecast has been saved automatically. MAE and "
                "RMSE will appear after the Usage Data for its target month "
                "is uploaded and Production is run again."
            )
        else:
            month_label = list(month_options.keys())[
                model_evaluation["month"] - 1
            ]
            evaluation_columns = st.columns(4)
            evaluation_columns[0].metric(
                "Evaluation Period",
                f"{month_label} {model_evaluation['year']}",
            )
            evaluation_columns[1].metric(
                "MAE",
                f"{model_evaluation['mae']:.4f}",
                delta=(
                    f"{model_evaluation['mae'] - REFERENCE_MAE:+.4f} "
                    "vs reference"
                ),
                delta_color="inverse",
            )
            evaluation_columns[2].metric(
                "RMSE",
                f"{model_evaluation['rmse']:.4f}",
                delta=(
                    f"{model_evaluation['rmse'] - REFERENCE_RMSE:+.4f} "
                    "vs reference"
                ),
                delta_color="inverse",
            )
            evaluation_columns[3].metric(
                "Matched Part Numbers",
                f"{model_evaluation['matched_pn']:,}".replace(",", "."),
            )

            status_message = (
                f"{model_evaluation['status']}: "
                f"{model_evaluation['message']}"
            )

            evaluation_details = model_evaluation["details"].copy()
            evaluation_details["Absolute_Error"] = (
                evaluation_details["Forecast_Demand"]
                - evaluation_details["Actual_Demand"]
            ).abs()
            evaluation_details = evaluation_details.sort_values(
                "Absolute_Error", ascending=False
            )
            review_count = max(
                10,
                int(len(evaluation_details) * 0.10),
            )
            high_error_pn = evaluation_details.head(review_count).copy()

            if model_evaluation["status"] == "STABLE":
                st.success(status_message)
                st.info(
                    "What to do: Continue the regular Production workflow. "
                    "The forecast remains consistent and no model action is required."
                )
            elif model_evaluation["status"] == "MONITOR":
                st.warning(status_message)
                st.markdown(
                    """
                    **What to do:**

                    1. Confirm that the latest Usage Data and selected month are correct.
                    2. Download and review the Part Numbers with the largest forecast differences.
                    3. Check whether unusual demand, one-time transactions, or unrecorded events occurred.
                    4. D-PRIME may still be used, with additional attention to the review list.
                    5. Check the evaluation status again in the next period.
                    """
                )
            else:
                st.error(status_message)
                st.markdown(
                    """
                    **What to do before finalizing replenishment:**

                    1. Confirm that all input files and selected periods are correct.
                    2. Download and review the Part Numbers with the largest forecast differences.
                    3. Check for major changes in demand, seasonality, calendar events, or PMI.
                    4. Use the recommendation as decision support and require manual approval.
                    5. Contact the model owner or data analyst for further model evaluation.
                    """
                )

            if model_evaluation["status"] in {"MONITOR", "REVIEW MODEL"}:
                safe_status = model_evaluation["status"].replace(" ", "_")
                st.download_button(
                    "⬇ Download High-Error PN Review List",
                    data=high_error_pn[
                        [
                            "Part Number",
                            "Forecast_Demand",
                            "Actual_Demand",
                            "Absolute_Error",
                        ]
                    ].to_csv(index=False).encode("utf-8-sig"),
                    file_name=(
                        f"DPRIME_{safe_status}_{month_label}_"
                        f"{model_evaluation['year']}.csv"
                    ),
                    mime="text/csv",
                    width="stretch",
                )
                st.caption(
                    f"The review list contains the {review_count} Part Numbers "
                    "with the highest absolute forecast errors."
                )

            with st.expander("View forecast vs. actual details"):
                st.dataframe(
                    evaluation_details[
                        [
                            "Part Number",
                            "Forecast_Demand",
                            "Actual_Demand",
                            "Absolute_Error",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

            with st.expander("What do these metrics mean?"):
                st.markdown(
                    """
                    - **MAE:** The average unit difference between forecast and actual demand. Lower is better.
                    - **RMSE:** A forecast error measure that gives more attention to large differences. Lower is better.
                    - **Absolute Error:** The unit difference between forecast and actual demand for one Part Number.
                    - **Backtesting:** Testing a forecasting model against historical actual demand before changing the Production model.

                    **Responsibility:** Dashboard users validate inputs and review affected Part Numbers. Model changes and backtesting are handled by the model owner or data analyst.
                    """
                )

            st.caption(
                "Status thresholds use the validated hybrid reference "
                f"(MAE {REFERENCE_MAE:.4f}; RMSE {REFERENCE_RMSE:.4f}). "
                "STABLE is within 10%; MONITOR is within 25%; values above "
                "that are marked REVIEW MODEL."
            )

    impact_columns = {
        "Internal_Forecast",
        "External_Forecast",
        "External_Impact_Units",
        "External_Impact_Pct",
        "Forecast_Model",
    }
    if impact_columns.issubset(result.columns):
        st.markdown(
            '<div class="section-title">🌐 External Factor Impact</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-copy">Compare the internal-demand forecast '
            'with the forecast that includes seasonality, calendar events, '
            'and lagged PMI. External features are currently applied only to '
            'Erratic Part Numbers routed to XGBoost.</div>',
            unsafe_allow_html=True,
        )

        external_impact = result[result["Forecast_Model"].eq("XGBoost")].copy()
        if external_impact.empty:
            st.info(
                "No Erratic Part Numbers were routed to XGBoost in this run. "
                "External impact is therefore zero for the current scope."
            )
        else:
            impact_units = external_impact["External_Impact_Units"].fillna(0)
            impact_metrics = [
                ("Forecast Increased", int(impact_units.gt(0).sum())),
                ("Net Forecast Impact", float(impact_units.sum())),
            ]
            for column, (label, value) in zip(
                st.columns(2), impact_metrics
            ):
                formatted = (
                    f"{value:+,.2f}" if label == "Net Forecast Impact"
                    else f"{int(value):,}"
                )
                column.metric(label, formatted)

            impact_table_columns = [
                "Part Number",
                "Part Description",
                "Demand_Pattern",
                "Forecast_Model",
                "Internal_Forecast",
                "External_Forecast",
                "External_Impact_Units",
                "External_Impact_Pct",
            ]
            impact_table_columns = [
                column for column in impact_table_columns
                if column in external_impact.columns
            ]
            impact_table = external_impact[impact_table_columns].copy()
            impact_table["_Absolute_Impact"] = impact_table[
                "External_Impact_Units"
            ].abs()
            impact_table = impact_table.sort_values(
                "_Absolute_Impact", ascending=False
            ).drop(columns="_Absolute_Impact")

            top_external_impact = external_impact.assign(
                _Absolute_Impact=impact_units.abs()
            ).nlargest(15, "_Absolute_Impact")
            st.markdown("#### Top 15 External Forecast Impacts")
            st.bar_chart(
                top_external_impact[
                    ["Part Number", "External_Impact_Units"]
                ],
                x="Part Number",
                y="External_Impact_Units",
                width="stretch",
            )

            with st.expander("View external impact by Part Number", expanded=True):
                st.dataframe(
                    impact_table,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Internal_Forecast": st.column_config.NumberColumn(
                            "Internal Forecast", format="%.2f"
                        ),
                        "External_Forecast": st.column_config.NumberColumn(
                            "External Forecast", format="%.2f"
                        ),
                        "External_Impact_Units": st.column_config.NumberColumn(
                            "Impact (Units)", format="%+.2f"
                        ),
                        "External_Impact_Pct": st.column_config.NumberColumn(
                            "Impact (%)", format="%+.2f%%"
                        ),
                    },
                )
            st.caption(
                "Impact shows model sensitivity, not proof that an external "
                "factor caused the demand change. Currency is not included."
            )

    comparison = st.session_state.get("comparison_result")
    if comparison is not None:
        result_comparison_label = st.session_state.get(
            "comparison_label", "selected period"
        )
        st.markdown(
            '<div class="section-title">⚖️ D-PRIME vs G-Force</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-copy">Replenishment comparison based on '
            f'Suggested Order by Part Number for '
            f'{result_comparison_label}.</div>',
            unsafe_allow_html=True,
        )

        dprime_qty = comparison["DPRIME_Suggested_Order"]
        gforce_qty = comparison["GForce_Suggested_Order"]
        comparison_metrics = [
            ("D-PRIME Order PN", int(dprime_qty.gt(0).sum())),
            ("G-Force Order PN", int(gforce_qty.gt(0).sum())),
            ("D-PRIME Units", int(dprime_qty.sum())),
            ("G-Force Units", int(gforce_qty.sum())),
        ]
        for column, (label, value) in zip(
            st.columns(4), comparison_metrics
        ):
            column.metric(label, f"{value:,}".replace(",", "."))

        status_counts = comparison["Comparison_Status"].value_counts()
        status_metrics = [
            "BOTH ORDER", "D-PRIME ONLY",
            "G-FORCE ONLY", "BOTH NO ORDER",
        ]
        for column, status in zip(st.columns(4), status_metrics):
            column.metric(status, int(status_counts.get(status, 0)))

        selected_statuses = st.multiselect(
            "Filter comparison status:",
            options=status_metrics,
            default=["BOTH ORDER", "D-PRIME ONLY", "G-FORCE ONLY"],
            key="comparison_status_filter",
        )
        comparison_shown = (
            comparison[
                comparison["Comparison_Status"].isin(selected_statuses)
            ]
            if selected_statuses
            else comparison
        )
        st.dataframe(
            comparison_shown,
            width="stretch",
            hide_index=True,
            height=430,
        )
        st.caption(
            "Comparison mode compares replenishment recommendations; it does "
            "not calculate G-Force forecast MAE or RMSE."
        )


    # VISUAL ANALYTICS

    st.markdown(
        '<div class="section-title">📈 Visual Analytics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-copy">Use these visuals to understand decisions, '
        'demand patterns, and replenishment priorities more quickly.</div>',
        unsafe_allow_html=True,
    )

    decision_column, pattern_column = st.columns(2)

    with decision_column:
        st.markdown("#### Decision Distribution")
        decision_chart = (
            result["Decision"]
            .value_counts()
            .rename_axis("Decision")
            .reset_index(name="Total PN")
        )
        st.bar_chart(
            decision_chart,
            x="Decision",
            y="Total PN",
            color="Decision",
            width="stretch",
        )

    with pattern_column:
        st.markdown("#### Demand Pattern Distribution")
        pattern_chart = (
            result["Demand_Pattern"]
            .value_counts()
            .rename_axis("Demand Pattern")
            .reset_index(name="Total PN")
        )
        st.bar_chart(
            pattern_chart,
            x="Demand Pattern",
            y="Total PN",
            color="Demand Pattern",
            width="stretch",
        )

    st.markdown("#### Top 15 Suggested Order")
    top_orders = (
        result[result["Decision"].eq("ORDER")]
        .nlargest(15, "Suggested_Order")
        [["Part Number", "Suggested_Order"]]
    )
    if top_orders.empty:
        st.info("No part numbers are currently recommended for ordering.")
    else:
        st.bar_chart(
            top_orders,
            x="Part Number",
            y="Suggested_Order",
            width="stretch",
        )

    st.markdown("#### Demand History & Next-Month Forecast")
    chart_pn_options = result["Part Number"].dropna().astype(str).tolist()
    selected_chart_pn = st.selectbox(
        "Select a Part Number to view its trend:",
        options=chart_pn_options,
        key="forecast_chart_pn",
    )

    usage_history = st.session_state.get("usage_history")
    if usage_history is not None and selected_chart_pn:
        usage_chart = usage_history.copy()
        usage_chart["Part Number"] = (
            usage_chart["Part Number"].astype("string").str.strip()
        )
        selected_usage = usage_chart[
            usage_chart["Part Number"].eq(selected_chart_pn)
        ]
        period_columns = [
            column for column in usage_chart.columns
            if str(column).strip().isdigit()
        ]

        if not selected_usage.empty and period_columns:
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }
            history_end = st.session_state.get("forecast_history_end", 7)
            history_end_year = st.session_state.get(
                "forecast_latest_year", 2026
            )
            target_period = (history_end % 12) + 1
            validation_mode = (
                st.session_state.get("analysis_mode")
                == "Validation — July 2026"
            )
            usable_periods = [
                int(str(column).strip()) for column in period_columns
                if not (validation_mode and int(str(column).strip()) == 7)
            ]
            first_period = target_period
            ordered_periods = sorted(
                usable_periods,
                key=lambda period: (period - first_period) % 12,
            )
            values_by_period = {
                int(str(column).strip()): pd.to_numeric(
                    selected_usage.iloc[0][column], errors="coerce"
                )
                for column in period_columns
            }
            forecast_value = result.loc[
                result["Part Number"].astype(str).eq(selected_chart_pn),
                "Forecast_Demand",
            ].iloc[0]

            actual_period_labels = []
            for period in ordered_periods:
                period_year = (
                    history_end_year
                    if period <= history_end
                    else history_end_year - 1
                )
                actual_period_labels.append(
                    f"{month_names[period]} {period_year}"
                )

            target_year = (
                history_end_year + 1
                if history_end == 12
                else history_end_year
            )
            forecast_period_label = (
                f"{month_names[target_period]} {target_year} Forecast"
            )
            period_order = actual_period_labels + [forecast_period_label]
            trend = pd.DataFrame({
                "Period": period_order,
                "Demand": [values_by_period[p] for p in ordered_periods]
                + [forecast_value],
                "Data Type": ["Actual"] * len(ordered_periods)
                + ["D-PRIME Forecast"],
            })

            trend_chart = (
                alt.Chart(trend)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(
                        "Period:N",
                        sort=period_order,
                        title="Period",
                        axis=alt.Axis(labelAngle=-45),
                    ),
                    y=alt.Y("Demand:Q", title="Demand"),
                    color=alt.Color(
                        "Data Type:N",
                        title="Data Type",
                        scale=alt.Scale(
                            domain=["Actual", "D-PRIME Forecast"],
                            range=["#126BC5", "#7FC0EE"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("Period:N", title="Period"),
                        alt.Tooltip("Data Type:N", title="Data Type"),
                        alt.Tooltip("Demand:Q", title="Demand", format=",.2f"),
                    ],
                )
                .properties(height=340)
            )
            st.altair_chart(
                trend_chart,
                width="stretch",
            )
            st.caption(
                "The chart shows the next-month forecast produced by the "
                "validated D-PRIME model."
            )
        else:
            st.info("No usage history was found for this Part Number.")


    # RECOMMENDATION TABLE

    st.markdown(
        '<div class="section-title">🔎 Recommendation Detail</div>',
        unsafe_allow_html=True,
    )

    selected_decisions = st.multiselect(
        "Filter decisions:",
        options=[
            "ORDER",
            "NO ORDER",
            "REVIEW REQUIRED",
        ],
        default=[
            "ORDER",
            "REVIEW REQUIRED",
        ],
    )

    if selected_decisions:
        shown_result = operational_result[
            operational_result["Decision"].isin(
                selected_decisions
            )
        ]
    else:
        shown_result = operational_result

    st.dataframe(
        shown_result,
        width="stretch",
        hide_index=True,
        height=480,
    )


    # EXPORT TO EXCEL
    

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        operational_result.to_excel(
            writer,
            sheet_name="Final Recommendation",
            index=False,
        )

        operational_result[
            operational_result["Decision"].eq("ORDER")
        ].to_excel(
            writer,
            sheet_name="Order Only",
            index=False,
        )

        operational_result[
            operational_result["Decision"].eq(
                "REVIEW REQUIRED"
            )
        ].to_excel(
            writer,
            sheet_name="Review Required",
            index=False,
        )

        if comparison is not None:
            comparison.to_excel(
                writer,
                sheet_name="Comparison",
                index=False,
            )

    st.download_button(
        label="⬇ Download D-PRIME Weekly Output",
        data=output.getvalue(),
        file_name=st.session_state.get(
            "download_filename", "DPRIME_Weekly_Output.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        width="stretch",
    )

elif "result" in st.session_state:
    st.info(
        "The analysis mode has changed. Select Run D-PRIME Analysis to display "
        "results for the current mode."
    )


# FOOTER

st.markdown(
    """
    <div class="dprime-footer">
        © 2026 D-PRIME · Developed by
        <strong>Azaria Syahla Fitan Adibah</strong><br>
        <a href="https://www.linkedin.com/in/azariasyahla/"
           target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <a href="https://www.instagram.com/azariasyahla/"
           target="_blank" rel="noopener noreferrer">Instagram</a>
    </div>
    """,
    unsafe_allow_html=True,
)
