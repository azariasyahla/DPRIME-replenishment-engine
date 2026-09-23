# D-PRIME

**Demand Forecasting-Driven Predictive Replenishment & Inventory Management Engine**

D-PRIME is a Streamlit-based decision-support application for monthly demand
forecasting and weekly spare-parts replenishment. It combines demand patterns,
inventory position, incoming supply, lead time, stock policy, and validated
external signals to produce transparent order recommendations.

> Internal project. Operational company data is intentionally excluded from
> this repository.

## Main capabilities

- Classifies demand as Smooth, Erratic, Intermittent, Lumpy, or No Demand.
- Routes forecasting through TSB, Random Forest, or XGBoost according to the
  active analysis configuration.
- Calculates lead-time demand, safety stock, inventory position, suggested
  order quantity, and decision reasons.
- Supports Production, Validation, and G-FORCE Comparison modes.
- Stores active inputs locally and archives replaced versions.
- Monitors forecast accuracy using MAE and RMSE once actual demand is available.
- Exports complete recommendations to Excel.

## Repository structure

```text
app.py                       Streamlit dashboard
dprime_engine.py             Stable standalone replenishment engine
dprime_engine_hybrid.py      Pattern-based hybrid forecast engine
test_dprime_engine.py        Engine regression test
verify_checkpoint.py         Optional historical checkpoint verification
docs/                        Methodology flowcharts
```

Runtime data folders and operational spreadsheets are ignored by Git and must
never be committed.

## Local setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open <http://localhost:8501> after Streamlit starts.

## Internal data inputs

D-PRIME uses seven operational workbooks:

1. Usage Data
2. Stock Data
3. Stock Policy
4. Watchlist
5. Incoming MIGO
6. Backorder
7. WRS

On Windows, active data is stored outside the repository under:

```text
%LOCALAPPDATA%\DPRIME\
```

This location contains active input, archive, external input, and model
monitoring files. Do not copy these folders into the repository.

## Optional local files

The application can run without the following private or brand-controlled
files. When authorized, they may be added only to the local working copy:

```text
assets/dprime_logo.png
assets/traknus_logo.png
master_data/master_part.xlsx
```

`master_part.xlsx` must contain a sheet named `Part Master` and a `PN` column.
If it is absent, D-PRIME still produces its native recommendation columns but
does not append optional SAP/EPC master attributes.

## Testing

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For historical checkpoint verification, provide the seven authorized input
files locally:

```powershell
python verify_checkpoint.py C:\path\to\authorized\input_folder
```

## Validated checkpoint

The historical reproduction checkpoint used during development produced:

- 790 Part Numbers
- 144 ORDER
- 507 NO ORDER
- 139 REVIEW REQUIRED
- 1,473 total suggested units

These values are a reproducibility checkpoint, not a guarantee for future
periods. Future recommendations change when demand and inventory inputs change.

## Data protection

Do not commit SAP exports, active inputs, archived versions, G-FORCE outputs,
forecast monitoring history, credentials, or generated recommendation files.
Review `git status` before every commit.

## Author

Developed by **Azaria Syahla Fitan Adibah** in 2026.

- [LinkedIn](https://www.linkedin.com/in/azariasyahla/)
- [Instagram](https://www.instagram.com/azariasyahla/)

