"""Run the frozen notebook checkpoint against a directory containing seven inputs."""

from pathlib import Path

from dprime_engine import (
    DPrimeConfig,
    load_weekly_inputs,
    regression_summary,
    run_dprime,
)

EXPECTED = {"total_pn": 790, "order": 144, "no_order": 507,
            "review_required": 139, "total_suggested_order": 1473}


def verify(base_dir: str | Path) -> dict[str, int]:
    base = Path(base_dir)
    inputs = load_weekly_inputs(
        usage_file=base / "usage_data.xlsx",
        stock_file=base / "stock_data.xlsx",
        policy_file=base / "stock_policy_data.xlsx",
        watchlist_file=base / "watchlist_data.xlsx",
        migo_file=base / "incoming_migo_data.XLSX",
        backorder_file=base / "backorder_data.XLSX",
        wrs_file=base / "wrs_data.XLSX",
    )
    actual = regression_summary(run_dprime(inputs))
    assert actual == EXPECTED, f"checkpoint mismatch: expected={EXPECTED}, actual={actual}"
    return actual


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    args = parser.parse_args()
    print(verify(args.input_directory))
