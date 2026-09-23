import pandas as pd

from dprime_engine import regression_summary, run_dprime


def test_standalone_engine_decisions():
    usage = pd.DataFrame({"Part Number": ["A", "B", "C"], "Description": ["a", "b", "c"],
                          "RANK Freq Call": ["A", "D", "E"], "MAD": [1, 0, 1],
                          **{i: [1, 0, 1 if i in (1, 6) else 0] for i in range(1, 13)}})
    inputs = {
        "usage": usage,
        "stock": pd.DataFrame({"Material": ["A", "B"], "Available stock": [0, 10], "Plant": ["HDO", "HDO"]}),
        "policy": pd.DataFrame({"Part Number": ["A", "B", "C"], "Qty Min": [0, 0, 0], "Qty Max": [0, 0, 0]}),
        "watchlist": pd.DataFrame({"Part Number": ["A", "B", "C"]}),
        "migo": pd.DataFrame({"Material": ["A"], "Qty. Supply": [0]}),
        "backorder": pd.DataFrame({"Material": ["A"], "Qty. Order": [0]}),
        "wrs": pd.DataFrame({"Material": ["A"], "Qty. Order": [0], "Lead Time (Days)": [60]}),
    }
    result = run_dprime(inputs)
    assert result["Decision"].tolist() == ["ORDER", "NO ORDER", "REVIEW REQUIRED"]
    assert regression_summary(result)["total_pn"] == 3
