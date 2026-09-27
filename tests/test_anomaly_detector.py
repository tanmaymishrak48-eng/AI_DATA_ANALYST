"""
Unit tests for anomaly_detector.py — the IQR-based anomaly detection
layer. Pure pandas/numpy statistics, no API calls, always runs offline.
"""

import pandas as pd
import pytest

from anomaly_detector import (
    detect_numerical_anomalies,
    get_anomaly_summary,
    get_anomalous_rows,
)


def test_detect_numerical_anomalies_returns_dataframe(sample_df):
    result = detect_numerical_anomalies(sample_df)
    assert isinstance(result, pd.DataFrame)
    expected_columns = {
        "Row_Index",
        "Column",
        "Value",
        "Lower_Bound",
        "Upper_Bound",
        "Reason",
    }
    assert expected_columns.issubset(set(result.columns))


def test_detected_values_are_actually_outside_bounds(sample_df):
    result = detect_numerical_anomalies(sample_df)
    for _, row in result.iterrows():
        assert (row["Value"] < row["Lower_Bound"]) or (
            row["Value"] > row["Upper_Bound"]
        )


def test_no_numeric_columns_returns_empty_dataframe():
    df = pd.DataFrame({"Name": ["a", "b", "c"], "City": ["X", "Y", "Z"]})
    result = detect_numerical_anomalies(df)
    assert result.empty


def test_constant_column_produces_no_anomalies():
    # Zero IQR (all values identical) must not raise or flag anything.
    df = pd.DataFrame({"Value": [5, 5, 5, 5, 5, 5]})
    result = detect_numerical_anomalies(df)
    assert result.empty


def test_obvious_outlier_is_detected():
    df = pd.DataFrame({"Value": [10, 11, 9, 10, 12, 11, 9, 500]})
    result = detect_numerical_anomalies(df)
    assert 500 in result["Value"].values


def test_get_anomaly_summary_counts_match(sample_df):
    anomalies = detect_numerical_anomalies(sample_df)
    summary = get_anomaly_summary(anomalies)
    assert summary["total_anomalies"] == len(anomalies)
    assert summary["columns_with_anomalies"] == anomalies["Column"].nunique()


def test_get_anomaly_summary_handles_empty_input():
    empty = pd.DataFrame(columns=["Row_Index", "Column", "Value"])
    summary = get_anomaly_summary(empty)
    assert summary["total_anomalies"] == 0
    assert summary["anomalies_by_column"] == {}


def test_get_anomalous_rows_returns_original_columns(sample_df):
    anomalies = detect_numerical_anomalies(sample_df)
    rows = get_anomalous_rows(sample_df, anomalies)
    assert list(rows.columns) == list(sample_df.columns)
    # every flagged row index must exist in the returned subset
    assert set(anomalies["Row_Index"]).issubset(set(rows.index))
