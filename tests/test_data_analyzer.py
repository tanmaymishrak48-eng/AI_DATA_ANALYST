"""
Unit tests for data_analyzer.py

These test the pure-pandas calculation layer directly — the layer
that actually produces every number the app shows. No Gemini/API
calls are involved, so this file always runs, offline.
"""

import pytest

from data_analyzer import (
    calculate_sum,
    calculate_average,
    calculate_minimum,
    calculate_maximum,
    sum_by_group,
    average_by_group,
    count_by_group,
    highest_group,
    lowest_group,
    execute_analysis_plan,
)


def test_calculate_sum(sample_df):
    expected = sample_df["Revenue"].sum()
    assert calculate_sum(sample_df, "Revenue") == pytest.approx(expected)


def test_calculate_average(sample_df):
    expected = sample_df["Profit"].mean()
    assert calculate_average(sample_df, "Profit") == pytest.approx(expected)


def test_calculate_minimum(sample_df):
    expected = sample_df["Revenue"].min()
    assert calculate_minimum(sample_df, "Revenue") == pytest.approx(expected)


def test_calculate_maximum(sample_df):
    expected = sample_df["Revenue"].max()
    assert calculate_maximum(sample_df, "Revenue") == pytest.approx(expected)


def test_calculate_sum_missing_column_raises(sample_df):
    with pytest.raises(ValueError):
        calculate_sum(sample_df, "Does_Not_Exist")


def test_sum_by_group_matches_pandas_groupby(sample_df):
    result = sum_by_group(sample_df, "Region", "Revenue")
    expected = (
        sample_df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    )
    assert list(result["Region"]) == list(expected.index)
    assert result["Revenue"].iloc[0] == pytest.approx(expected.iloc[0])


def test_average_by_group_is_sorted_descending(sample_df):
    result = average_by_group(sample_df, "Region", "Profit")
    values = list(result["Profit"])
    assert values == sorted(values, reverse=True)


def test_count_by_group_totals_match_row_count(sample_df):
    result = count_by_group(sample_df, "Region")
    # Rows with a missing Region are dropped by pandas' groupby (the
    # sample dataset has a couple of these), so compare against the
    # number of rows that actually have a Region value, not len(df).
    expected_total = sample_df["Region"].notna().sum()
    assert result["Count"].sum() == expected_total


def test_highest_group_is_actually_the_max(sample_df):
    result = highest_group(sample_df, "Region", "Revenue")
    grouped = sample_df.groupby("Region")["Revenue"].sum()
    assert result["group"] == grouped.idxmax()
    assert result["value"] == pytest.approx(grouped.max())


def test_lowest_group_is_actually_the_min(sample_df):
    result = lowest_group(sample_df, "Region", "Revenue")
    grouped = sample_df.groupby("Region")["Revenue"].sum()
    assert result["group"] == grouped.idxmin()
    assert result["value"] == pytest.approx(grouped.min())


def test_execute_analysis_plan_sum(sample_df):
    plan = {"operation": "sum", "value_column": "Revenue"}
    result = execute_analysis_plan(sample_df, plan)
    assert result["type"] == "scalar"
    assert result["value"] == pytest.approx(sample_df["Revenue"].sum())


def test_execute_analysis_plan_highest_group(sample_df):
    plan = {
        "operation": "highest_group",
        "group_column": "Region",
        "value_column": "Revenue",
    }
    result = execute_analysis_plan(sample_df, plan)
    grouped = sample_df.groupby("Region")["Revenue"].sum()
    assert result["group"] == grouped.idxmax()


def test_execute_analysis_plan_invalid_operation_raises(sample_df):
    plan = {"operation": "not_a_real_operation"}
    with pytest.raises(ValueError):
        execute_analysis_plan(sample_df, plan)
