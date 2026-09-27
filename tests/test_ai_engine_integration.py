"""
Integration tests for ai_engine.py.

These call the LIVE Groq API, so they are automatically skipped
when GROQ_API_KEY is not set (see conftest.py). Run them with a
real key to smoke-test the AI layer end to end:

    pytest tests/test_ai_engine_integration.py -v
"""

import pandas as pd

from ai_engine import (
    generate_dataset_summary,
    create_analysis_plan,
    generate_anomaly_insights,
    generate_business_recommendations,
)
from anomaly_detector import detect_numerical_anomalies
from data_analyzer import execute_analysis_plan

from conftest import requires_groq


@requires_groq
def test_generate_dataset_summary_returns_text(sample_df):
    result = generate_dataset_summary(sample_df)
    assert isinstance(result, str)
    assert len(result) > 0


@requires_groq
def test_create_analysis_plan_returns_valid_operation(sample_df):
    # NOTE: signature is create_analysis_plan(df, question) —
    # an earlier version of this test had the arguments reversed.
    plan = create_analysis_plan(sample_df, "What is the total revenue?")
    assert isinstance(plan, dict)
    assert plan.get("operation") in {
        "sum",
        "mean",
        "average",
        "count",
        "count_by_group",
        "sum_by_group",
        "average_by_group",
        "mean_by_group",
        "max_by_group",
        "highest_group",
        "lowest_group",
    }


@requires_groq
def test_full_question_pipeline_executes_without_error(sample_df):
    """
    End-to-end: natural-language question -> Groq plan -> pandas result.
    This is the core "generative AI" pipeline the whole app is built on.
    """
    plan = create_analysis_plan(
        sample_df, "Which region generated the highest revenue?"
    )
    result = execute_analysis_plan(sample_df, plan)
    assert result["type"] in {"scalar", "table", "highest_group", "lowest_group"}


@requires_groq
def test_generate_anomaly_insights_returns_text(sample_df):
    anomalies = detect_numerical_anomalies(sample_df)
    insights = generate_anomaly_insights(sample_df, anomalies)
    assert isinstance(insights, str)
    assert len(insights) > 0


@requires_groq
def test_generate_business_recommendations_returns_text(sample_df):
    anomalies = detect_numerical_anomalies(sample_df)
    recommendations = generate_business_recommendations(sample_df, anomalies)
    assert isinstance(recommendations, str)
    assert len(recommendations) > 0
