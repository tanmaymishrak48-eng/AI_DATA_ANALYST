import os
import json
import time
import random

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# ENVIRONMENT / GROQ CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found in the .env file."
    )


client = Groq(
    api_key=API_KEY
)

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# GROQ REQUEST WITH RETRY / QUOTA HANDLING
# ============================================================

def generate_with_retry(
    prompt,
    max_retries=3,
):
    """
    Send a prompt to Groq with limited retry handling.

    Temporary server/rate-limit errors are retried.

    Daily quota exhaustion is NOT retried because
    repeatedly sending requests cannot restore the quota.
    """

    last_error = None

    for attempt in range(max_retries):

        try:

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            return response.choices[0].message.content

        except Exception as error:

            last_error = error

            error_text = str(error).upper()

            # ------------------------------------------------
            # Daily quota exhaustion
            # ------------------------------------------------
            # Groq's free tier enforces both per-minute (RPM/TPM)
            # and per-day (RPD/TPD) limits. Only the per-day kind
            # is treated as non-retryable, since retrying cannot
            # restore a daily quota.

            if (
                "RATE_LIMIT" in error_text
                or "429" in error_text
                or "TOO MANY REQUESTS" in error_text
            ) and (
                "PER DAY" in error_text
                or "RPD" in error_text
                or "TPD" in error_text
                or "DAILY" in error_text
            ):

                raise RuntimeError(
                    "Groq daily Free Tier quota has "
                    "been exhausted. AI-generated "
                    "features are temporarily unavailable "
                    "until the quota resets. "
                    "Non-AI data analysis features "
                    "remain available."
                ) from error


            # ------------------------------------------------
            # Temporary errors
            # ------------------------------------------------

            temporary_errors = [
                "500",
                "502",
                "503",
                "504",
                "429",
                "UNAVAILABLE",
                "DEADLINE",
                "TIMEOUT",
                "RATE_LIMIT",
                "TOO MANY REQUESTS",
            ]

            is_temporary = any(
                item in error_text
                for item in temporary_errors
            )


            if not is_temporary:

                raise RuntimeError(
                    f"Groq request failed: {error}"
                ) from error


            # ------------------------------------------------
            # Retry with exponential backoff
            # ------------------------------------------------

            if attempt < max_retries - 1:

                wait_time = (
                    (2 ** attempt)
                    + random.uniform(0.2, 0.8)
                )

                time.sleep(
                    wait_time
                )


    raise RuntimeError(
        f"Groq request failed after "
        f"{max_retries} attempts: {last_error}"
    ) from last_error


# ============================================================
# DATASET CONTEXT
# ============================================================

def create_dataset_context(df):
    """
    Create a compact text representation of a DataFrame
    that can safely be sent to Groq.

    IMPORTANT:
    This function returns a STRING.

    The original DataFrame must never be replaced by this
    string in functions that still need DataFrame operations.
    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "create_dataset_context expected "
            "a Pandas DataFrame."
        )


    columns = df.columns.tolist()

    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )


    missing_values = int(
        df.isna()
        .sum()
        .sum()
    )


    duplicate_rows = int(
        df.duplicated()
        .sum()
    )


    context_parts = []

    context_parts.append(
        f"Rows: {len(df)}"
    )

    context_parts.append(
        f"Columns: {len(df.columns)}"
    )

    context_parts.append(
        f"Column names: {columns}"
    )

    context_parts.append(
        f"Numerical columns: {numeric_columns}"
    )

    context_parts.append(
        f"Categorical columns: {categorical_columns}"
    )

    context_parts.append(
        f"Missing values: {missing_values}"
    )

    context_parts.append(
        f"Duplicate rows: {duplicate_rows}"
    )


    # --------------------------------------------------------
    # Numerical summary
    # --------------------------------------------------------

    if numeric_columns:

        numerical_summary = (
            df[numeric_columns]
            .describe()
            .round(2)
            .to_string()
        )

        context_parts.append(
            "Numerical summary:\n"
            + numerical_summary
        )


    # --------------------------------------------------------
    # Categorical information
    # --------------------------------------------------------

    if categorical_columns:

        categorical_information = []

        for column in categorical_columns:

            unique_values = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            # Avoid sending extremely large lists
            if len(unique_values) > 30:

                unique_values = (
                    unique_values[:30]
                    + ["..."]
                )

            categorical_information.append(
                f"{column}: {unique_values}"
            )


        context_parts.append(
            "Categorical values:\n"
            + "\n".join(
                categorical_information
            )
        )


    # --------------------------------------------------------
    # Sample records
    # --------------------------------------------------------

    sample = (
        df.head(5)
        .copy()
    )


    for column in sample.columns:

        if sample[column].dtype == "object":

            sample[column] = (
                sample[column]
                .astype(str)
            )


    context_parts.append(
        "Sample records:\n"
        + sample.to_string(
            index=False
        )
    )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json(text):
    """
    Extract JSON from Groq response.

    Handles:
    - plain JSON
    - ```json ... ```
    - surrounding explanatory text
    """

    if not text:

        raise ValueError(
            "Groq returned an empty response."
        )


    text = str(text).strip()


    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if len(lines) >= 3:

            lines = lines[1:]

            if lines[-1].strip().startswith("```"):

                lines = lines[:-1]

            text = "\n".join(lines).strip()


    # --------------------------------------------------------
    # Direct JSON
    # --------------------------------------------------------

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass


    # --------------------------------------------------------
    # Search for JSON object
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")


    if start != -1 and end != -1:

        json_text = text[
            start:end + 1
        ]

        try:

            return json.loads(
                json_text
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Groq returned invalid JSON."
            ) from error


    raise ValueError(
        "No valid JSON object was found "
        "in the Groq response."
    )


# ============================================================
# AI DATASET SUMMARY
# ============================================================

def generate_dataset_summary(df):
    """
    Generate an AI-powered summary of the dataset.

    Groq explains the verified dataset statistics.
    It does not perform the underlying calculations.
    """

    if not isinstance(df, pd.DataFrame):

        raise TypeError(
            "generate_dataset_summary expected "
            "a Pandas DataFrame."
        )


    dataset_context = (
        create_dataset_context(df)
    )


    prompt = f"""
You are an AI Data Analyst.

Analyze the following verified dataset information
and produce a concise but useful business-oriented
dataset summary.

IMPORTANT RULES:

1. Use only information present in the dataset context.
2. Do not invent numerical values.
3. Do not invent trends.
4. Do not claim causation without evidence.
5. Clearly distinguish observations from assumptions.
6. Use the actual numbers provided.
7. Keep the explanation understandable for a student
   demonstrating the project.

Provide the following sections:

## Dataset Overview

## Data Quality

## Important Statistical Observations

## Business Insights

## Key Takeaways

DATASET CONTEXT:

{dataset_context}
"""


    return generate_with_retry(
        prompt
    )


# ============================================================
# NATURAL-LANGUAGE ANALYSIS PLAN
# ============================================================

def create_analysis_plan(
    df,
    question,
):
    """
    Convert a natural-language question into a
    structured analysis plan.

    IMPORTANT ARCHITECTURE:

    Pandas DataFrame
        ↓
    Dataset context string
        ↓
    Groq
        ↓
    JSON analysis plan
        ↓
    Pandas performs calculation

    Groq does NOT calculate the final numerical answer.
    """

    # --------------------------------------------------------
    # Validate DataFrame
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "create_analysis_plan expected "
            "a Pandas DataFrame."
        )


    if not question or not str(question).strip():

        raise ValueError(
            "Please enter a valid question."
        )


    # --------------------------------------------------------
    # Extract DataFrame information
    # BEFORE creating text context
    # --------------------------------------------------------

    columns = (
        df.columns
        .tolist()
    )


    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )


    # This is intentionally a STRING.
    dataset_context = (
        create_dataset_context(df)
    )


    # --------------------------------------------------------
    # Groq prompt
    # --------------------------------------------------------

    prompt = f"""
You are the planning component of an AI Data Analyst.

Convert the user's natural-language question into
a structured JSON analysis plan.

IMPORTANT RULES:

1. Do NOT calculate the answer.
2. Do NOT invent numerical results.
3. Pandas will perform the actual calculation.
4. Use only columns that exist in the dataset.
5. Return ONLY valid JSON.
6. Do not use Markdown code fences.
7. "column" must be a numerical column when required.
8. "group_by" must be a real dataset column when grouping.
9. Never create a new column name.
10. Never guess a column name.

AVAILABLE DATASET COLUMNS:

{columns}

NUMERICAL COLUMNS:

{numeric_columns}

CATEGORICAL COLUMNS:

{categorical_columns}

DATASET CONTEXT:

{dataset_context}


SUPPORTED OPERATIONS:

------------------------------------------------------------
1. sum
------------------------------------------------------------

Use when the user asks for the total of a numerical column.

Example:

"What is the total revenue?"

Return:

{{
    "operation": "sum",
    "column": "Revenue",
    "group_by": null
}}


------------------------------------------------------------
2. mean
------------------------------------------------------------

Use when the user asks for an average.

Example:

"What is the average profit?"

Return:

{{
    "operation": "mean",
    "column": "Profit",
    "group_by": null
}}


------------------------------------------------------------
3. count
------------------------------------------------------------

Use when the user asks for the total number of records,
orders or rows.

Example:

"How many orders are there?"

Return:

{{
    "operation": "count",
    "column": null,
    "group_by": null
}}


------------------------------------------------------------
4. count_by_group
------------------------------------------------------------

Use when the user asks for the number of records/orders
in each category.

Example:

"How many orders are in each region?"

Return:

{{
    "operation": "count_by_group",
    "column": null,
    "group_by": "Region"
}}


------------------------------------------------------------
5. sum_by_group
------------------------------------------------------------

Use when the user asks for a total numerical value
grouped by a category.

Example:

"Show revenue by region."

Return:

{{
    "operation": "sum_by_group",
    "column": "Revenue",
    "group_by": "Region"
}}


------------------------------------------------------------
6. mean_by_group
------------------------------------------------------------

Use when the user asks for an average numerical value
grouped by a category.

Example:

"Show average profit by region."

Return:

{{
    "operation": "mean_by_group",
    "column": "Profit",
    "group_by": "Region"
}}


------------------------------------------------------------
7. max_by_group
------------------------------------------------------------

Use when the user asks which category/region/product
has the highest total numerical value.

Example:

"Which region has the highest revenue?"

Return:

{{
    "operation": "max_by_group",
    "column": "Revenue",
    "group_by": "Region"
}}


USER QUESTION:

{question}

Return ONLY the JSON object.
"""


    # --------------------------------------------------------
    # Ask Groq for plan
    # --------------------------------------------------------

    response = generate_with_retry(
        prompt
    )


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    plan = extract_json(
        response
    )


    if not isinstance(
        plan,
        dict,
    ):

        raise ValueError(
            "Groq returned an invalid "
            "analysis plan."
        )


    # --------------------------------------------------------
    # Validate operation
    # --------------------------------------------------------

    allowed_operations = {
        "sum",
        "mean",
        "count",
        "count_by_group",
        "sum_by_group",
        "mean_by_group",
        "max_by_group",
    }


    operation = plan.get(
        "operation"
    )


    if operation not in allowed_operations:

        raise ValueError(
            "Unsupported analysis operation: "
            f"{operation}"
        )


    # --------------------------------------------------------
    # Validate column
    # --------------------------------------------------------

    column = plan.get(
        "column"
    )


    if column is not None:

        if column not in columns:

            raise ValueError(
                "AI selected an invalid column: "
                f"{column}"
            )


    # --------------------------------------------------------
    # Validate group_by
    # --------------------------------------------------------

    group_by = plan.get(
        "group_by"
    )


    if group_by is not None:

        if group_by not in columns:

            raise ValueError(
                "AI selected an invalid grouping "
                f"column: {group_by}"
            )


    # --------------------------------------------------------
    # Validate numerical operations
    # --------------------------------------------------------

    numerical_operations = {
        "sum",
        "mean",
        "sum_by_group",
        "mean_by_group",
        "max_by_group",
    }


    if operation in numerical_operations:

        if column is None:

            raise ValueError(
                f"Operation '{operation}' "
                "requires a numerical column."
            )


        if column not in numeric_columns:

            raise ValueError(
                f"Column '{column}' "
                "is not numerical."
            )


    # --------------------------------------------------------
    # Validate grouping operations
    # --------------------------------------------------------

    grouping_operations = {
        "count_by_group",
        "sum_by_group",
        "mean_by_group",
        "max_by_group",
    }


    if operation in grouping_operations:

        if group_by is None:

            raise ValueError(
                f"Operation '{operation}' "
                "requires a grouping column."
            )


    # --------------------------------------------------------
    # Return clean plan
    # --------------------------------------------------------

    return {
        "operation": operation,
        "column": column,
        "group_by": group_by,
    }


# ============================================================
# AI AUTOMATIC VISUALIZATION PLAN
# ============================================================

def create_visualization_plan(
    df,
    question,
):
    """
    Convert a natural-language visualization request
    into a structured JSON chart plan.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "create_visualization_plan expected "
            "a Pandas DataFrame."
        )


    if not question or not str(question).strip():

        raise ValueError(
            "Please enter a visualization request."
        )


    columns = (
        df.columns
        .tolist()
    )


    numeric_columns = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )


    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
            ]
        )
        .columns
        .tolist()
    )


    dataset_context = (
        create_dataset_context(df)
    )


    prompt = f"""
You are an AI visualization planner.

Convert the user's visualization request into
a structured JSON visualization plan.

IMPORTANT RULES:

1. Do NOT calculate numerical results.
2. Use only columns that exist.
3. Return ONLY valid JSON.
4. Do not use Markdown.
5. Do not invent column names.

AVAILABLE COLUMNS:

{columns}

NUMERICAL COLUMNS:

{numeric_columns}

CATEGORICAL COLUMNS:

{categorical_columns}

SUPPORTED CHART TYPES:

- bar
- line
- histogram
- scatter

JSON FORMAT:

{{
    "chart_type": "bar",
    "x_column": "Region",
    "y_column": "Revenue"
}}

For a histogram:

{{
    "chart_type": "histogram",
    "x_column": "Quantity",
    "y_column": null
}}

USER REQUEST:

{question}

DATASET CONTEXT:

{dataset_context}

Return ONLY the JSON object.
"""


    response = generate_with_retry(
        prompt
    )


    plan = extract_json(
        response
    )


    if not isinstance(
        plan,
        dict,
    ):

        raise ValueError(
            "Groq returned an invalid "
            "visualization plan."
        )


    chart_type = plan.get(
        "chart_type"
    )

    x_column = plan.get(
        "x_column"
    )

    y_column = plan.get(
        "y_column"
    )


    allowed_chart_types = {
        "bar",
        "line",
        "histogram",
        "scatter",
    }


    if chart_type not in allowed_chart_types:

        raise ValueError(
            f"Unsupported chart type: "
            f"{chart_type}"
        )


    if x_column not in columns:

        raise ValueError(
            f"AI selected invalid x_column: "
            f"{x_column}"
        )


    if chart_type != "histogram":

        if y_column not in columns:

            raise ValueError(
                f"AI selected invalid y_column: "
                f"{y_column}"
            )


        if y_column not in numeric_columns:

            raise ValueError(
                f"AI selected y_column "
                f"'{y_column}', which is not numerical."
            )


    else:

        y_column = None


    return {
        "chart_type": chart_type,
        "x_column": x_column,
        "y_column": y_column,
    }


# ============================================================
# AI ANOMALY INSIGHTS
# ============================================================

def generate_anomaly_insights(
    df,
    anomaly_df,
    max_anomalies=20,
):
    """
    Generate AI explanations for statistically detected
    anomalies.

    Important:
    This function does NOT detect anomalies.
    anomaly_detector.py performs anomaly detection.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "generate_anomaly_insights expected "
            "a Pandas DataFrame."
        )


    if anomaly_df is None:

        raise ValueError(
            "Anomaly results are required."
        )


    # --------------------------------------------------------
    # Calculate verified anomaly statistics
    # --------------------------------------------------------

    anomaly_observations = len(
        anomaly_df
    )


    # Attempt to determine affected row IDs/indexes
    unique_affected_rows = 0


    if isinstance(
        anomaly_df,
        pd.DataFrame,
    ):

        if "Row_Index" in anomaly_df.columns:

            unique_affected_rows = (
                anomaly_df["Row_Index"]
                .nunique()
            )

        elif "Index" in anomaly_df.columns:

            unique_affected_rows = (
                anomaly_df["Index"]
                .nunique()
            )

        elif "Row" in anomaly_df.columns:

            unique_affected_rows = (
                anomaly_df["Row"]
                .nunique()
            )

        else:

            # If the anomaly dataframe has one row per
            # affected observation, use its index.
            unique_affected_rows = (
                anomaly_df.index.nunique()
            )


    affected_percentage = (
        (
            unique_affected_rows
            / len(df)
            * 100
        )
        if len(df) > 0
        else 0
    )


    # --------------------------------------------------------
    # Limit anomaly records sent to Groq
    # --------------------------------------------------------

    anomaly_sample = (
        anomaly_df
        .head(max_anomalies)
        .copy()
    )


    for column in anomaly_sample.columns:

        if anomaly_sample[column].dtype == "object":

            anomaly_sample[column] = (
                anomaly_sample[column]
                .astype(str)
            )


    anomaly_text = (
        anomaly_sample
        .to_string(index=False)
    )


    dataset_context = (
        create_dataset_context(df)
    )


    prompt = f"""
You are an AI Data Analyst explaining statistical
anomalies detected in a dataset.

The anomalies were already detected by a statistical
IQR-based anomaly detection system.

DO NOT detect new anomalies.

DO NOT invent anomaly values.

DO NOT invent numerical results.

IMPORTANT TERMINOLOGY:

- "Anomaly observations" means the total number of
  anomaly flags/observations returned by the detector.
- "Unique affected rows" means the number of distinct
  dataset rows containing anomalies.
- These two numbers must NOT be treated as the same.

VERIFIED STATISTICS:

Total dataset rows:
{len(df)}

Anomaly observations:
{anomaly_observations}

Unique affected rows:
{unique_affected_rows}

Affected row percentage:
{affected_percentage:.2f}%

DATASET CONTEXT:

{dataset_context}

DETECTED ANOMALY SAMPLE:

{anomaly_text}

Explain the anomalies using these sections:

## 1. Anomaly Overview

## 2. Important Anomalies

## 3. Possible Business Reasons

## 4. Business Impact

## 5. Recommended Actions

Be careful to distinguish statistical evidence
from possible explanations.
"""


    return generate_with_retry(
        prompt
    )


# ============================================================
# AI BUSINESS RECOMMENDATIONS
# ============================================================

def generate_business_recommendations(
    df,
    anomaly_df=None,
    max_recommendations=7,
):
    """
    Generate evidence-based business recommendations.

    All numerical evidence is calculated using Pandas first.
    Groq only interprets the verified evidence.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "generate_business_recommendations "
            "expected a Pandas DataFrame."
        )


    evidence = {}


    # --------------------------------------------------------
    # General metrics
    # --------------------------------------------------------

    evidence["total_rows"] = len(df)

    evidence["total_columns"] = len(
        df.columns
    )

    evidence["missing_values"] = int(
        df.isna()
        .sum()
        .sum()
    )

    evidence["duplicate_rows"] = int(
        df.duplicated()
        .sum()
    )


    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    if "Revenue" in df.columns:

        evidence["total_revenue"] = float(
            df["Revenue"]
            .sum()
        )

        evidence["average_revenue"] = float(
            df["Revenue"]
            .mean()
        )

        if "Region" in df.columns:

            evidence["revenue_by_region"] = (
                df.groupby("Region")["Revenue"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .to_dict()
            )


        if "Category" in df.columns:

            evidence["revenue_by_category"] = (
                df.groupby("Category")["Revenue"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .to_dict()
            )


        if "Product" in df.columns:

            evidence["revenue_by_product"] = (
                df.groupby("Product")["Revenue"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
                .to_dict()
            )


    # --------------------------------------------------------
    # Profit
    # --------------------------------------------------------

    if "Profit" in df.columns:

        evidence["total_profit"] = float(
            df["Profit"]
            .sum()
        )

        evidence["average_profit"] = float(
            df["Profit"]
            .mean()
        )


        if "Region" in df.columns:

            evidence["profit_by_region"] = (
                df.groupby("Region")["Profit"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .to_dict()
            )


        if "Category" in df.columns:

            evidence["profit_by_category"] = (
                df.groupby("Category")["Profit"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .to_dict()
            )


    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    if "Quantity" in df.columns:

        evidence["total_quantity"] = float(
            df["Quantity"]
            .sum()
        )

        evidence["average_quantity"] = float(
            df["Quantity"]
            .mean()
        )


    # --------------------------------------------------------
    # Discount
    # --------------------------------------------------------

    if "Discount" in df.columns:

        evidence["average_discount"] = float(
            df["Discount"]
            .mean()
        )

        evidence["maximum_discount"] = float(
            df["Discount"]
            .max()
        )


    # --------------------------------------------------------
    # Customer rating
    # --------------------------------------------------------

    if "Customer_Rating" in df.columns:

        evidence["average_customer_rating"] = (
            float(
                df["Customer_Rating"]
                .mean()
            )
        )


    # --------------------------------------------------------
    # Anomaly evidence
    # --------------------------------------------------------

    if anomaly_df is not None:

        anomaly_observations = len(
            anomaly_df
        )

        unique_affected_rows = 0


        if isinstance(
            anomaly_df,
            pd.DataFrame,
        ):

            if "Row_Index" in anomaly_df.columns:

                unique_affected_rows = (
                    anomaly_df["Row_Index"]
                    .nunique()
                )

            elif "Index" in anomaly_df.columns:

                unique_affected_rows = (
                    anomaly_df["Index"]
                    .nunique()
                )

            elif "Row" in anomaly_df.columns:

                unique_affected_rows = (
                    anomaly_df["Row"]
                    .nunique()
                )

            else:

                unique_affected_rows = (
                    anomaly_df.index.nunique()
                )


        evidence[
            "anomaly_observations"
        ] = anomaly_observations

        evidence[
            "unique_affected_rows"
        ] = unique_affected_rows

        evidence[
            "affected_row_percentage"
        ] = (
            (
                unique_affected_rows
                / len(df)
                * 100
            )
            if len(df) > 0
            else 0
        )


    # --------------------------------------------------------
    # Convert evidence to JSON
    # --------------------------------------------------------

    evidence_json = json.dumps(
        evidence,
        indent=2,
        default=str,
    )


    # --------------------------------------------------------
    # Groq prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI Business Analyst.

Generate evidence-based business recommendations
from the verified dataset evidence below.

IMPORTANT RULES:

1. Do not invent numbers.
2. Do not invent trends.
3. Do not claim a cause unless supported by evidence.
4. Every recommendation must be connected to evidence.
5. Do not confuse anomaly observations with unique rows.
6. The numerical calculations were already performed
   using Pandas.
7. Groq is only responsible for interpretation.
8. Do not create unsupported statistics.

Generate up to {max_recommendations}
useful recommendations.

For each recommendation use:

## Recommendation

### Evidence

### Business Reason

### Expected Benefit

### Priority

Use priorities such as:

- High
- Medium
- Low

Do not provide an overall ranking of the recommendations.

VERIFIED DATASET EVIDENCE:

{evidence_json}
"""


    return generate_with_retry(
        prompt
    )