import pandas as pd
import numpy as np


# ============================================================
# NUMERICAL ANOMALY DETECTION
# ============================================================

def detect_numerical_anomalies(
    df,
    columns=None,
    threshold=1.5
):
    """
    Detects unusual values in numerical columns
    using the Interquartile Range (IQR) method.

    A value is considered an anomaly when it is:

        value < Q1 - threshold * IQR

    OR

        value > Q3 + threshold * IQR

    Returns a DataFrame containing the detected anomalies.
    """

    # --------------------------------------------------------
    # Select numerical columns
    # --------------------------------------------------------

    if columns is None:

        columns = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

    else:

        columns = [
            column
            for column in columns
            if column in df.columns
            and pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]


    # --------------------------------------------------------
    # No numerical columns
    # --------------------------------------------------------

    if not columns:

        return pd.DataFrame()


    # --------------------------------------------------------
    # Store anomaly records
    # --------------------------------------------------------

    anomaly_records = []


    # --------------------------------------------------------
    # Analyze each numerical column
    # --------------------------------------------------------

    for column in columns:

        series = df[column].dropna()

        # Not enough data
        if len(series) < 4:
            continue


        # ----------------------------------------------------
        # Calculate quartiles
        # ----------------------------------------------------

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1


        # ----------------------------------------------------
        # Handle zero IQR
        # ----------------------------------------------------

        if iqr == 0:
            continue


        lower_bound = (
            q1 - threshold * iqr
        )

        upper_bound = (
            q3 + threshold * iqr
        )


        # ----------------------------------------------------
        # Find anomalies
        # ----------------------------------------------------

        anomaly_mask = (
            (df[column] < lower_bound)
            |
            (df[column] > upper_bound)
        )


        anomaly_indices = df.index[
            anomaly_mask
            & df[column].notna()
        ]


        # ----------------------------------------------------
        # Save anomaly information
        # ----------------------------------------------------

        for index in anomaly_indices:

            value = df.loc[
                index,
                column
            ]

            if value < lower_bound:

                reason = (
                    f"Unusually low {column}"
                )

            else:

                reason = (
                    f"Unusually high {column}"
                )


            anomaly_records.append({

                "Row_Index": index,

                "Column": column,

                "Value": value,

                "Lower_Bound": lower_bound,

                "Upper_Bound": upper_bound,

                "Reason": reason

            })


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    if not anomaly_records:

        return pd.DataFrame(
            columns=[
                "Row_Index",
                "Column",
                "Value",
                "Lower_Bound",
                "Upper_Bound",
                "Reason"
            ]
        )


    return pd.DataFrame(
        anomaly_records
    )


# ============================================================
# ANOMALY SUMMARY
# ============================================================

def get_anomaly_summary(
    anomaly_df
):
    """
    Creates a summary of detected anomalies.
    """

    if anomaly_df.empty:

        return {
            "total_anomalies": 0,
            "columns_with_anomalies": 0,
            "anomalies_by_column": {}
        }


    anomalies_by_column = (
        anomaly_df["Column"]
        .value_counts()
        .to_dict()
    )


    return {

        "total_anomalies": int(
            len(anomaly_df)
        ),

        "columns_with_anomalies": int(
            anomaly_df["Column"].nunique()
        ),

        "anomalies_by_column":
            anomalies_by_column

    }


# ============================================================
# GET ORIGINAL ROWS CONTAINING ANOMALIES
# ============================================================

def get_anomalous_rows(
    df,
    anomaly_df
):
    """
    Returns the original dataset rows that
    contain at least one detected anomaly.
    """

    if anomaly_df.empty:

        return pd.DataFrame(
            columns=df.columns
        )


    indices = (
        anomaly_df["Row_Index"]
        .drop_duplicates()
        .tolist()
    )


    return df.loc[
        indices
    ].copy()