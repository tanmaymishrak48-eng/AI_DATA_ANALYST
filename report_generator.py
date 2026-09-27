from io import BytesIO
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_value(value):
    """
    Convert a value into safe text for the PDF.
    """

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    return str(value)


def format_number(value, decimals=2):
    """
    Format numerical values safely.
    """

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):,.{decimals}f}"

    except Exception:

        return safe_value(value)


def dataframe_to_table(
    dataframe,
    max_rows=25,
):
    """
    Convert a Pandas DataFrame into a ReportLab table.
    """

    if dataframe is None:
        return None

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        return None

    if dataframe.empty:
        return None

    display_df = dataframe.head(
        max_rows
    ).copy()

    # Convert every value to text
    for column in display_df.columns:

        display_df[column] = (
            display_df[column]
            .apply(safe_value)
        )

    table_data = [
        list(display_df.columns)
    ]

    table_data.extend(
        display_df.values.tolist()
    )

    table = Table(
        table_data,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, 0),
                    8,
                ),
                (
                    "FONTSIZE",
                    (0, 1),
                    (-1, -1),
                    7,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    4,
                ),
            ]
        )
    )

    return table


def add_section_title(
    story,
    title,
    styles,
):
    """
    Add a section heading.
    """

    story.append(
        Paragraph(
            title,
            styles["SectionHeading"],
        )
    )

    story.append(
        Spacer(
            1,
            0.10 * inch,
        )
    )


def add_text_section(
    story,
    text,
    styles,
):
    """
    Add AI-generated text to the PDF.

    Markdown-style headings are converted into
    readable PDF paragraphs.
    """

    if text is None:
        return

    text = str(text).strip()

    if not text:
        return

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:

            story.append(
                Spacer(
                    1,
                    0.06 * inch,
                )
            )

            continue

        # Remove Markdown heading markers
        clean_line = line.lstrip("#").strip()

        # Basic HTML escaping
        clean_line = (
            clean_line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        story.append(
            Paragraph(
                clean_line,
                styles["BodyTextCustom"],
            )
        )

        story.append(
            Spacer(
                1,
                0.03 * inch,
            )
        )


def add_simple_table(
    story,
    data,
    column_widths=None,
):
    """
    Add a simple formatted table.
    """

    if not data:
        return

    if column_widths:

        table = Table(
            data,
            colWidths=column_widths,
            repeatRows=1,
        )

    else:

        table = Table(
            data,
            repeatRows=1,
        )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(table)


# ============================================================
# MAIN PDF GENERATOR
# ============================================================

def generate_pdf_report(
    df,
    file_name=None,
    anomaly_summary=None,
    anomalous_rows=None,
    question_plan=None,
    question_result=None,
    ai_dataset_summary=None,
    anomaly_ai_insights=None,
    business_recommendations=None,
):
    """
    Generate the complete AI Data Analyst PDF report.

    Parameters:
        df:
            Pandas DataFrame.

        file_name:
            Name of uploaded dataset.

        anomaly_summary:
            Summary generated by anomaly detector.

        anomalous_rows:
            DataFrame containing affected rows.

        question_plan:
            Natural-language analysis plan.

        question_result:
            Result of natural-language analysis.

        ai_dataset_summary:
            AI-generated dataset summary.

        anomaly_ai_insights:
            AI explanation of anomalies.

        business_recommendations:
            AI-generated business recommendations.

    Returns:
        BytesIO object containing the PDF.
    """

    # ========================================================
    # VALIDATE DATAFRAME
    # ========================================================

    if not isinstance(
        df,
        pd.DataFrame,
    ):

        raise TypeError(
            "generate_pdf_report expected "
            "a Pandas DataFrame."
        )


    # ========================================================
    # CREATE PDF BUFFER
    # ========================================================

    buffer = BytesIO()


    # ========================================================
    # CREATE DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=40,
        bottomMargin=40,
        title="AI Data Analyst Report",
        author="AI Data Analyst",
    )


    # ========================================================
    # REPORTLAB STYLES
    # ========================================================

    styles = getSampleStyleSheet()


    # IMPORTANT:
    # We ADD custom styles.
    #
    # We do NOT do:
    #
    # styles["BodyText"] = ...
    #
    # because ReportLab's StyleSheet does not support
    # item assignment.
    # ========================================================

    styles.add(
        ParagraphStyle(
            name="ReportTitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=32,
            alignment=TA_CENTER,
            spaceAfter=18,
            textColor=colors.HexColor(
                "#111827"
            ),
        )
    )


    styles.add(
        ParagraphStyle(
            name="ReportSubtitleCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=12,
            textColor=colors.HexColor(
                "#475569"
            ),
        )
    )


    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceBefore=10,
            spaceAfter=10,
            textColor=colors.HexColor(
                "#111827"
            ),
        )
    )


    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            spaceAfter=5,
            textColor=colors.HexColor(
                "#334155"
            ),
        )
    )


    # ========================================================
    # STORY
    # ========================================================

    story = []


    # ========================================================
    # 1. TITLE PAGE
    # ========================================================

    story.append(
        Spacer(
            1,
            1.2 * inch,
        )
    )


    story.append(
        Paragraph(
            "AI Data Analyst",
            styles["ReportTitleCustom"],
        )
    )


    story.append(
        Paragraph(
            "Generative AI for Natural-Language "
            "Data Analysis, Visualization and "
            "Business Insights",
            styles["ReportSubtitleCustom"],
        )
    )


    story.append(
        Spacer(
            1,
            0.3 * inch,
        )
    )


    report_file_name = (
        file_name
        if file_name
        else "Uploaded Dataset"
    )


    story.append(
        Paragraph(
            f"<b>Dataset:</b> "
            f"{safe_value(report_file_name)}",
            styles["BodyTextCustom"],
        )
    )


    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["BodyTextCustom"],
        )
    )


    story.append(
        Spacer(
            1,
            0.3 * inch,
        )
    )


    story.append(
        Paragraph(
            "This report contains the statistical "
            "analysis, business metrics, anomaly "
            "detection results and available "
            "Generative AI insights produced by "
            "the AI Data Analyst system.",
            styles["BodyTextCustom"],
        )
    )


    story.append(
        PageBreak()
    )


    # ========================================================
    # 2. DATASET OVERVIEW
    # ========================================================

    add_section_title(
        story,
        "1. Dataset Overview",
        styles,
    )


    overview_data = [
        ["Metric", "Value"],
        [
            "Rows",
            f"{len(df):,}",
        ],
        [
            "Columns",
            f"{len(df.columns):,}",
        ],
        [
            "Missing Values",
            f"{int(df.isna().sum().sum()):,}",
        ],
        [
            "Duplicate Rows",
            f"{int(df.duplicated().sum()):,}",
        ],
    ]


    add_simple_table(
        story,
        overview_data,
        [
            2.5 * inch,
            2.5 * inch,
        ],
    )


    story.append(
        Spacer(
            1,
            0.2 * inch,
        )
    )


    story.append(
        Paragraph(
            "<b>Dataset Columns</b>",
            styles["BodyTextCustom"],
        )
    )


    columns_text = ", ".join(
        str(column)
        for column in df.columns
    )


    story.append(
        Paragraph(
            columns_text,
            styles["BodyTextCustom"],
        )
    )


    # ========================================================
    # 3. DATA QUALITY
    # ========================================================

    add_section_title(
        story,
        "2. Data Quality",
        styles,
    )


    missing_df = (
        df.isna()
        .sum()
        .reset_index()
    )


    missing_df.columns = [
        "Column",
        "Missing Values",
    ]


    missing_df = missing_df[
        missing_df["Missing Values"] > 0
    ]


    if missing_df.empty:

        story.append(
            Paragraph(
                "No missing values were detected "
                "in the dataset.",
                styles["BodyTextCustom"],
            )
        )

    else:

        missing_table = dataframe_to_table(
            missing_df,
            max_rows=50,
        )

        if missing_table:

            story.append(
                missing_table
            )


    story.append(
        Spacer(
            1,
            0.15 * inch,
        )
    )


    story.append(
        Paragraph(
            f"<b>Total duplicate rows:</b> "
            f"{int(df.duplicated().sum()):,}",
            styles["BodyTextCustom"],
        )
    )


    # ========================================================
    # 4. STATISTICAL SUMMARY
    # ========================================================

    add_section_title(
        story,
        "3. Statistical Summary",
        styles,
    )


    try:

        stats = (
            df.describe(
                include="all"
            )
            .transpose()
            .reset_index()
        )


        stats.rename(
            columns={
                "index": "Column"
            },
            inplace=True,
        )


        stats_table = dataframe_to_table(
            stats,
            max_rows=50,
        )


        if stats_table:

            story.append(
                stats_table
            )

    except Exception as error:

        story.append(
            Paragraph(
                "Statistical summary could not "
                "be generated: "
                f"{safe_value(error)}",
                styles["BodyTextCustom"],
            )
        )


    story.append(
        PageBreak()
    )


    # ========================================================
    # 5. KEY BUSINESS METRICS
    # ========================================================

    add_section_title(
        story,
        "4. Key Business Metrics",
        styles,
    )


    business_metrics = [
        ["Metric", "Value"],
    ]


    if "Revenue" in df.columns:

        business_metrics.append(
            [
                "Total Revenue",
                f"₹{format_number(df['Revenue'].sum())}",
            ]
        )

        business_metrics.append(
            [
                "Average Revenue",
                f"₹{format_number(df['Revenue'].mean())}",
            ]
        )


    if "Profit" in df.columns:

        business_metrics.append(
            [
                "Total Profit",
                f"₹{format_number(df['Profit'].sum())}",
            ]
        )

        business_metrics.append(
            [
                "Average Profit",
                f"₹{format_number(df['Profit'].mean())}",
            ]
        )


    if "Quantity" in df.columns:

        business_metrics.append(
            [
                "Total Quantity",
                format_number(
                    df["Quantity"].sum()
                ),
            ]
        )

        business_metrics.append(
            [
                "Average Quantity",
                format_number(
                    df["Quantity"].mean()
                ),
            ]
        )


    if "Customer_Rating" in df.columns:

        business_metrics.append(
            [
                "Average Customer Rating",
                format_number(
                    df["Customer_Rating"].mean()
                ),
            ]
        )


    add_simple_table(
        story,
        business_metrics,
        [
            3.0 * inch,
            2.5 * inch,
        ],
    )


    # ========================================================
    # 6. REVENUE BY REGION
    # ========================================================

    if (
        "Region" in df.columns
        and "Revenue" in df.columns
    ):

        add_section_title(
            story,
            "5. Revenue by Region",
            styles,
        )


        revenue_region = (
            df.groupby(
                "Region",
                as_index=False,
            )["Revenue"]
            .sum()
            .sort_values(
                by="Revenue",
                ascending=False,
            )
        )


        revenue_region["Revenue"] = (
            revenue_region["Revenue"]
            .apply(
                lambda value:
                f"₹{format_number(value)}"
            )
        )


        region_table = dataframe_to_table(
            revenue_region,
            max_rows=50,
        )


        if region_table:

            story.append(
                region_table
            )


    # ========================================================
    # 7. ANOMALY DETECTION
    # ========================================================

    add_section_title(
        story,
        "6. Anomaly Detection",
        styles,
    )


    anomaly_observations = 0
    unique_affected_rows = 0


    # --------------------------------------------------------
    # Determine unique affected rows
    # --------------------------------------------------------

    if isinstance(
        anomalous_rows,
        pd.DataFrame,
    ):

        unique_affected_rows = len(
            anomalous_rows
        )


    # --------------------------------------------------------
    # Determine anomaly observations
    # --------------------------------------------------------

    if isinstance(
        anomaly_summary,
        dict,
    ):

        possible_keys = [
            "anomaly_observations",
            "total_anomalies",
            "anomalies",
            "count",
        ]

        for key in possible_keys:

            if key in anomaly_summary:

                try:

                    anomaly_observations = int(
                        anomaly_summary[key]
                    )

                    break

                except Exception:

                    pass


    # If anomaly summary isn't a dictionary,
    # don't invent a value.
    #
    # We deliberately keep the value as 0 rather
    # than incorrectly treating affected rows as
    # anomaly observations.


    affected_percentage = (
        (
            unique_affected_rows
            / len(df)
            * 100
        )
        if len(df) > 0
        else 0
    )


    anomaly_data = [
        ["Metric", "Value"],
        [
            "Anomaly Observations",
            f"{anomaly_observations:,}",
        ],
        [
            "Unique Affected Rows",
            f"{unique_affected_rows:,}",
        ],
        [
            "Affected Row Percentage",
            f"{affected_percentage:.2f}%",
        ],
    ]


    add_simple_table(
        story,
        anomaly_data,
        [
            3.0 * inch,
            2.5 * inch,
        ],
    )


    if (
        isinstance(
            anomalous_rows,
            pd.DataFrame,
        )
        and not anomalous_rows.empty
    ):

        story.append(
            Spacer(
                1,
                0.15 * inch,
            )
        )


        story.append(
            Paragraph(
                "<b>Sample of Affected Rows</b>",
                styles["BodyTextCustom"],
            )
        )


        affected_table = dataframe_to_table(
            anomalous_rows,
            max_rows=20,
        )


        if affected_table:

            story.append(
                affected_table
            )


    # ========================================================
    # 8. NATURAL-LANGUAGE ANALYSIS
    # ========================================================

    if (
        question_plan is not None
        or question_result is not None
    ):

        story.append(
            PageBreak()
        )


        add_section_title(
            story,
            "7. Natural-Language Analysis",
            styles,
        )


        if question_plan:

            story.append(
                Paragraph(
                    "<b>Analysis Plan</b>",
                    styles["BodyTextCustom"],
                )
            )


            plan_text = safe_value(
                question_plan
            )


            story.append(
                Paragraph(
                    plan_text,
                    styles["BodyTextCustom"],
                )
            )


        if question_result is not None:

            story.append(
                Spacer(
                    1,
                    0.1 * inch,
                )
            )


            story.append(
                Paragraph(
                    "<b>Analysis Result</b>",
                    styles["BodyTextCustom"],
                )
            )


            if isinstance(
                question_result,
                pd.DataFrame,
            ):

                question_table = dataframe_to_table(
                    question_result,
                    max_rows=30,
                )


                if question_table:

                    story.append(
                        question_table
                    )

            else:

                story.append(
                    Paragraph(
                        safe_value(
                            question_result
                        ),
                        styles["BodyTextCustom"],
                    )
                )


    # ========================================================
    # 9. AI DATASET INSIGHTS
    # ========================================================

    if ai_dataset_summary:

        story.append(
            PageBreak()
        )


        add_section_title(
            story,
            "8. AI Dataset Insights",
            styles,
        )


        add_text_section(
            story,
            ai_dataset_summary,
            styles,
        )


    # ========================================================
    # 10. AI ANOMALY EXPLANATION
    # ========================================================

    if anomaly_ai_insights:

        story.append(
            PageBreak()
        )


        add_section_title(
            story,
            "9. AI Anomaly Explanation",
            styles,
        )


        add_text_section(
            story,
            anomaly_ai_insights,
            styles,
        )


    # ========================================================
    # 11. AI BUSINESS RECOMMENDATIONS
    # ========================================================

    if business_recommendations:

        story.append(
            PageBreak()
        )


        add_section_title(
            story,
            "10. AI Business Recommendations",
            styles,
        )


        add_text_section(
            story,
            business_recommendations,
            styles,
        )


    # ========================================================
    # 12. CONCLUSION
    # ========================================================

    story.append(
        PageBreak()
    )


    add_section_title(
        story,
        "11. Conclusion",
        styles,
    )


    conclusion = (
        f"The AI Data Analyst system analyzed "
        f"the uploaded dataset containing "
        f"{len(df):,} rows and "
        f"{len(df.columns):,} columns. "
        f"The system combines traditional data "
        f"analysis using Pandas with Generative AI "
        f"for natural-language interaction, "
        f"automated visualization, anomaly "
        f"explanation and business-oriented "
        f"insights. "
        f"Numerical dataset calculations are "
        f"performed programmatically, while "
        f"Generative AI is used primarily for "
        f"planning, interpretation and explanation."
    )


    story.append(
        Paragraph(
            conclusion,
            styles["BodyTextCustom"],
        )
    )


    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )


    # Move buffer pointer to beginning
    buffer.seek(0)


    return buffer