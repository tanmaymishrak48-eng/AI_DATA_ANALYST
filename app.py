import pandas as pd
import plotly.express as px
import streamlit as st

from ai_engine import (
    generate_dataset_summary,
    create_analysis_plan,
    create_visualization_plan,
    generate_anomaly_insights,
    generate_business_recommendations,
)

from anomaly_detector import (
    detect_numerical_anomalies,
    get_anomaly_summary,
    get_anomalous_rows,
)

from report_generator import generate_pdf_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #111827 !important;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #475569 !important;
        margin-bottom: 1.5rem;
    }

    .section-header {
        font-size: 1.45rem;
        font-weight: 650;
        color: #111827 !important;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #334155 !important;
    }

    [data-testid="stAppViewContainer"]
    [data-testid="stMarkdownContainer"] {
        color: #111827 !important;
    }

    p {
        color: #334155;
    }

    label {
        color: #334155 !important;
    }

    input,
    textarea {
        color: #111827 !important;
    }

    [data-baseweb="select"] {
        color: #111827 !important;
    }

    [data-testid="stFileUploader"] label {
        color: #334155 !important;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    button[data-baseweb="tab"] {
        color: #334155 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATES = {
    "data": None,
    "file_name": None,

    "question_plan": None,
    "question_result": None,

    "visualization_plan": None,
    "visualization_figure": None,

    "anomaly_results": None,
    "anomaly_summary": None,
    "anomalous_rows": None,
    "anomaly_ai_insights": None,

    "business_recommendations": None,

    "ai_dataset_summary": None,
}


for key, value in DEFAULT_STATES.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_analysis_states():
    """Reset all calculated/AI results for a new dataset."""

    keys = [
        "question_plan",
        "question_result",
        "visualization_plan",
        "visualization_figure",
        "anomaly_results",
        "anomaly_summary",
        "anomalous_rows",
        "anomaly_ai_insights",
        "business_recommendations",
        "ai_dataset_summary",
    ]

    for key in keys:
        st.session_state[key] = None


def dataframe_for_display(dataframe):
    """
    Convert object columns to strings before displaying.
    Helps prevent Streamlit/PyArrow mixed-type errors.
    """

    display_df = dataframe.copy()

    for column in display_df.columns:

        if display_df[column].dtype == "object":

            display_df[column] = (
                display_df[column].astype(str)
            )

    return display_df


def format_number(value):
    """Format numbers for the dashboard."""

    if value is None:
        return "N/A"

    try:

        if pd.isna(value):
            return "N/A"

    except Exception:
        pass

    if isinstance(value, float):
        return f"{value:,.2f}"

    return f"{value:,}"


def create_kpi(title, value, description=""):
    """Create a clean native Streamlit KPI card."""

    with st.container(border=True):

        st.metric(
            label=title,
            value=value,
        )

        if description:

            st.caption(description)


def load_uploaded_file(uploaded_file):
    """Load CSV/XLSX/XLS files."""

    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if filename.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    if filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError(
        "Unsupported file format. "
        "Please upload CSV or Excel."
    )


def calculate_basic_metrics(dataframe):
    """Calculate basic dataset/business metrics."""

    metrics = {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing": int(
            dataframe.isna().sum().sum()
        ),
        "duplicates": int(
            dataframe.duplicated().sum()
        ),
        "revenue": None,
        "profit": None,
        "quantity": None,
    }

    if "Revenue" in dataframe.columns:

        metrics["revenue"] = float(
            dataframe["Revenue"].sum()
        )

    if "Profit" in dataframe.columns:

        metrics["profit"] = float(
            dataframe["Profit"].sum()
        )

    if "Quantity" in dataframe.columns:

        metrics["quantity"] = float(
            dataframe["Quantity"].sum()
        )

    return metrics


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.55rem;
            font-weight:700;
            color:#111827;
        ">
            🤖 AI Data Analyst
        </div>

        <div style="
            color:#64748b;
            font-size:0.85rem;
            margin-bottom:1.2rem;
        ">
            Generative AI Data Analysis Platform
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # DATASET UPLOAD
    # --------------------------------------------------------

    st.markdown("### 📂 Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Upload the dataset you want to analyze.",
    )


    if uploaded_file is not None:

        if (
            st.session_state.file_name
            != uploaded_file.name
        ):

            try:

                dataframe = load_uploaded_file(
                    uploaded_file
                )

                st.session_state.data = dataframe

                st.session_state.file_name = (
                    uploaded_file.name
                )

                reset_analysis_states()

            except Exception as error:

                st.error(
                    f"Unable to load file: {error}"
                )


    st.divider()


    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    st.markdown("### 🧭 Navigation")

    page = st.radio(
        "Go to",
        [
            "📊 Dashboard",
            "📈 Analytics",
            "💬 Ask Data",
            "🤖 AI Insights",
            "🚨 Anomalies",
            "💡 Recommendations",
            "📄 Reports",
        ],
        label_visibility="collapsed",
    )


    st.divider()


    # --------------------------------------------------------
    # DATASET INFO
    # --------------------------------------------------------

    if st.session_state.data is not None:

        dataframe = st.session_state.data

        metrics = calculate_basic_metrics(
            dataframe
        )

        st.markdown("### 📋 Dataset Info")

        st.write(
            f"**File:** "
            f"{st.session_state.file_name}"
        )

        st.write(
            f"**Rows:** {metrics['rows']:,}"
        )

        st.write(
            f"**Columns:** {metrics['columns']:,}"
        )

        if metrics["missing"] == 0:

            st.success(
                "No missing values"
            )

        else:

            st.warning(
                f"{metrics['missing']:,} "
                "missing values"
            )


        st.markdown("### 🤖 AI Status")

        st.markdown(
            """
            <div style="
                padding:10px 14px;
                border-radius:10px;
                background:#f1f5f9;
                border:1px solid #e2e8f0;
                color:#334155;
                font-size:0.9rem;
            ">
                🟢 Groq AI integration configured
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "Upload a dataset to begin analysis."
        )


# ============================================================
# NO DATASET
# ============================================================

if st.session_state.data is None:

    st.markdown(
        '<div class="main-title">'
        '🤖 AI Data Analyst'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            A Generative AI-powered platform for
            natural-language data analysis,
            visualization and business insights.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "👈 Upload a CSV or Excel dataset "
        "from the sidebar to start."
    )


    st.markdown(
        "### 🚀 What this system can do"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            "#### 📊 Data Analysis"
        )

        st.write(
            """
            Explore dataset structure,
            statistics, missing values
            and business metrics.
            """
        )


    with col2:

        st.markdown(
            "#### 🤖 Generative AI"
        )

        st.write(
            """
            Ask questions using natural
            language and generate
            AI-powered explanations.
            """
        )


    with col3:

        st.markdown(
            "#### 📈 Visual Analytics"
        )

        st.write(
            """
            Automatically create
            visualizations and identify
            important patterns.
            """
        )


    st.stop()


# ============================================================
# DATASET AVAILABLE
# ============================================================

df = st.session_state.data

metrics = calculate_basic_metrics(df)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '📊 Dashboard'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="subtitle">
            Overview of
            <b>{st.session_state.file_name}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # KPI ROW 1
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        create_kpi(
            "Rows",
            format_number(metrics["rows"]),
            "Total observations",
        )


    with col2:

        create_kpi(
            "Columns",
            format_number(metrics["columns"]),
            "Dataset features",
        )


    with col3:

        create_kpi(
            "Missing Values",
            format_number(metrics["missing"]),
            "Total missing cells",
        )


    with col4:

        create_kpi(
            "Duplicate Rows",
            format_number(metrics["duplicates"]),
            "Repeated records",
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # KPI ROW 2
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        if metrics["revenue"] is not None:

            create_kpi(
                "Total Revenue",
                f"₹{metrics['revenue']:,.2f}",
                "Total revenue",
            )

        else:

            create_kpi(
                "Total Revenue",
                "N/A",
                "Revenue column not found",
            )


    with col2:

        if metrics["profit"] is not None:

            create_kpi(
                "Total Profit",
                f"₹{metrics['profit']:,.2f}",
                "Total profit",
            )

        else:

            create_kpi(
                "Total Profit",
                "N/A",
                "Profit column not found",
            )


    with col3:

        if metrics["quantity"] is not None:

            create_kpi(
                "Total Quantity",
                format_number(metrics["quantity"]),
                "Total units",
            )

        else:

            create_kpi(
                "Total Quantity",
                "N/A",
                "Quantity column not found",
            )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # BUSINESS OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '📈 Business Overview'
        '</div>',
        unsafe_allow_html=True,
    )


    chart_col1, chart_col2 = st.columns(2)


    with chart_col1:

        if (
            "Region" in df.columns
            and "Revenue" in df.columns
        ):

            region_revenue = (
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

            fig = px.bar(
                region_revenue,
                x="Region",
                y="Revenue",
                title="Revenue by Region",
                text_auto=".2s",
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="dashboard_revenue_region",
            )

        else:

            st.info(
                "Region and Revenue columns "
                "are required."
            )


    with chart_col2:

        if (
            "Category" in df.columns
            and "Profit" in df.columns
        ):

            category_profit = (
                df.groupby(
                    "Category",
                    as_index=False,
                )["Profit"]
                .sum()
                .sort_values(
                    by="Profit",
                    ascending=False,
                )
            )

            fig = px.bar(
                category_profit,
                x="Category",
                y="Profit",
                title="Profit by Category",
                text_auto=".2s",
            )

            fig.update_layout(
                height=400,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="dashboard_profit_category",
            )

        else:

            st.info(
                "Category and Profit columns "
                "are required."
            )


    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '🧹 Data Quality'
        '</div>',
        unsafe_allow_html=True,
    )


    quality_col1, quality_col2, quality_col3 = (
        st.columns(3)
    )


    with quality_col1:

        if metrics["missing"] == 0:

            st.success(
                "✅ No missing values"
            )

        else:

            st.warning(
                f"⚠️ {metrics['missing']:,} "
                "missing values detected"
            )


    with quality_col2:

        if metrics["duplicates"] == 0:

            st.success(
                "✅ No duplicate rows"
            )

        else:

            st.warning(
                f"⚠️ {metrics['duplicates']:,} "
                "duplicate rows detected"
            )


    with quality_col3:

        numeric_count = len(
            df.select_dtypes(
                include="number"
            ).columns
        )

        categorical_count = len(
            df.select_dtypes(
                include=[
                    "object",
                    "category",
                ]
            ).columns
        )

        st.info(
            f"🔢 {numeric_count} numeric | "
            f"🔤 {categorical_count} categorical"
        )


# ============================================================
# PAGE 2 — ANALYTICS
# ============================================================

elif page == "📈 Analytics":

    st.markdown(
        '<div class="main-title">'
        '📈 Analytics'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Explore your dataset using statistical
            analysis and visual analytics.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # DATASET OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '📋 Dataset Overview'
        '</div>',
        unsafe_allow_html=True,
    )


    overview_col1, overview_col2 = st.columns(2)


    with overview_col1:

        st.write("**Shape**")

        shape_df = pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                ],
                "Value": [
                    df.shape[0],
                    df.shape[1],
                ],
            }
        )

        st.dataframe(
            shape_df,
            use_container_width=True,
            hide_index=True,
        )


    with overview_col2:

        st.write("**Column Types**")

        type_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in df.dtypes
                ],
            }
        )

        st.dataframe(
            type_df,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '👀 Dataset Preview'
        '</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        dataframe_for_display(
            df.head(10)
        ),
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '🧹 Data Quality'
        '</div>',
        unsafe_allow_html=True,
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

    missing_df = missing_df.sort_values(
        by="Missing Values",
        ascending=False,
    )

    st.dataframe(
        missing_df,
        use_container_width=True,
        hide_index=True,
    )


    # --------------------------------------------------------
    # STATISTICAL SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '📊 Statistical Summary'
        '</div>',
        unsafe_allow_html=True,
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

        st.dataframe(
            dataframe_for_display(stats),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as error:

        st.error(
            f"Unable to generate statistical "
            f"summary: {error}"
        )


    # --------------------------------------------------------
    # VISUAL ANALYTICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '📈 Visual Analytics'
        '</div>',
        unsafe_allow_html=True,
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


    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Distribution",
            "📈 Category Comparison",
            "🔗 Correlation",
        ]
    )


    with tab1:

        if numeric_columns:

            selected_numeric = st.selectbox(
                "Select numerical column",
                numeric_columns,
                key="analytics_histogram_column",
            )

            fig = px.histogram(
                df,
                x=selected_numeric,
                title=(
                    f"Distribution of "
                    f"{selected_numeric}"
                ),
                nbins=30,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="analytics_histogram",
            )

        else:

            st.info(
                "No numerical columns available."
            )


    with tab2:

        if (
            categorical_columns
            and numeric_columns
        ):

            selected_category = st.selectbox(
                "Category",
                categorical_columns,
                key="analytics_category_column",
            )

            selected_value = st.selectbox(
                "Numerical value",
                numeric_columns,
                key="analytics_value_column",
            )

            grouped = (
                df.groupby(
                    selected_category,
                    as_index=False,
                )[selected_value]
                .mean()
                .sort_values(
                    by=selected_value,
                    ascending=False,
                )
            )

            fig = px.bar(
                grouped,
                x=selected_category,
                y=selected_value,
                title=(
                    f"Average {selected_value} "
                    f"by {selected_category}"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="analytics_category_chart",
            )

        else:

            st.info(
                "Categorical and numerical "
                "columns are required."
            )


    with tab3:

        if len(numeric_columns) >= 2:

            correlation = (
                df[numeric_columns]
                .corr()
            )

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                title="Correlation Matrix",
                aspect="auto",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="analytics_correlation",
            )

        else:

            st.info(
                "At least two numerical columns "
                "are required."
            )


# ============================================================
# PAGE 3 — ASK DATA
# ============================================================

elif page == "💬 Ask Data":

    st.markdown(
        '<div class="main-title">'
        '💬 Ask Data'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Ask questions about your dataset
            using natural language.
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.info(
        "Example: What is the total revenue? "
        "Which region has the highest revenue?"
    )


    question = st.text_area(
        "Ask a question",
        placeholder=(
            "Example: Show me the average "
            "profit by region"
        ),
        height=100,
        key="question_input",
    )


    if st.button(
        "🤖 Analyze Question",
        type="primary",
        key="ask_data_button",
    ):

        # Clear old result BEFORE running a new question.
        st.session_state.question_plan = None
        st.session_state.question_result = None


        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "AI is analyzing your question..."
            ):

                try:

                    # ------------------------------------------------
                    # Groq creates the structured plan
                    # ------------------------------------------------

                    plan = create_analysis_plan(
                        df,
                        question,
                    )

                    st.session_state.question_plan = (
                        plan
                    )


                    operation = plan.get(
                        "operation"
                    )

                    column = plan.get(
                        "column"
                    )

                    group_by = plan.get(
                        "group_by"
                    )


                    result = None


                    # ------------------------------------------------
                    # SUM
                    # ------------------------------------------------

                    if operation == "sum":

                        result = df[column].sum()


                    # ------------------------------------------------
                    # MEAN
                    # ------------------------------------------------

                    elif operation == "mean":

                        result = df[column].mean()


                    # ------------------------------------------------
                    # COUNT
                    # ------------------------------------------------

                    elif operation == "count":

                        result = len(df)


                    # ------------------------------------------------
                    # COUNT BY GROUP
                    # ------------------------------------------------

                    elif operation == "count_by_group":

                        result = (
                            df.groupby(
                                group_by,
                                as_index=False,
                            )
                            .size()
                            .rename(
                                columns={
                                    "size": "Count"
                                }
                            )
                        )


                    # ------------------------------------------------
                    # SUM BY GROUP
                    # ------------------------------------------------

                    elif operation == "sum_by_group":

                        result = (
                            df.groupby(
                                group_by,
                                as_index=False,
                            )[column]
                            .sum()
                            .sort_values(
                                by=column,
                                ascending=False,
                            )
                        )


                    # ------------------------------------------------
                    # MEAN BY GROUP
                    # ------------------------------------------------

                    elif operation == "mean_by_group":

                        result = (
                            df.groupby(
                                group_by,
                                as_index=False,
                            )[column]
                            .mean()
                            .sort_values(
                                by=column,
                                ascending=False,
                            )
                        )


                    # ------------------------------------------------
                    # MAX BY GROUP
                    #
                    # FIXED:
                    # Series.sort_values() was previously receiving
                    # the column as a positional argument.
                    #
                    # We now keep the grouped result as a DataFrame
                    # and explicitly use by=column.
                    # ------------------------------------------------

                    elif operation == "max_by_group":

                        result = (
                            df.groupby(
                                group_by,
                                as_index=False,
                            )[column]
                            .sum()
                            .sort_values(
                                by=column,
                                ascending=False,
                            )
                            .head(1)
                        )


                    # ------------------------------------------------
                    # UNKNOWN OPERATION
                    # ------------------------------------------------

                    else:

                        raise ValueError(
                            "The AI generated an "
                            "unsupported analysis operation."
                        )


                    # ------------------------------------------------
                    # SAVE RESULT
                    # ------------------------------------------------

                    st.session_state.question_result = (
                        result
                    )


                except Exception as error:

                    st.session_state.question_plan = None
                    st.session_state.question_result = None

                    st.error(
                        f"Unable to analyze question: "
                        f"{error}"
                    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if (
        st.session_state.question_result
        is not None
    ):

        st.markdown(
            '<div class="section-header">'
            '📌 Result'
            '</div>',
            unsafe_allow_html=True,
        )


        result = (
            st.session_state.question_result
        )


        if isinstance(
            result,
            pd.DataFrame,
        ):

            st.dataframe(
                dataframe_for_display(
                    result
                ),
                use_container_width=True,
                hide_index=True,
            )


        elif isinstance(
            result,
            (int, float),
        ):

            st.metric(
                "Calculated Result",
                f"{result:,.2f}",
            )


        else:

            st.write(result)


    if (
        st.session_state.question_plan
        is not None
    ):

        with st.expander(
            "🔍 View AI Analysis Plan"
        ):

            st.json(
                st.session_state.question_plan
            )


# ============================================================
# PAGE 4 — AI INSIGHTS
# ============================================================

elif page == "🤖 AI Insights":

    st.markdown(
        '<div class="main-title">'
        '🤖 AI Insights'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Use Generative AI to understand your
            dataset and automatically generate
            visual analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # DATASET INSIGHTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '🧠 AI Dataset Insights'
        '</div>',
        unsafe_allow_html=True,
    )


    if st.button(
        "✨ Generate AI Dataset Insights",
        type="primary",
        key="dataset_insights_button",
    ):

        with st.spinner(
            "Groq is generating dataset insights..."
        ):

            try:

                insights = (
                    generate_dataset_summary(
                        df
                    )
                )

                st.session_state.ai_dataset_summary = (
                    insights
                )

                st.success(
                    "AI analysis completed!"
                )

            except Exception as error:

                st.error(
                    str(error)
                )


    if st.session_state.ai_dataset_summary:

        st.markdown(
            st.session_state.ai_dataset_summary
        )


    # --------------------------------------------------------
    # AUTOMATIC VISUALIZATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '🎨 AI Automatic Visualization'
        '</div>',
        unsafe_allow_html=True,
    )


    visualization_question = st.text_input(
        "What would you like to visualize?",
        placeholder=(
            "Example: Show me revenue by region"
        ),
        key="visualization_question",
    )


    if st.button(
        "🎨 Generate Visualization",
        type="primary",
        key="auto_visualization_button",
    ):

        st.session_state.visualization_plan = None
        st.session_state.visualization_figure = None


        if not visualization_question.strip():

            st.warning(
                "Please describe the visualization "
                "you want."
            )

        else:

            with st.spinner(
                "AI is creating the visualization..."
            ):

                try:

                    plan = (
                        create_visualization_plan(
                            df,
                            visualization_question,
                        )
                    )

                    st.session_state.visualization_plan = (
                        plan
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


                    # ------------------------------------------------
                    # BAR
                    # ------------------------------------------------

                    if chart_type == "bar":

                        grouped = (
                            df.groupby(
                                x_column,
                                as_index=False,
                            )[y_column]
                            .sum()
                            .sort_values(
                                by=y_column,
                                ascending=False,
                            )
                        )

                        fig = px.bar(
                            grouped,
                            x=x_column,
                            y=y_column,
                            title=(
                                visualization_question
                            ),
                        )


                    # ------------------------------------------------
                    # LINE
                    # ------------------------------------------------

                    elif chart_type == "line":

                        grouped = (
                            df.groupby(
                                x_column,
                                as_index=False,
                            )[y_column]
                            .sum()
                        )

                        fig = px.line(
                            grouped,
                            x=x_column,
                            y=y_column,
                            title=(
                                visualization_question
                            ),
                        )


                    # ------------------------------------------------
                    # HISTOGRAM
                    # ------------------------------------------------

                    elif chart_type == "histogram":

                        fig = px.histogram(
                            df,
                            x=x_column,
                            title=(
                                visualization_question
                            ),
                        )


                    # ------------------------------------------------
                    # SCATTER
                    # ------------------------------------------------

                    elif chart_type == "scatter":

                        fig = px.scatter(
                            df,
                            x=x_column,
                            y=y_column,
                            title=(
                                visualization_question
                            ),
                        )


                    else:

                        raise ValueError(
                            f"Unsupported chart type: "
                            f"{chart_type}"
                        )


                    st.session_state.visualization_figure = (
                        fig
                    )


                except Exception as error:

                    st.session_state.visualization_plan = None
                    st.session_state.visualization_figure = None

                    st.error(
                        f"Unable to generate "
                        f"visualization: {error}"
                    )


    if (
        st.session_state.visualization_figure
        is not None
    ):

        st.plotly_chart(
            st.session_state.visualization_figure,
            use_container_width=True,
            key="ai_generated_visualization",
        )


        if st.session_state.visualization_plan:

            with st.expander(
                "🔍 View Visualization Plan"
            ):

                st.json(
                    st.session_state.visualization_plan
                )


# ============================================================
# PAGE 5 — ANOMALIES
# ============================================================

elif page == "🚨 Anomalies":

    st.markdown(
        '<div class="main-title">'
        '🚨 Anomaly Detection'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Detect unusual numerical observations
            using statistical analysis and explain
            them using Generative AI.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    if st.button(
        "🔎 Detect Anomalies",
        type="primary",
        key="detect_anomalies_button",
    ):

        with st.spinner(
            "Analyzing numerical columns for anomalies..."
        ):

            try:

                anomaly_results = (
                    detect_numerical_anomalies(
                        df
                    )
                )

                anomaly_summary = (
                    get_anomaly_summary(
                        anomaly_results
                    )
                )

                anomalous_rows = (
                    get_anomalous_rows(
                        df,
                        anomaly_results,
                    )
                )


                st.session_state.anomaly_results = (
                    anomaly_results
                )

                st.session_state.anomaly_summary = (
                    anomaly_summary
                )

                st.session_state.anomalous_rows = (
                    anomalous_rows
                )

                st.session_state.anomaly_ai_insights = None

                st.success(
                    "Anomaly detection completed!"
                )


            except Exception as error:

                st.error(
                    f"Unable to detect anomalies: "
                    f"{error}"
                )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if (
        st.session_state.anomaly_results
        is not None
    ):

        anomaly_results = (
            st.session_state.anomaly_results
        )

        anomalous_rows = (
            st.session_state.anomalous_rows
        )


        anomaly_observations = len(
            anomaly_results
        )


        unique_affected_rows = (
            len(anomalous_rows)
            if anomalous_rows is not None
            else 0
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


        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            create_kpi(
                "Anomaly Observations",
                format_number(
                    anomaly_observations
                ),
                "Total anomaly flags",
            )


        with col2:

            create_kpi(
                "Unique Affected Rows",
                format_number(
                    unique_affected_rows
                ),
                "Rows containing anomalies",
            )


        with col3:

            create_kpi(
                "Affected Row %",
                f"{affected_percentage:.2f}%",
                "Percentage of dataset rows",
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # ----------------------------------------------------
        # DETECTED ANOMALIES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-header">'
            '🔎 Detected Anomalies'
            '</div>',
            unsafe_allow_html=True,
        )


        st.dataframe(
            dataframe_for_display(
                anomaly_results
            ),
            use_container_width=True,
            hide_index=True,
        )


        # ----------------------------------------------------
        # AFFECTED ROWS
        # ----------------------------------------------------

        if (
            anomalous_rows is not None
            and len(anomalous_rows) > 0
        ):

            st.markdown(
                '<div class="section-header">'
                '📋 Affected Rows'
                '</div>',
                unsafe_allow_html=True,
            )


            st.dataframe(
                dataframe_for_display(
                    anomalous_rows
                ),
                use_container_width=True,
                hide_index=True,
            )


        # ----------------------------------------------------
        # AI ANOMALY EXPLANATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-header">'
            '🤖 AI Anomaly Insights'
            '</div>',
            unsafe_allow_html=True,
        )


        if st.button(
            "🧠 Explain Anomalies with AI",
            type="primary",
            key="anomaly_ai_button",
        ):

            with st.spinner(
                "Groq is analyzing the detected anomalies..."
            ):

                try:

                    insights = (
                        generate_anomaly_insights(
                            df,
                            anomaly_results,
                        )
                    )

                    st.session_state.anomaly_ai_insights = (
                        insights
                    )

                except Exception as error:

                    st.error(
                        str(error)
                    )


        if st.session_state.anomaly_ai_insights:

            st.markdown(
                st.session_state.anomaly_ai_insights
            )


# ============================================================
# PAGE 6 — BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.markdown(
        '<div class="main-title">'
        '💡 Business Recommendations'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Generate evidence-based business
            recommendations using your dataset
            and detected anomalies.
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.info(
        """
        Recommendations are generated from
        verified calculations performed on
        the uploaded dataset.
        """
    )


    if st.button(
        "💡 Generate Business Recommendations",
        type="primary",
        key="recommendations_button",
    ):

        with st.spinner(
            "Groq is generating business recommendations..."
        ):

            try:

                recommendations = (
                    generate_business_recommendations(
                        df,
                        st.session_state.anomaly_results,
                    )
                )

                st.session_state.business_recommendations = (
                    recommendations
                )


            except Exception as error:

                st.error(
                    str(error)
                )


    if st.session_state.business_recommendations:

        st.markdown(
            st.session_state.business_recommendations
        )

    else:

        st.write(
            """
            Click the button above to generate
            AI-powered business recommendations.
            """
        )


# ============================================================
# PAGE 7 — REPORTS
# ============================================================

elif page == "📄 Reports":

    st.markdown(
        '<div class="main-title">'
        '📄 Reports'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
            Generate a complete PDF report containing
            your data analysis, metrics, anomalies
            and available AI insights.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # REPORT CONTENT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-header">'
        '📋 Report Contents'
        '</div>',
        unsafe_allow_html=True,
    )


    report_items = [
        "Dataset overview",
        "Data quality analysis",
        "Statistical summary",
        "Key business metrics",
        "Revenue by region",
        "Anomaly detection",
        "Natural-language analysis",
        "AI dataset insights",
        "AI anomaly explanation",
        "AI business recommendations",
        "Conclusion",
    ]


    for item in report_items:

        st.write(
            f"✓ {item}"
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # PDF GENERATION
    # --------------------------------------------------------

    if st.button(
        "📄 Generate Full Analysis Report",
        type="primary",
        key="generate_report_button",
    ):

        with st.spinner(
            "Generating PDF report..."
        ):

            try:

                pdf_data = (
                    generate_pdf_report(
                        df=df,
                        file_name=(
                            st.session_state.file_name
                        ),
                        anomaly_summary=(
                            st.session_state.anomaly_summary
                        ),
                        anomalous_rows=(
                            st.session_state.anomalous_rows
                        ),
                        question_plan=(
                            st.session_state.question_plan
                        ),
                        question_result=(
                            st.session_state.question_result
                        ),
                        ai_dataset_summary=(
                            st.session_state.ai_dataset_summary
                        ),
                        anomaly_ai_insights=(
                            st.session_state.anomaly_ai_insights
                        ),
                        business_recommendations=(
                            st.session_state.business_recommendations
                        ),
                    )
                )


                st.success(
                    "PDF report generated successfully!"
                )


                st.download_button(
                    label=(
                        "⬇️ Download "
                        "AI Data Analyst Report"
                    ),
                    data=pdf_data,
                    file_name=(
                        "AI_Data_Analyst_Report.pdf"
                    ),
                    mime="application/pdf",
                    key="download_report_button",
                )


            except Exception as error:

                st.error(
                    f"Unable to generate report: "
                    f"{error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        font-size:0.8rem;
        padding:20px;
        border-top:1px solid #e5e7eb;
    ">
        🤖 AI Data Analyst |
        Generative AI for Natural-Language
        Data Analysis, Visualization and
        Business Insights
    </div>
    """,
    unsafe_allow_html=True,
)