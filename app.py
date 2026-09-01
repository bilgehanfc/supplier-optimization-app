"""Streamlit supplier-selection portal.

Run locally with:
    streamlit run app.py

The Supplier Optimization page preserves the existing optimization UI and
imports all mathematical logic from ``supplier_selection_methods.py``.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from supplier_selection_methods import (
    goal_programming,
    preemptive_optimization,
    topsis,
    weighted_sum,
)


st.set_page_config(
    page_title="Supplier Selection Optimization",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Shared data and utility functions
# -----------------------------------------------------------------------------


def default_supplier_data() -> pd.DataFrame:
    """Return the dummy dataset used by the core algorithm module."""
    return pd.DataFrame(
        {
            "Supplier": ["Supplier A", "Supplier B", "Supplier C"],
            "Cost": [100.0, 120.0, 90.0],
            "Quality": [80.0, 90.0, 75.0],
            "Delivery": [10.0, 7.0, 12.0],
            "Capacity": [50.0, 80.0, 70.0],
        }
    )


@st.cache_data(show_spinner=False)
def read_supplier_file(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read CSV/XLSX input without changing the original uploaded file."""
    if file_name.lower().endswith(".csv"):
        data = pd.read_csv(io.BytesIO(file_bytes))
    else:
        data = pd.read_excel(io.BytesIO(file_bytes))
    return standardize_supplier_table(data)


def standardize_supplier_table(data: pd.DataFrame) -> pd.DataFrame:
    """Make common supplier-column variations usable by the app."""
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]

    supplier_column = next(
        (
            column
            for column in data.columns
            if column.lower() in {"supplier", "supplier name", "vendor", "vendor name"}
        ),
        None,
    )

    # Excel files often contain the former index as an "Unnamed: 0" column.
    if supplier_column is None and len(data.columns) > 0:
        first_column = data.columns[0]
        if first_column.lower().startswith("unnamed"):
            supplier_column = first_column

    if supplier_column is not None:
        data = data.rename(columns={supplier_column: "Supplier"})
    else:
        data.insert(0, "Supplier", [f"Supplier {i + 1}" for i in range(len(data))])

    capacity_column = next(
        (column for column in data.columns if column.lower() == "capacity"), None
    )
    if capacity_column is not None and capacity_column != "Capacity":
        data = data.rename(columns={capacity_column: "Capacity"})

    return data


def inferred_impact(column: str) -> str:
    """Choose a useful default impact for familiar cost-like column names."""
    cost_words = ("cost", "price", "fee", "time", "day", "delivery", "lead", "risk")
    return "cost" if any(word in column.lower() for word in cost_words) else "benefit"


def default_column(columns: list[str], preferred: tuple[str, ...]) -> str:
    """Return the first preferred column that exists, otherwise the first column."""
    for column in preferred:
        if column in columns:
            return column
    return columns[0]


def prepare_numeric_data(edited_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the editable table and return criteria data plus numeric values."""
    if "Supplier" not in edited_data.columns:
        raise ValueError("The table must contain a Supplier column")
    if edited_data.empty:
        raise ValueError("Enter at least one supplier")
    if edited_data["Supplier"].isna().any():
        raise ValueError("Supplier names cannot be blank")

    supplier_names = edited_data["Supplier"].astype(str).str.strip()
    if (supplier_names == "").any() or supplier_names.duplicated().any():
        raise ValueError("Supplier names must be non-empty and unique")

    numeric = edited_data.drop(columns=["Supplier"]).apply(pd.to_numeric, errors="coerce")
    if numeric.empty:
        raise ValueError("Add at least one numeric criterion column")
    if numeric.isna().any().any():
        invalid_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(f"These columns contain blank or non-numeric values: {invalid_columns}")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Numeric values must be finite")

    numeric.index = supplier_names
    numeric.index.name = "Supplier"
    criteria = numeric.drop(columns=["Capacity"], errors="ignore")
    return criteria, numeric


def result_for_download(
    result: pd.DataFrame,
    metrics: pd.Series | None = None,
    stages: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Convert displayed results into one CSV-friendly final output table."""
    output = result.reset_index()
    if "index" in output.columns:
        output = output.rename(columns={"index": "Supplier"})

    # Allocation metrics are repeated as metadata columns so the single CSV
    # contains both supplier-level results and the solution summary.
    if metrics is not None:
        for name, value in metrics.items():
            output[name] = value
    if stages is not None:
        for name, value in stages.items():
            output[name] = value
    return output


# -----------------------------------------------------------------------------
# Homepage and database pages
# -----------------------------------------------------------------------------


def homepage_page() -> None:
    """Render the portal homepage."""
    st.title("Supplier Selection Optimization Portal")
    st.caption("A decision-support workspace for supplier, part, and OSA analysis.")

    # Placeholder KPIs for the MVP homepage.
    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
    kpi_1.metric("Total Suppliers", "45")
    kpi_2.metric("Total Parts", "1,240")
    kpi_3.metric("Active Assessments", "18")
    kpi_4.metric("Optimization Scenarios", "32")

    st.divider()
    st.subheader("What are OSA criteria?")
    st.markdown(
        """
        **On-Site Assessment (OSA)** criteria are structured measures used to
        evaluate a supplier during a physical or remote assessment. They can
        cover quality-management maturity, production capability, process
        controls, delivery performance, financial resilience, sustainability,
        safety, and compliance.

        Each criterion should have a clear definition, scoring scale, evidence
        requirement, responsible assessor, and weighting policy. Consistent
        definitions make supplier comparisons more transparent and auditable.
        """
    )

    st.subheader("Optimization methods")
    st.markdown(
        """
        - **TOPSIS:** Ranks suppliers by their distance from an ideal supplier
          and an anti-ideal supplier.
        - **Weighted Sum:** Converts criteria to comparable utilities and
          calculates a weighted overall score.
        - **Preemptive Optimization:** Applies priorities lexicographically;
          higher-priority goals are protected before lower-priority goals.
        - **Goal Programming:** Minimizes weighted deviations from quality,
          delivery, and cost targets.
        """
    )


def parts_database_page() -> None:
    """Render the dummy parts database."""
    st.title("Parts Database")
    parts = pd.DataFrame(
        {
            "Part ID": ["P-1001", "P-1002", "P-1003", "P-1004"],
            "Part Name": [
                "Battery Housing",
                "Brake Caliper",
                "Dashboard Module",
                "Cooling Fan Assembly",
            ],
            "Responsible Engineer": [
                "A. Yılmaz",
                "B. Schmidt",
                "C. Kaya",
                "D. Weber",
            ],
        }
    )
    st.dataframe(parts, width="stretch", hide_index=True)


def supplier_database_page() -> None:
    """Render the dummy supplier database."""
    st.title("Supplier Database")
    suppliers = pd.DataFrame(
        {
            "Supplier Name": ["Supplier A", "Supplier B", "Supplier C", "Supplier D"],
            "Location": ["Stuttgart", "Bursa", "Munich", "Prague"],
            "Capacity": [50, 80, 70, 65],
            "Supplied Part": ["P-1001", "P-1002", "P-1003", "P-1004"],
        }
    )
    st.dataframe(suppliers, width="stretch", hide_index=True)


def osa_assessment_page() -> None:
    """Render the OSA placeholder page."""
    st.title("OSA Assessment")
    st.info("This module is under development.")


# -----------------------------------------------------------------------------
# Existing Supplier Optimization UI
# -----------------------------------------------------------------------------


def display_ranking(result: pd.DataFrame, score_title: str) -> None:
    """Display ranking table and a basic score chart."""
    st.subheader("Ranking results")
    st.dataframe(result, width="stretch")
    st.caption(f"Higher {score_title} values indicate a better supplier.")
    chart_data = result.sort_values("score")[["score"]]
    st.bar_chart(chart_data, horizontal=True)


def display_allocation(
    allocation: pd.DataFrame,
    metrics: pd.Series,
    title: str,
) -> None:
    """Display allocation results, metrics, and a basic allocation chart."""
    st.subheader(title)
    visible_allocation = allocation[allocation["allocation_units"] > 1e-8]
    st.dataframe(visible_allocation, width="stretch")
    st.bar_chart(
        allocation.sort_values("allocation_units")[["allocation_units"]],
        horizontal=True,
    )

    st.subheader("Solution metrics")
    metric_items = list(metrics.items())
    for start in range(0, len(metric_items), 4):
        metric_columns = st.columns(min(4, len(metric_items) - start))
        for column, (name, value) in zip(metric_columns, metric_items[start : start + 4]):
            column.metric(name.replace("_", " ").title(), f"{value:,.4f}")


def run_analysis(
    method: str,
    criteria: pd.DataFrame,
    weights: dict[str, float],
    impacts: dict[str, str],
    numeric_data: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch the selected UI configuration to the core algorithm functions."""
    if method == "TOPSIS":
        result = topsis(criteria, weights, impacts)
        display_ranking(result, "TOPSIS score")
        return result_for_download(result)

    if method == "Weighted Sum":
        result = weighted_sum(criteria, weights, impacts)
        display_ranking(result, "weighted-sum score")
        return result_for_download(result)

    capacity_column = settings["capacity_column"]
    if capacity_column is None:
        raise ValueError(
            "Preemptive Optimization and Goal Programming require a Capacity column"
        )

    capacities = numeric_data[capacity_column].to_dict()
    common_arguments = {
        "data": criteria,
        "demand_units": settings["demand_units"],
        "capacities": capacities,
        "cost_col": settings["cost_column"],
        "quality_col": settings["quality_column"],
        "delivery_col": settings["delivery_column"],
    }

    if method == "Preemptive Optimization":
        allocation, metrics, stages = preemptive_optimization(
            **common_arguments,
            quality_target=settings["quality_target"],
            delivery_target=settings["delivery_target"],
        )
        display_allocation(allocation, metrics, "Preemptive allocation")
        st.subheader("Lexicographic priority results")
        st.dataframe(pd.Series(stages, name="value").to_frame(), width="stretch")
        return result_for_download(allocation, metrics, stages)

    allocation, metrics = goal_programming(
        **common_arguments,
        quality_target=settings["quality_target"],
        delivery_target=settings["delivery_target"],
        cost_target=settings["cost_target"],
        goal_weights=settings["goal_weights"],
    )
    display_allocation(allocation, metrics, "Goal-programming allocation")
    return result_for_download(allocation, metrics)


def supplier_optimization_page() -> None:
    """Render the colleague's existing UI plus CSV result download."""
    # Preserve the colleague's custom visual treatment for this workspace.
    st.markdown(
        """
        <style>
            .stApp {
                background-image: radial-gradient(
                    circle at 8% 0%,
                    color-mix(in srgb, currentColor 5%, transparent) 0,
                    transparent 32rem
                );
            }
            .block-container {
                max-width: 1440px;
                padding-top: calc(5rem + env(safe-area-inset-top));
                padding-bottom: 4rem;
            }
            .st-key-analysis_workspace {
                background: color-mix(in srgb, currentColor 3%, transparent);
                border-color: color-mix(in srgb, currentColor 18%, transparent) !important;
                box-shadow: 0 14px 38px rgba(0, 0, 0, 0.12);
            }
            .workspace-title {
                color: inherit !important;
                font-size: clamp(2rem, 4vw, 3.25rem);
                font-weight: 730;
                letter-spacing: -0.045em;
                line-height: 1.04;
                margin: 0;
            }
            .workspace-copy {
                color: inherit;
                font-size: 1.05rem;
                margin: 0.75rem 0 0;
                max-width: 48rem;
                opacity: 0.72;
            }
            .section-label {
                color: inherit;
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: 0.1rem;
                opacity: 0.92;
            }
            .st-key-analysis_workspace [data-testid="stBaseButton-primary"] {
                background-color: #c0392b;
                border-color: #c0392b;
                color: #ffffff;
            }
            .st-key-analysis_workspace [data-testid="stBaseButton-primary"]:hover {
                background-color: #a93226;
                border-color: #a93226;
            }
            .st-key-analysis_workspace [data-testid="stBaseButton-primary"]:focus-visible {
                outline: 3px solid currentColor;
                outline-offset: 2px;
            }
            @media (max-width: 768px) {
                .block-container {
                    padding-top: calc(4.5rem + env(safe-area-inset-top));
                }
            }
        </style>
        """
    )

    with st.container(border=True, key="analysis_workspace"):
        st.markdown(
            '<h1 class="workspace-title">Supplier selection analysis</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="workspace-copy">Compare supplier performance, tune the decision model, '
            'and calculate the best allocation from one unified workspace.</p>',
            unsafe_allow_html=True,
        )

        st.divider()
        method_column, upload_column = st.columns([1.15, 1], gap="large")
        with method_column:
            st.markdown(
                '<div class="section-label">1 · Choose an analysis</div>',
                unsafe_allow_html=True,
            )
            method = st.selectbox(
                "Optimization method",
                ["TOPSIS", "Weighted Sum", "Preemptive Optimization", "Goal Programming"],
                label_visibility="collapsed",
            )
            st.caption(
                "Rank individual suppliers or optimize a capacity-constrained split award."
            )

        with upload_column:
            st.markdown(
                '<div class="section-label">2 · Add supplier data</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader(
                "Upload supplier data",
                type=["csv", "xlsx", "xls"],
                help="Use columns such as Supplier, Cost, Quality, Delivery, and Capacity.",
                label_visibility="collapsed",
            )
            if uploaded_file is None:
                st.caption("Using the sample dataset · Upload CSV or Excel to replace it")
            else:
                st.caption(f"Using {uploaded_file.name}")

        if uploaded_file is None:
            source_data = default_supplier_data()
        else:
            try:
                source_data = read_supplier_file(uploaded_file.name, uploaded_file.getvalue())
            except Exception as error:  # Streamlit should show a friendly input error.
                st.error(f"Could not read the uploaded file: {error}")
                st.stop()

        st.divider()
        configuration_column, data_column = st.columns([1, 1.7], gap="large")

        with data_column:
            st.markdown('<div class="section-label">Supplier data</div>', unsafe_allow_html=True)
            st.caption("Review or edit the values that will be used in this analysis.")
            edited_data = st.data_editor(
                source_data,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key=f"supplier_data_editor_{uploaded_file.name if uploaded_file else 'sample'}",
            )

        try:
            criteria_data, numeric_data = prepare_numeric_data(edited_data)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        criteria_columns = list(criteria_data.columns)
        if not criteria_columns:
            st.error("At least one criterion is required; Capacity is not a criterion.")
            st.stop()

        with configuration_column:
            st.markdown(
                '<div class="section-label">Model configuration</div>',
                unsafe_allow_html=True,
            )
            st.caption("Weights normalize automatically. Set whether higher or lower is better.")
            weights: dict[str, float] = {}
            impacts: dict[str, str] = {}
            default_weight = 1.0 / len(criteria_columns)
            for criterion in criteria_columns:
                weight_column, impact_column = st.columns([1, 1])
                with weight_column:
                    weights[criterion] = st.number_input(
                        f"{criterion} weight",
                        min_value=0.0,
                        value=default_weight,
                        step=0.05,
                        key=f"weight_{criterion}",
                    )
                with impact_column:
                    impacts[criterion] = st.selectbox(
                        f"{criterion} direction",
                        ["benefit", "cost"],
                        index=0 if inferred_impact(criterion) == "benefit" else 1,
                        key=f"impact_{criterion}",
                    )

            settings: dict[str, Any] = {
                "capacity_column": "Capacity" if "Capacity" in numeric_data.columns else None,
                "demand_units": 100.0,
                "quality_target": 85.0,
                "delivery_target": 8.0,
                "cost_target": 105.0,
                "goal_weights": {"quality": 0.45, "delivery": 0.35, "cost": 0.20},
            }

            if method in {"Preemptive Optimization", "Goal Programming"}:
                st.markdown("##### Allocation settings")
                if settings["capacity_column"] is None:
                    st.warning("Add a Capacity column to use this method.")
                else:
                    settings["demand_units"] = st.number_input(
                        "Demand units",
                        min_value=0.0001,
                        value=100.0,
                        step=10.0,
                    )
                    settings["cost_column"] = st.selectbox(
                        "Cost criterion",
                        criteria_columns,
                        index=criteria_columns.index(
                            default_column(criteria_columns, ("Cost", "Price"))
                        ),
                    )
                    settings["quality_column"] = st.selectbox(
                        "Quality criterion",
                        criteria_columns,
                        index=criteria_columns.index(
                            default_column(criteria_columns, ("Quality",))
                        ),
                    )
                    settings["delivery_column"] = st.selectbox(
                        "Delivery criterion",
                        criteria_columns,
                        index=criteria_columns.index(
                            default_column(criteria_columns, ("Delivery", "Lead Time", "Time"))
                        ),
                    )

                    quality_values = numeric_data[settings["quality_column"]]
                    delivery_values = numeric_data[settings["delivery_column"]]
                    cost_values = numeric_data[settings["cost_column"]]
                    settings["quality_target"] = st.number_input(
                        "Minimum quality target",
                        value=float(quality_values.max()),
                        step=1.0,
                    )
                    settings["delivery_target"] = st.number_input(
                        "Maximum delivery target",
                        value=float(delivery_values.min()),
                        step=1.0,
                    )

                    if method == "Goal Programming":
                        settings["cost_target"] = st.number_input(
                            "Maximum cost target",
                            value=float(cost_values.median()),
                            step=1.0,
                        )
                        st.caption("Goal-programming weights")
                        settings["goal_weights"] = {
                            "quality": st.number_input(
                                "Quality goal weight", min_value=0.0, value=0.45, step=0.05
                            ),
                            "delivery": st.number_input(
                                "Delivery goal weight", min_value=0.0, value=0.35, step=0.05
                            ),
                            "cost": st.number_input(
                                "Cost goal weight", min_value=0.0, value=0.20, step=0.05
                            ),
                        }

            run_button = st.button(
                "Run analysis",
                type="primary",
                width="stretch",
            )

    if run_button:
        try:
            if sum(weights.values()) <= 0:
                raise ValueError("At least one criterion weight must be greater than zero")
            if method in {"Preemptive Optimization", "Goal Programming"}:
                if settings["capacity_column"] is None:
                    raise ValueError("Add a Capacity column before running this method")
                if method == "Goal Programming" and sum(settings["goal_weights"].values()) <= 0:
                    raise ValueError(
                        "At least one goal-programming weight must be greater than zero"
                    )

            final_output = run_analysis(
                method, criteria_data, weights, impacts, numeric_data, settings
            )

            # Provide the exact final table rendered above as an Excel-friendly CSV.
            csv_bytes = final_output.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download final results (CSV)",
                data=csv_bytes,
                file_name="supplier_optimization_results.csv",
                mime="text/csv",
                width="stretch",
            )
        except Exception as error:  # Surface solver and validation errors in the UI.
            st.error(f"Analysis could not be completed: {error}")
    else:
        st.caption("Choose a method, review the model configuration, then run the analysis.")


# -----------------------------------------------------------------------------
# Navigation: exactly five portal pages
# -----------------------------------------------------------------------------


with st.sidebar:
    st.title("Navigation")
    selected_page = st.radio(
        "Go to",
        [
            "Homepage",
            "Parts Database",
            "Supplier Database",
            "OSA Assessment",
            "Supplier Optimization",
        ],
    )


if selected_page == "Homepage":
    homepage_page()
elif selected_page == "Parts Database":
    parts_database_page()
elif selected_page == "Supplier Database":
    supplier_database_page()
elif selected_page == "OSA Assessment":
    osa_assessment_page()
elif selected_page == "Supplier Optimization":
    supplier_optimization_page()
