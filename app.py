"""Streamlit MVP for supplier selection analysis.

Run locally with:
    streamlit run app.py

Expected uploaded table format:
    Supplier, Cost, Quality, Delivery, Capacity

Capacity is required only for Preemptive Optimization and Goal Programming.
For TOPSIS and Weighted Sum, every numeric column except Capacity is treated
as a criterion and can be configured as a benefit or cost criterion in the UI.
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


def display_ranking(result: pd.DataFrame, score_title: str) -> None:
    """Display ranking table and a basic score chart."""
    st.subheader("Ranking results")
    st.dataframe(result, use_container_width=True)
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
    st.dataframe(visible_allocation, use_container_width=True)
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
) -> None:
    """Dispatch the selected UI configuration to the core algorithm functions."""
    if method == "TOPSIS":
        display_ranking(topsis(criteria, weights, impacts), "TOPSIS score")
        return

    if method == "Weighted Sum":
        display_ranking(weighted_sum(criteria, weights, impacts), "weighted-sum score")
        return

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
        st.dataframe(pd.Series(stages, name="value").to_frame(), use_container_width=True)
        return

    allocation, metrics = goal_programming(
        **common_arguments,
        quality_target=settings["quality_target"],
        delivery_target=settings["delivery_target"],
        cost_target=settings["cost_target"],
        goal_weights=settings["goal_weights"],
    )
    display_allocation(allocation, metrics, "Goal-programming allocation")


st.title("Supplier Selection Optimization")
st.caption("Compare supplier rankings or calculate a capacity-constrained allocation.")

uploaded_file = st.file_uploader(
    "Upload supplier data",
    type=["csv", "xlsx", "xls"],
    help="Use columns such as Supplier, Cost, Quality, Delivery, and Capacity.",
)

if uploaded_file is None:
    source_data = default_supplier_data()
    st.info("No file uploaded. The dummy supplier dataset is shown below.")
else:
    try:
        source_data = read_supplier_file(uploaded_file.name, uploaded_file.getvalue())
        st.success(f"Loaded {uploaded_file.name}")
    except Exception as error:  # Streamlit should show a friendly input error.
        st.error(f"Could not read the uploaded file: {error}")
        st.stop()

st.subheader("Supplier data")
edited_data = st.data_editor(
    source_data,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key=f"supplier_data_editor_{uploaded_file.name if uploaded_file else 'dummy'}",
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

with st.sidebar:
    st.header("Analysis setup")
    method = st.selectbox(
        "Optimization method",
        ["TOPSIS", "Weighted Sum", "Preemptive Optimization", "Goal Programming"],
    )

    st.subheader("Criterion configuration")
    st.caption("Weights are normalized automatically. Choose benefit or cost direction.")
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
        st.subheader("Allocation settings")
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
                index=criteria_columns.index(default_column(criteria_columns, ("Cost", "Price"))),
            )
            settings["quality_column"] = st.selectbox(
                "Quality criterion",
                criteria_columns,
                index=criteria_columns.index(default_column(criteria_columns, ("Quality",))),
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

    run_button = st.button("Run analysis", type="primary", use_container_width=True)

if run_button:
    try:
        if sum(weights.values()) <= 0:
            raise ValueError("At least one criterion weight must be greater than zero")
        if method in {"Preemptive Optimization", "Goal Programming"}:
            if settings["capacity_column"] is None:
                raise ValueError("Add a Capacity column before running this method")
            if method == "Goal Programming" and sum(settings["goal_weights"].values()) <= 0:
                raise ValueError("At least one goal-programming weight must be greater than zero")

        run_analysis(method, criteria_data, weights, impacts, numeric_data, settings)
    except Exception as error:  # Surface solver and validation errors in the UI.
        st.error(f"Analysis could not be completed: {error}")
else:
    st.info("Configure the method in the sidebar, then click Run analysis.")
