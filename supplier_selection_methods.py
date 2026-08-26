"""Reference implementation for supplier-selection methods.

Dependencies:
    pip install numpy pandas scipy

The ranking methods (TOPSIS and weighted sum) score individual suppliers.
The optimization methods model a split award: x_i is the number of demand
units allocated to supplier i. This makes capacity, service-level, and target
constraints explicit.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def _numeric_series(
    values: Mapping[str, float] | list[float] | tuple[float, ...],
    labels: pd.Index,
    name: str,
) -> pd.Series:
    """Convert a mapping/list to a labelled numeric Series and validate it."""
    if isinstance(values, Mapping):
        series = pd.Series(values, dtype=float).reindex(labels)
    else:
        series = pd.Series(values, index=labels, dtype=float)

    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{name} must contain one finite value for every label")
    return series


def _validate_criteria(
    data: pd.DataFrame,
    weights: Mapping[str, float] | list[float] | tuple[float, ...],
    impacts: Mapping[str, str] | list[str] | tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Validate a supplier-by-criterion matrix and normalize its weights."""
    if data.empty:
        raise ValueError("data must contain at least one supplier and one criterion")
    if data.index.has_duplicates:
        raise ValueError("supplier names must be unique")
    if data.columns.has_duplicates:
        raise ValueError("criterion names must be unique")

    matrix = data.astype(float)
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("criteria data must contain only finite numbers")

    weight_series = _numeric_series(weights, matrix.columns, "weights")
    if (weight_series < 0).any() or weight_series.sum() <= 0:
        raise ValueError("weights must be non-negative and have a positive sum")
    weight_series = weight_series / weight_series.sum()

    if isinstance(impacts, Mapping):
        raw_impacts = pd.Series(impacts, dtype="object").reindex(matrix.columns)
    else:
        raw_impacts = pd.Series(impacts, index=matrix.columns, dtype="object")
    if raw_impacts.isna().any():
        raise ValueError("an impact value is missing for one or more criteria")
    impact_series = raw_impacts
    impact_series = impact_series.astype(str).str.lower()
    valid_impacts = {"benefit", "cost"}
    if not set(impact_series).issubset(valid_impacts):
        raise ValueError("each impact must be either 'benefit' or 'cost'")

    return matrix, weight_series, impact_series


def topsis(
    data: pd.DataFrame,
    weights: Mapping[str, float] | list[float] | tuple[float, ...],
    impacts: Mapping[str, str] | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Rank suppliers using TOPSIS.

    Benefit criteria are maximized and cost criteria are minimized. Euclidean
    vector normalization is used before applying the criterion weights.
    """
    matrix, weight_series, impact_series = _validate_criteria(data, weights, impacts)

    denominator = np.sqrt((matrix**2).sum(axis=0))
    if (denominator == 0).any():
        raise ValueError("TOPSIS cannot normalize an all-zero criterion")

    normalized = matrix.div(denominator, axis=1)
    weighted = normalized.mul(weight_series, axis=1)

    ideal_best = np.array(
        [
            weighted[column].max()
            if impact_series[column] == "benefit"
            else weighted[column].min()
            for column in matrix.columns
        ]
    )
    ideal_worst = np.array(
        [
            weighted[column].min()
            if impact_series[column] == "benefit"
            else weighted[column].max()
            for column in matrix.columns
        ]
    )

    distance_to_best = np.sqrt(((weighted.to_numpy() - ideal_best) ** 2).sum(axis=1))
    distance_to_worst = np.sqrt(((weighted.to_numpy() - ideal_worst) ** 2).sum(axis=1))
    denominator = distance_to_best + distance_to_worst
    closeness = np.divide(
        distance_to_worst,
        denominator,
        out=np.full_like(distance_to_worst, 0.5),
        where=denominator != 0,
    )

    result = pd.DataFrame({"score": closeness}, index=matrix.index)
    result["rank"] = result["score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["rank", "score"], ascending=[True, False])


def _min_max_normalize(matrix: pd.DataFrame, impacts: pd.Series) -> pd.DataFrame:
    """Map each criterion to [0, 1], with 1 representing the preferred value."""
    normalized = pd.DataFrame(index=matrix.index, columns=matrix.columns, dtype=float)
    for column in matrix.columns:
        values = matrix[column]
        low, high = values.min(), values.max()
        if np.isclose(low, high):
            # A constant criterion cannot distinguish suppliers.
            normalized[column] = 1.0
        elif impacts[column] == "benefit":
            normalized[column] = (values - low) / (high - low)
        else:
            normalized[column] = (high - values) / (high - low)
    return normalized


def weighted_sum(
    data: pd.DataFrame,
    weights: Mapping[str, float] | list[float] | tuple[float, ...],
    impacts: Mapping[str, str] | list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Rank suppliers using a weighted sum of min-max utilities."""
    matrix, weight_series, impact_series = _validate_criteria(data, weights, impacts)
    utilities = _min_max_normalize(matrix, impact_series)
    scores = utilities.mul(weight_series, axis=1).sum(axis=1)

    result = pd.DataFrame({"score": scores}, index=matrix.index)
    result["rank"] = result["score"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["rank", "score"], ascending=[True, False])


def _allocation_setup(
    data: pd.DataFrame,
    demand_units: float,
    capacities: Mapping[str, float] | list[float] | tuple[float, ...],
    cost_col: str,
    quality_col: str,
    delivery_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the LP inputs and return criteria plus supplier capacities."""
    if demand_units <= 0 or not np.isfinite(demand_units):
        raise ValueError("demand_units must be a positive finite number")
    required = [cost_col, quality_col, delivery_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"missing optimization columns: {missing}")
    if data.index.has_duplicates:
        raise ValueError("supplier names must be unique")

    criteria = data[required].astype(float)
    if not np.isfinite(criteria.to_numpy()).all():
        raise ValueError("optimization criteria must contain only finite numbers")
    if (criteria < 0).any().any():
        raise ValueError("cost, quality, and delivery values must be non-negative")

    capacity_series = _numeric_series(capacities, data.index, "capacities")
    if (capacity_series < 0).any() or capacity_series.sum() < demand_units:
        raise ValueError("capacities must be non-negative and cover total demand")
    return criteria, capacity_series


def _allocation_table(
    supplier_index: pd.Index, allocation: np.ndarray, demand_units: float
) -> pd.DataFrame:
    """Create a user-friendly allocation result table."""
    result = pd.DataFrame(
        {
            "allocation_units": allocation,
            "allocation_share": allocation / demand_units,
        },
        index=supplier_index,
    )
    return result.sort_values("allocation_units", ascending=False)


def preemptive_optimization(
    data: pd.DataFrame,
    demand_units: float,
    capacities: Mapping[str, float] | list[float] | tuple[float, ...],
    quality_target: float,
    delivery_target: float,
    cost_col: str = "Cost",
    quality_col: str = "Quality",
    delivery_col: str = "Delivery",
    tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    """Solve a lexicographic (preemptive) supplier-allocation model.

    Priority order:
        P1: minimize quality shortfall below quality_target.
        P2: minimize delivery excess above delivery_target, preserving P1.
        P3: minimize average cost, preserving P1 and P2.

    The two deviation variables are included in the LP so that an infeasible
    target remains diagnosable instead of causing the entire model to fail.
    """
    criteria, capacity_series = _allocation_setup(
        data, demand_units, capacities, cost_col, quality_col, delivery_col
    )
    if quality_target < 0 or delivery_target < 0:
        raise ValueError("quality_target and delivery_target must be non-negative")

    n = len(criteria)
    cost = criteria[cost_col].to_numpy() / demand_units
    quality = criteria[quality_col].to_numpy() / demand_units
    delivery = criteria[delivery_col].to_numpy() / demand_units

    # Variables are [x_1, ..., x_n, quality_shortfall, delivery_excess].
    bounds = [(0.0, float(capacity)) for capacity in capacity_series]
    bounds += [(0.0, None), (0.0, None)]
    equality = np.zeros((1, n + 2))
    equality[0, :n] = 1.0

    inequality_rows: list[np.ndarray] = []
    inequality_rhs: list[float] = []

    # quality average + shortfall >= target
    row = np.zeros(n + 2)
    row[:n] = -quality
    row[n] = -1.0
    inequality_rows.append(row)
    inequality_rhs.append(-quality_target)

    # delivery average - excess <= target
    row = np.zeros(n + 2)
    row[:n] = delivery
    row[n + 1] = -1.0
    inequality_rows.append(row)
    inequality_rhs.append(delivery_target)

    def solve(objective: np.ndarray, extra_rows: list[np.ndarray] | None = None,
              extra_rhs: list[float] | None = None):
        rows = inequality_rows + (extra_rows or [])
        rhs = inequality_rhs + (extra_rhs or [])
        result = linprog(
            objective,
            A_ub=np.vstack(rows),
            b_ub=np.asarray(rhs, dtype=float),
            A_eq=equality,
            b_eq=np.array([demand_units]),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"preemptive LP failed: {result.message}")
        return result

    stage_1 = solve(np.r_[np.zeros(n), 1.0, 0.0])
    best_quality_shortfall = float(stage_1.x[n])

    preserve_quality = np.zeros(n + 2)
    preserve_quality[n] = 1.0
    stage_2 = solve(
        np.r_[np.zeros(n), 0.0, 1.0],
        [preserve_quality],
        [best_quality_shortfall + tolerance],
    )
    best_delivery_excess = float(stage_2.x[n + 1])

    preserve_delivery = np.zeros(n + 2)
    preserve_delivery[n + 1] = 1.0
    stage_3 = solve(
        np.r_[cost, 0.0, 0.0],
        [preserve_quality, preserve_delivery],
        [best_quality_shortfall + tolerance, best_delivery_excess + tolerance],
    )

    allocation = stage_3.x[:n]
    actual_quality = float(quality @ allocation)
    actual_delivery = float(delivery @ allocation)
    actual_cost = float(cost @ allocation)
    metrics = pd.Series(
        {
            "average_cost": actual_cost,
            "average_quality": actual_quality,
            "average_delivery": actual_delivery,
            "quality_shortfall": max(0.0, quality_target - actual_quality),
            "delivery_excess": max(0.0, actual_delivery - delivery_target),
        }
    )
    stages = {
        "optimal_quality_shortfall": best_quality_shortfall,
        "optimal_delivery_excess_after_P1": best_delivery_excess,
    }
    return _allocation_table(criteria.index, allocation, demand_units), metrics, stages


def goal_programming(
    data: pd.DataFrame,
    demand_units: float,
    capacities: Mapping[str, float] | list[float] | tuple[float, ...],
    quality_target: float,
    delivery_target: float,
    cost_target: float,
    goal_weights: Mapping[str, float],
    cost_col: str = "Cost",
    quality_col: str = "Quality",
    delivery_col: str = "Delivery",
) -> tuple[pd.DataFrame, pd.Series]:
    """Solve weighted goal programming for a supplier allocation.

    Only undesirable deviations are penalized:
        - quality shortfall below quality_target;
        - delivery excess above delivery_target;
        - cost excess above cost_target.

    Deviation terms are divided by their target values, making the goal
    weights comparable even though the criteria use different units.
    """
    criteria, capacity_series = _allocation_setup(
        data, demand_units, capacities, cost_col, quality_col, delivery_col
    )
    if min(quality_target, delivery_target, cost_target) < 0:
        raise ValueError("goal targets must be non-negative")

    goal_names = ["quality", "delivery", "cost"]
    weights = pd.Series(goal_weights, dtype=float).reindex(goal_names)
    if weights.isna().any() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("goal_weights must contain non-negative values with a positive sum")
    weights /= weights.sum()

    n = len(criteria)
    cost = criteria[cost_col].to_numpy() / demand_units
    quality = criteria[quality_col].to_numpy() / demand_units
    delivery = criteria[delivery_col].to_numpy() / demand_units

    # Variables are [x_1, ..., x_n, quality_shortfall,
    #                 delivery_excess, cost_excess].
    bounds = [(0.0, float(capacity)) for capacity in capacity_series]
    bounds += [(0.0, None)] * 3
    equality = np.zeros((1, n + 3))
    equality[0, :n] = 1.0

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    # quality average + shortfall >= target
    row = np.zeros(n + 3)
    row[:n] = -quality
    row[n] = -1.0
    rows.append(row)
    rhs.append(-quality_target)

    # delivery average - excess <= target
    row = np.zeros(n + 3)
    row[:n] = delivery
    row[n + 1] = -1.0
    rows.append(row)
    rhs.append(delivery_target)

    # cost average - excess <= target
    row = np.zeros(n + 3)
    row[:n] = cost
    row[n + 2] = -1.0
    rows.append(row)
    rhs.append(cost_target)

    scale = np.maximum(
        np.array([quality_target, delivery_target, cost_target], dtype=float), 1e-12
    )
    objective = np.r_[
        np.zeros(n),
        weights["quality"] / scale[0],
        weights["delivery"] / scale[1],
        weights["cost"] / scale[2],
    ]

    result = linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs, dtype=float),
        A_eq=equality,
        b_eq=np.array([demand_units]),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"goal-programming LP failed: {result.message}")

    allocation = result.x[:n]
    actual_quality = float(quality @ allocation)
    actual_delivery = float(delivery @ allocation)
    actual_cost = float(cost @ allocation)
    deviations = np.array(
        [
            max(0.0, quality_target - actual_quality),
            max(0.0, actual_delivery - delivery_target),
            max(0.0, actual_cost - cost_target),
        ]
    )
    normalized_objective = float(np.sum(weights.to_numpy() * deviations / scale))

    metrics = pd.Series(
        {
            "average_cost": actual_cost,
            "average_quality": actual_quality,
            "average_delivery": actual_delivery,
            "quality_shortfall": deviations[0],
            "delivery_excess": deviations[1],
            "cost_excess": deviations[2],
            "weighted_normalized_deviation": normalized_objective,
        }
    )
    return _allocation_table(criteria.index, allocation, demand_units), metrics


if __name__ == "__main__":
    # Dummy data: costs and delivery are minimized; quality is maximized.
    criteria_data = pd.DataFrame(
        {
            "Cost": [100.0, 120.0, 90.0],
            "Quality": [80.0, 90.0, 75.0],
            "Delivery": [10.0, 7.0, 12.0],
        },
        index=["Supplier A", "Supplier B", "Supplier C"],
    )
    capacities = {"Supplier A": 50.0, "Supplier B": 80.0, "Supplier C": 70.0}
    demand = 100.0

    weights = {"Cost": 0.40, "Quality": 0.35, "Delivery": 0.25}
    impacts = {"Cost": "cost", "Quality": "benefit", "Delivery": "cost"}

    print("\nTOPSIS")
    print(topsis(criteria_data, weights, impacts))

    print("\nWEIGHTED SUM")
    print(weighted_sum(criteria_data, weights, impacts))

    print("\nPREEMPTIVE OPTIMIZATION")
    allocation, metrics, stages = preemptive_optimization(
        criteria_data,
        demand_units=demand,
        capacities=capacities,
        quality_target=85.0,
        delivery_target=8.0,
    )
    print(allocation)
    print(metrics)
    print("Lexicographic stages:", stages)

    print("\nGOAL PROGRAMMING")
    allocation, metrics = goal_programming(
        criteria_data,
        demand_units=demand,
        capacities=capacities,
        quality_target=85.0,
        delivery_target=8.0,
        cost_target=105.0,
        goal_weights={"quality": 0.45, "delivery": 0.35, "cost": 0.20},
    )
    print(allocation)
    print(metrics)
