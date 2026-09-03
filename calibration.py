from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from config import SimulationConfig
from scenarios import run_single


# The calibration matches simulated facility workload to a literature-informed
# occupancy range; it does not imply that the selected number of detected children
# is a national epidemiological estimate.
DEFAULT_LOADS: Tuple[int, ...] = (
    1000,
    2500,
    5000,
    7500,
    10000,
    15000,
    20000,
)
CALIBRATION_BAND: Tuple[float, float] = (0.70, 0.90)
CALIBRATED_CAPACITIES: Tuple[int, ...] = (5, 10, 12, 20)


def _validate_replications(replications: int) -> None:
    if replications <= 0:
        raise ValueError("replications must be positive.")


def run_load_calibration(
    base_config: SimulationConfig,
    loads: Iterable[int] = DEFAULT_LOADS,
    replications: int = 30,
) -> pd.DataFrame:
    """Vary only detected cohort size, using matched seeds at every load."""
    _validate_replications(replications)
    rows: List[dict] = []

    for number_of_children in loads:
        if number_of_children <= 0:
            raise ValueError("Every calibration load must be positive.")

        for replication in range(replications):
            seed = base_config.random_seed + replication
            config = replace(
                base_config,
                number_of_children=int(number_of_children),
                random_seed=seed,
            )
            _, _, _, _, summary = run_single(
                config,
                scenario_name=f"Load {number_of_children}",
            )
            summary["number_of_children"] = int(number_of_children)
            summary["replication"] = replication + 1
            rows.append(summary)

    return pd.DataFrame(rows)


def _summarise_by(
    replication_results: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Build publication-facing summaries while retaining core audit metrics."""
    if replication_results.empty:
        return pd.DataFrame()

    aggregation = {
        "replications": ("replication", "nunique"),
        "mean_nrc_utilisation": ("nrc_utilisation", "mean"),
        "sd_nrc_utilisation": ("nrc_utilisation", "std"),
        "mean_queue_wait_days": ("mean_queue_wait_days", "mean"),
        "mean_peak_queue_length": ("peak_queue_length", "mean"),
        "mean_nrc_occupancy": ("mean_nrc_occupancy", "mean"),
        "mean_peak_nrc_occupancy": ("peak_nrc_occupancy", "mean"),
        "recovery_rate": ("recovery_rate", "mean"),
        "dropout_rate": ("dropout_rate", "mean"),
        "non_responder_rate": ("non_responder_rate", "mean"),
        "referral_attempt_completion_rate": (
            "referral_attempt_completion_rate",
            "mean",
        ),
        "eventual_referral_reach_rate": (
            "eventual_referral_reach_rate",
            "mean",
        ),
        "overall_followup_attendance_rate": (
            "overall_followup_attendance_rate",
            "mean",
        ),
        "mean_referral_delay_per_attempt": (
            "mean_referral_delay_per_attempt",
            "mean",
        ),
        "mean_arrival_delay_per_successful_attempt": (
            "mean_arrival_delay_per_successful_attempt",
            "mean",
        ),
    }
    return (
        replication_results.groupby(group_column, as_index=False)
        .agg(**aggregation)
        .sort_values(group_column)
        .reset_index(drop=True)
    )


def summarise_load_calibration(
    calibration_results: pd.DataFrame,
) -> pd.DataFrame:
    return _summarise_by(calibration_results, "number_of_children")


def select_calibrated_load(
    calibration_summary: pd.DataFrame,
    target_band: Tuple[float, float] = CALIBRATION_BAND,
) -> Optional[int]:
    """Return the lowest tested load with mean utilisation inside the band."""
    low, high = target_band
    if not 0 <= low <= high <= 1:
        raise ValueError("target_band must satisfy 0 <= low <= high <= 1.")
    if calibration_summary.empty:
        return None

    eligible = calibration_summary.loc[
        calibration_summary["mean_nrc_utilisation"].between(
            low,
            high,
            inclusive="both",
        )
    ]
    if eligible.empty:
        return None
    return int(eligible["number_of_children"].min())


def run_calibrated_capacity_analysis(
    base_config: SimulationConfig,
    calibrated_load: int,
    capacities: Iterable[int] = CALIBRATED_CAPACITIES,
    replications: int = 100,
) -> pd.DataFrame:
    """Vary only NRC beds at the selected load, with common random seeds."""
    _validate_replications(replications)
    rows: List[dict] = []

    for nrc_capacity in capacities:
        if nrc_capacity <= 0:
            raise ValueError("Every NRC capacity must be positive.")

        for replication in range(replications):
            seed = base_config.random_seed + replication
            config = replace(
                base_config,
                number_of_children=int(calibrated_load),
                nrc_beds=int(nrc_capacity),
                random_seed=seed,
            )
            _, _, _, _, summary = run_single(
                config,
                scenario_name=f"{nrc_capacity} NRC beds",
            )
            summary["number_of_children"] = int(calibrated_load)
            summary["nrc_capacity"] = int(nrc_capacity)
            summary["replication"] = replication + 1
            rows.append(summary)

    return pd.DataFrame(rows)


def summarise_calibrated_capacity(
    capacity_results: pd.DataFrame,
) -> pd.DataFrame:
    return _summarise_by(capacity_results, "nrc_capacity")
