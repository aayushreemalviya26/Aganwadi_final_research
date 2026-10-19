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


def _add_calibration_metrics(
    summary: dict,
    resource_df: pd.DataFrame,
    event_df: pd.DataFrame,
    config: SimulationConfig,
) -> None:
    """Add window-specific workload metrics without changing model outputs."""
    active_resource = resource_df.loc[
        resource_df["day"] < config.intake_days
    ]
    occupied_bed_days = float(active_resource["occupied_beds"].sum())

    summary["active_window_nrc_utilisation"] = (
        occupied_bed_days / (config.nrc_beds * config.intake_days)
    )
    summary["full_horizon_nrc_utilisation"] = summary["nrc_utilisation"]
    summary["active_window_mean_nrc_occupancy"] = (
        occupied_bed_days / config.intake_days
    )
    summary["full_horizon_mean_nrc_occupancy"] = summary[
        "mean_nrc_occupancy"
    ]

    saturated = active_resource["occupied_beds"] >= config.nrc_beds
    summary["active_window_saturation_rate"] = float(saturated.mean())
    longest_saturated_run = 0
    current_run = 0
    for is_saturated in saturated:
        current_run = current_run + 1 if is_saturated else 0
        longest_saturated_run = max(longest_saturated_run, current_run)
    summary["longest_saturated_run_days"] = int(longest_saturated_run)

    if event_df.empty:
        queue_waits = pd.Series(dtype=float)
        arrivals = 0
    else:
        queue_waits = event_df.loc[
            event_df["event"] == "nrc_admitted",
            "queue_wait",
        ].dropna()
        arrivals = int((event_df["event"] == "arrived_at_nrc").sum())

    admissions = len(queue_waits)
    unresolved_arrivals = max(arrivals - admissions, 0)
    positive_waits = int((queue_waits > 0).sum()) + unresolved_arrivals
    summary["mean_queue_wait_days"] = (
        float(queue_waits.mean()) if admissions else 0.0
    )
    summary["median_queue_wait_days"] = (
        float(queue_waits.median()) if admissions else 0.0
    )
    summary["p95_queue_wait_days"] = (
        float(queue_waits.quantile(0.95)) if admissions else 0.0
    )
    summary["proportion_nrc_arrivals_waited_gt_0"] = (
        float(positive_waits / arrivals) if arrivals else 0.0
    )
    summary["nrc_arrivals"] = arrivals
    summary["observed_nrc_admissions"] = admissions
    summary["queue_wait_censored_arrivals"] = unresolved_arrivals


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
            _, _, resource_df, event_df, summary = run_single(
                config,
                scenario_name=f"Load {number_of_children}",
            )
            _add_calibration_metrics(summary, resource_df, event_df, config)
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
        "mean_active_window_nrc_utilisation": (
            "active_window_nrc_utilisation",
            "mean",
        ),
        "sd_active_window_nrc_utilisation": (
            "active_window_nrc_utilisation",
            "std",
        ),
        "mean_full_horizon_nrc_utilisation": (
            "full_horizon_nrc_utilisation",
            "mean",
        ),
        "sd_full_horizon_nrc_utilisation": (
            "full_horizon_nrc_utilisation",
            "std",
        ),
        # Retained as compatibility aliases for the original whole-horizon metric.
        "mean_nrc_utilisation": ("nrc_utilisation", "mean"),
        "sd_nrc_utilisation": ("nrc_utilisation", "std"),
        "mean_queue_wait_days": ("mean_queue_wait_days", "mean"),
        "median_queue_wait_days": ("median_queue_wait_days", "mean"),
        "p95_queue_wait_days": ("p95_queue_wait_days", "mean"),
        "proportion_nrc_arrivals_waited_gt_0": (
            "proportion_nrc_arrivals_waited_gt_0",
            "mean",
        ),
        "mean_peak_queue_length": ("peak_queue_length", "mean"),
        "mean_nrc_occupancy": ("active_window_mean_nrc_occupancy", "mean"),
        "full_horizon_mean_nrc_occupancy": (
            "full_horizon_mean_nrc_occupancy",
            "mean",
        ),
        "mean_peak_nrc_occupancy": ("peak_nrc_occupancy", "mean"),
        "mean_active_window_saturation_rate": (
            "active_window_saturation_rate",
            "mean",
        ),
        "mean_longest_saturated_run_days": (
            "longest_saturated_run_days",
            "mean",
        ),
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
    summary = _summarise_by(calibration_results, "number_of_children")
    if summary.empty:
        return summary

    low, high = CALIBRATION_BAND
    summary["within_calibration_band"] = summary[
        "mean_active_window_nrc_utilisation"
    ].between(low, high, inclusive="both")
    eligible_loads = summary.loc[
        summary["within_calibration_band"],
        "number_of_children",
    ]
    selected = int(eligible_loads.min()) if not eligible_loads.empty else None
    summary["selected_as_calibrated_load"] = (
        summary["number_of_children"] == selected
        if selected is not None
        else False
    )
    return summary


def select_calibrated_load(
    calibration_summary: pd.DataFrame,
    target_band: Tuple[float, float] = CALIBRATION_BAND,
) -> Optional[int]:
    """Return the lowest load with active-window utilisation inside the band."""
    low, high = target_band
    if not 0 <= low <= high <= 1:
        raise ValueError("target_band must satisfy 0 <= low <= high <= 1.")
    if calibration_summary.empty:
        return None

    eligible = calibration_summary.loc[
        calibration_summary["mean_active_window_nrc_utilisation"].between(
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
            _, _, resource_df, event_df, summary = run_single(
                config,
                scenario_name=f"{nrc_capacity} NRC beds",
            )
            _add_calibration_metrics(summary, resource_df, event_df, config)
            summary["number_of_children"] = int(calibrated_load)
            summary["nrc_capacity"] = int(nrc_capacity)
            summary["replication"] = replication + 1
            rows.append(summary)

    return pd.DataFrame(rows)


def summarise_calibrated_capacity(
    capacity_results: pd.DataFrame,
) -> pd.DataFrame:
    summary = _summarise_by(capacity_results, "nrc_capacity")
    if summary.empty:
        return summary

    loads = capacity_results["number_of_children"].unique()
    if len(loads) != 1:
        raise ValueError("Capacity analysis must use one fixed calibrated load.")
    summary.insert(0, "number_of_children", int(loads[0]))
    capacity = summary["nrc_capacity"]
    excessive_wait = summary["mean_queue_wait_days"] > 14
    excessive_queue = summary["mean_peak_queue_length"] > (2 * capacity)
    prolonged_saturation = (
        summary["mean_longest_saturated_run_days"] >= (0.90 * 365)
    )
    summary["sanity_check_flag"] = (
        excessive_wait | excessive_queue | prolonged_saturation
    )
    summary["sanity_check_reasons"] = [
        "; ".join(
            reason
            for condition, reason in (
                (wait, "mean wait exceeds 14-day maximum NRC stay"),
                (queue, "mean peak queue exceeds twice bed capacity"),
                (saturation, "mean longest saturation run covers >=90% of intake window"),
            )
            if condition
        )
        or "none"
        for wait, queue, saturation in zip(
            excessive_wait,
            excessive_queue,
            prolonged_saturation,
        )
    ]
    return summary


def summarise_calibrated_outcomes(
    capacity_results: pd.DataFrame,
    nrc_capacity: int = 12,
) -> pd.DataFrame:
    """Summarise final outcome percentages and 95% CIs across replications."""
    selected = capacity_results.loc[
        capacity_results["nrc_capacity"] == nrc_capacity
    ]
    if selected.empty:
        raise ValueError(f"No results found for NRC capacity {nrc_capacity}.")

    outcomes = (
        ("Recovered", "recovery_rate"),
        ("Dropout", "dropout_rate"),
        ("Non-responder", "non_responder_rate"),
        ("Active MAM at horizon", "active_mam_rate"),
        ("Active SAM at horizon", "active_sam_rate"),
    )
    rows = []
    for label, column in outcomes:
        percentages = selected[column] * 100
        mean = float(percentages.mean())
        standard_error = float(percentages.sem())
        margin = 1.96 * standard_error
        rows.append(
            {
                "outcome": label,
                "replications": len(percentages),
                "mean_percentage": mean,
                "sd_percentage": float(percentages.std()),
                "ci_95_lower_percentage": max(0.0, mean - margin),
                "ci_95_upper_percentage": min(100.0, mean + margin),
            }
        )
    return pd.DataFrame(rows)
