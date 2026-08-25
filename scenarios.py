from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import pandas as pd

from config import SimulationConfig
from metrics import build_output_tables, summarise_run
from model import ReferralSystemModel
from validation import validate_outputs


# =========================================================
# DEFAULT OPERATIONAL SCENARIOS
# =========================================================

# Important:
# Only one major parameter is changed at a time so that
# its effect can be interpreted independently.

DEFAULT_SCENARIOS: Dict[str, Dict[str, object]] = {

    # Uses BASELINE_CONFIG exactly as defined in config.py
    "Baseline": {},

    # NRC capacity sensitivity
    "Low NRC capacity": {
        "nrc_beds": 5,
    },

    "High NRC capacity": {
        "nrc_beds": 20,
    },
}


# =========================================================
# SINGLE SIMULATION RUN
# =========================================================

def run_single(
    config: SimulationConfig,
    scenario_name: str = "Baseline",
):
    """
    Run one stochastic replication of the DES.

    Returns:
        model
        children_df
        resource_df
        event_df
        summary
    """

    model = ReferralSystemModel(config)

    model.run()

    children_df, resource_df, event_df = (
        build_output_tables(model)
    )

    summary = summarise_run(
        children_df,
        resource_df,
        event_df,
    )

    summary["scenario"] = scenario_name
    summary["random_seed"] = config.random_seed

    # -----------------------------------------------------
    # Validate output before accepting the run
    # -----------------------------------------------------

    errors = validate_outputs(
        children_df,
        resource_df,
        config,
    )

    if errors:

        raise RuntimeError(
            "Validation failed:\n- "
            + "\n- ".join(errors)
        )

    return (
        model,
        children_df,
        resource_df,
        event_df,
        summary,
    )


# =========================================================
# MULTIPLE SCENARIOS + REPLICATIONS
# =========================================================

def run_scenarios(
    base_config: SimulationConfig,
    scenarios: Dict[str, Dict[str, object]] = DEFAULT_SCENARIOS,
    replications: int = 100,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run multiple operational scenarios with repeated
    stochastic replications.

    The same replication seed is used across scenarios.
    This creates common random numbers and makes scenario
    comparisons less affected by unrelated random variation.

    Returns:
        summary_df
        all_children_df
        all_events_df
    """

    summary_rows: List[dict] = []

    child_tables: List[pd.DataFrame] = []

    event_tables: List[pd.DataFrame] = []


    for scenario_name, overrides in scenarios.items():

        for replication in range(replications):

            # -------------------------------------------------
            # COMMON RANDOM NUMBERS
            # -------------------------------------------------

            # Replication 1 uses the same seed across all
            # scenarios, replication 2 another same seed, etc.
            #
            # Example:
            # Baseline rep 1 -> seed 42
            # Low capacity rep 1 -> seed 42
            # High capacity rep 1 -> seed 42
            #
            # This makes scenario comparisons cleaner.

            replication_seed = (
                base_config.random_seed
                + replication
            )

            config = replace(
                base_config,
                random_seed=replication_seed,
                **overrides,
            )


            (
                _,
                children_df,
                _,
                event_df,
                summary,
            ) = run_single(
                config,
                scenario_name,
            )


            # -------------------------------------------------
            # ADD TRACEABILITY FIELDS
            # -------------------------------------------------

            replication_id = replication + 1

            summary["replication"] = replication_id

            summary_rows.append(summary)


            # Child-level data for ML / detailed analysis
            children_df = children_df.copy()

            children_df["scenario"] = scenario_name

            children_df["replication"] = replication_id

            children_df["random_seed"] = replication_seed

            children_df["nrc_capacity"] = config.nrc_beds

            children_df[
                "family_referral_compliance_parameter"
            ] = config.family_referral_compliance

            children_df[
                "followup_attendance_parameter"
            ] = config.followup_attendance_probability

            child_tables.append(
                children_df
            )


            # Event-level data
            event_df = event_df.copy()

            if not event_df.empty:

                event_df["scenario"] = scenario_name

                event_df["replication"] = (
                    replication_id
                )

                event_df["random_seed"] = (
                    replication_seed
                )

                event_tables.append(
                    event_df
                )


    # =====================================================
    # COMBINE OUTPUTS
    # =====================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    all_children_df = pd.concat(
        child_tables,
        ignore_index=True,
    )


    if event_tables:

        all_events_df = pd.concat(
            event_tables,
            ignore_index=True,
        )

    else:

        all_events_df = pd.DataFrame()


    return (
        summary_df,
        all_children_df,
        all_events_df,
    )


# =========================================================
# SCENARIO SUMMARY
# =========================================================

def summarise_scenarios(
    scenario_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and standard deviation across replications.
    """

    metrics = [
        "referral_reach_rate",
        "recovery_rate",
        "dropout_rate",
        "non_responder_rate",
        "mean_referral_delay_days",
        "mean_arrival_delay_days",
        "mean_queue_wait_days",
        "mean_operational_delay_days",
        "mean_followup_attendance_rate",
        "mean_programme_duration_days",
        "nrc_utilisation",
        "peak_queue_length",
    ]


    available_metrics = [
        metric
        for metric in metrics
        if metric in scenario_results.columns
    ]


    mean_df = (
        scenario_results
        .groupby("scenario")[available_metrics]
        .mean()
        .add_suffix("_mean")
    )


    std_df = (
        scenario_results
        .groupby("scenario")[available_metrics]
        .std()
        .add_suffix("_sd")
    )


    summary_df = (
        mean_df
        .join(std_df)
        .reset_index()
    )


    return summary_df