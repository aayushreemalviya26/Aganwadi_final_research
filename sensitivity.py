from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import pandas as pd

from config import SimulationConfig
from scenarios import run_single


# ============================================================
# ONE-WAY SENSITIVITY ANALYSIS
# ============================================================
#
# Only ONE parameter is changed at a time.
# All other parameters remain at their baseline values.
#
# Clinical transition probabilities without empirical
# low/high bounds use ±25% around baseline and must be
# reported as assumption-based uncertainty analysis.
# ============================================================


SENSITIVITY_VALUES: Dict[str, List[float]] = {

    # Operational/resource parameter
    "nrc_beds": [
        5,
        12,
        20,
    ],

    # Referral pathway
    "family_referral_compliance": [
        0.60,
        0.61,
        0.90,
    ],

    # Continuity of community care
    "followup_attendance_probability": [
        0.60,
        0.80,
        0.90,
    ],

    # Community SAM complication pressure
    "sam_complication_during_care_probability": [
        0.02,
        0.04,
        0.21,
    ],

    # Assumption-based ±25% around baseline 0.12
    "mam_recovery_probability": [
        0.09,
        0.12,
        0.15,
    ],

    # Assumption-based ±25% around baseline 0.10
    "sam_to_mam_probability": [
        0.075,
        0.10,
        0.125,
    ],
}


# ============================================================
# RUN ONE-WAY SENSITIVITY ANALYSIS
# ============================================================

def run_sensitivity_analysis(
    base_config: SimulationConfig,
    replications: int = 100,
) -> pd.DataFrame:

    rows = []

    for parameter, values in SENSITIVITY_VALUES.items():

        for value in values:

            for replication in range(replications):

                config = replace(
                    base_config,
                    random_seed=(
                        base_config.random_seed
                        + replication
                    ),
                    **{
                        parameter: value
                    },
                )

                scenario_name = (
                    f"{parameter}={value}"
                )

                (
                    _,
                    _,
                    _,
                    _,
                    summary,
                ) = run_single(
                    config,
                    scenario_name=scenario_name,
                )

                summary["sensitivity_parameter"] = (
                    parameter
                )

                summary["sensitivity_value"] = (
                    value
                )

                summary["replication"] = (
                    replication + 1
                )

                rows.append(summary)

    return pd.DataFrame(rows)


# ============================================================
# SUMMARISE SENSITIVITY RESULTS
# ============================================================

def summarise_sensitivity(
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [

        "recovery_rate",
        "dropout_rate",
        "non_responder_rate",

        "eventual_referral_reach_rate",
        "referral_attempt_completion_rate",

        "mean_queue_wait_days",
        "mean_operational_delay_days",

        "overall_followup_attendance_rate",

        "nrc_utilisation",
        "peak_queue_length",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in sensitivity_df.columns
    ]

    aggregation = {}

    for metric in available_metrics:

        aggregation[
            f"{metric}_mean"
        ] = (
            metric,
            "mean",
        )

        aggregation[
            f"{metric}_sd"
        ] = (
            metric,
            "std",
        )

    summary_df = (
        sensitivity_df
        .groupby(
            [
                "sensitivity_parameter",
                "sensitivity_value",
            ],
            as_index=False,
        )
        .agg(**aggregation)
    )

    return summary_df