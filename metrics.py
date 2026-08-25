from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from model import ReferralSystemModel


# =========================================================
# BUILD OUTPUT TABLES
# =========================================================

def build_output_tables(
    model: ReferralSystemModel,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert simulation objects into analysis-ready tables.

    Returns:
        children_df -> one row per child
        resource_df -> NRC occupancy/queue over time
        event_df    -> one row per child event
    """

    # -----------------------------------------------------
    # CHILD-LEVEL TABLE
    # -----------------------------------------------------

    children_df = pd.DataFrame(
        child.to_record()
        for child in model.children
    )

    if not children_df.empty:

        # Total operational delay accumulated by each child
        children_df["operational_delay_days"] = (
            children_df["referral_delay_days"]
            + children_df["arrival_delay_days"]
            + children_df["total_queue_wait"]
        )

        # Child-level follow-up attendance proportion
        children_df["followup_attendance_rate"] = np.where(
            children_df["followups_scheduled"] > 0,
            children_df["followups_attended"]
            / children_df["followups_scheduled"],
            np.nan,
        )

        # Programme duration only for children with an exit
        children_df["programme_duration_days"] = np.where(
            children_df["treatment_start_day"].notna()
            & children_df["treatment_end_day"].notna(),
            children_df["treatment_end_day"]
            - children_df["treatment_start_day"],
            np.nan,
        )

        # Ever-event indicators
        children_df["ever_referred"] = (
            children_df["referral_attempts"] > 0
        ).astype(int)

        children_df["ever_reached_nrc"] = (
            children_df["nrc_admission_day"].notna()
        ).astype(int)

        children_df["ever_completed_nrc"] = (
            children_df["nrc_discharge_day"].notna()
        ).astype(int)


    # -----------------------------------------------------
    # NRC RESOURCE TABLE
    # -----------------------------------------------------

    resource_df = pd.DataFrame(
        model.resource_log
    )


    # -----------------------------------------------------
    # EVENT-LEVEL TABLE
    # -----------------------------------------------------

    event_rows = []

    for child in model.children:

        for event in child.event_history:

            event_rows.append(
                {
                    "child_id": child.child_id,
                    **event,
                }
            )

    event_df = pd.DataFrame(
        event_rows
    )

    return (
        children_df,
        resource_df,
        event_df,
    )


# =========================================================
# SAFE RATE
# =========================================================

def safe_rate(
    numerator: int,
    denominator: int,
) -> float:

    return (
        float(numerator / denominator)
        if denominator
        else 0.0
    )


# =========================================================
# SUMMARY
# =========================================================

def summarise_run(
    children_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> Dict[str, float]:

    total = len(children_df)

    if total == 0:
        return {}


    # =====================================================
    # OUTCOMES
    # =====================================================

    outcomes = (
        children_df["final_outcome"]
        .value_counts()
    )


    # =====================================================
    # CHILD-LEVEL REFERRAL METRICS
    # =====================================================

    referred_children = int(
        children_df["ever_referred"].sum()
    )

    reached_nrc = int(
        children_df["ever_reached_nrc"].sum()
    )

    completed_nrc = int(
        children_df["ever_completed_nrc"].sum()
    )


    # =====================================================
    # ATTEMPT-LEVEL REFERRAL METRICS
    # =====================================================

    total_referral_attempts = int(
        children_df["referral_attempts"].sum()
    )


    if not event_df.empty:

        # Each failed family decision generates this event
        failed_referral_attempts = int(
            (
                event_df["event"]
                == "referral_not_completed_by_family"
            ).sum()
        )

        # Every facility arrival corresponds to a successful
        # referral-compliance decision.
        successful_referral_attempts = int(
            (
                event_df["event"]
                == "arrived_at_nrc"
            ).sum()
        )

        # Unique children who ever failed at least one referral
        referral_default_children = int(
            event_df.loc[
                event_df["event"]
                == "referral_not_completed_by_family",
                "child_id",
            ].nunique()
        )

    else:

        failed_referral_attempts = 0
        successful_referral_attempts = 0
        referral_default_children = 0


    # -----------------------------------------------------
    # Attempts still unresolved at simulation horizon
    # -----------------------------------------------------

    unresolved_referral_attempts = max(
        total_referral_attempts
        - successful_referral_attempts
        - failed_referral_attempts,
        0,
    )


    # Compliance should be evaluated only among attempts
    # where the family compliance decision was observed.
    resolved_referral_attempts = (
        successful_referral_attempts
        + failed_referral_attempts
    )

    referral_attempt_completion_rate = safe_rate(
        successful_referral_attempts,
        resolved_referral_attempts,
    )


    # Eventual child-level reach:
    # may approach 1.0 because failed referrals can be retried.
    eventual_referral_reach_rate = safe_rate(
        reached_nrc,
        referred_children,
    )


    # =====================================================
    # REFERRAL DELAY METRICS
    # =====================================================

    referred_df = children_df[
        children_df["ever_referred"] == 1
    ]


    if not referred_df.empty:

        # Cumulative delay experienced by each referred child.
        mean_cumulative_referral_delay = float(
            referred_df["referral_delay_days"].mean()
        )

        mean_cumulative_arrival_delay = float(
            referred_df["arrival_delay_days"].mean()
        )

        mean_cumulative_operational_delay = float(
            referred_df["operational_delay_days"].mean()
        )

    else:

        mean_cumulative_referral_delay = 0.0
        mean_cumulative_arrival_delay = 0.0
        mean_cumulative_operational_delay = 0.0


    # -----------------------------------------------------
    # Delay per referral attempt from event log
    # -----------------------------------------------------

    if (
        not event_df.empty
        and "delay" in event_df.columns
    ):

        referral_generation_events = event_df[
            event_df["event"] == "referral_generated"
        ]

        mean_referral_delay_per_attempt = (
            float(
                referral_generation_events["delay"]
                .dropna()
                .mean()
            )
            if not referral_generation_events.empty
            else 0.0
        )

    else:

        mean_referral_delay_per_attempt = 0.0


    # -----------------------------------------------------
    # Arrival delay per successful referral
    # -----------------------------------------------------

    if (
        not event_df.empty
        and "arrival_delay" in event_df.columns
    ):

        arrival_events = event_df[
            event_df["event"] == "arrived_at_nrc"
        ]

        mean_arrival_delay_per_successful_attempt = (
            float(
                arrival_events["arrival_delay"]
                .dropna()
                .mean()
            )
            if not arrival_events.empty
            else 0.0
        )

    else:

        mean_arrival_delay_per_successful_attempt = 0.0


    # =====================================================
    # FOLLOW-UP METRICS
    # =====================================================

    children_with_followups = children_df[
        children_df["followups_scheduled"] > 0
    ]


    if not children_with_followups.empty:

        # Mean of child-specific attendance proportions
        mean_followup_attendance = float(
            children_with_followups[
                "followup_attendance_rate"
            ].mean()
        )

    else:

        mean_followup_attendance = 0.0


    # -----------------------------------------------------
    # Overall visit-level attendance
    # -----------------------------------------------------

    total_followups_scheduled = int(
        children_df["followups_scheduled"].sum()
    )

    total_followups_attended = int(
        children_df["followups_attended"].sum()
    )

    overall_followup_attendance_rate = safe_rate(
        total_followups_attended,
        total_followups_scheduled,
    )


    # =====================================================
    # NRC RESOURCE METRICS
    # =====================================================

    if not resource_df.empty:

        occupancy = (
            resource_df["occupied_beds"]
        )

        capacity = float(
            resource_df["capacity"].iloc[0]
        )

        utilisation = (
            float(
                occupancy.mean() / capacity
            )
            if capacity > 0
            else 0.0
        )

        mean_occupancy = float(
            occupancy.mean()
        )

        peak_occupancy = int(
            occupancy.max()
        )

        peak_queue = int(
            resource_df["queue_length"].max()
        )

    else:

        utilisation = 0.0
        mean_occupancy = 0.0
        peak_occupancy = 0
        peak_queue = 0


    # =====================================================
    # QUEUE METRICS
    # =====================================================

    nrc_children = children_df[
        children_df["ever_reached_nrc"] == 1
    ]


    if not nrc_children.empty:

        mean_queue_wait = float(
            nrc_children[
                "total_queue_wait"
            ].mean()
        )

        max_queue_wait = float(
            nrc_children[
                "total_queue_wait"
            ].max()
        )

    else:

        mean_queue_wait = 0.0
        max_queue_wait = 0.0


    # =====================================================
    # PROGRAMME DURATION
    # =====================================================

    completed_duration = (
        children_df[
            "programme_duration_days"
        ]
        .dropna()
    )

    mean_programme_duration = (
        float(
            completed_duration.mean()
        )
        if not completed_duration.empty
        else 0.0
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    return {

        # -------------------------------------------------
        # Population
        # -------------------------------------------------

        "children":
            total,

        "initial_mam":
            int(
                (
                    children_df[
                        "initial_health_state"
                    ] == "MAM"
                ).sum()
            ),

        "initial_sam":
            int(
                (
                    children_df[
                        "initial_health_state"
                    ] == "SAM"
                ).sum()
            ),


        # -------------------------------------------------
        # Child-level referral pathway
        # -------------------------------------------------

        "referred_children":
            referred_children,

        "reached_nrc":
            reached_nrc,

        "completed_nrc_referrals":
            completed_nrc,

        "referral_default_children":
            referral_default_children,

        "eventual_referral_reach_rate":
            eventual_referral_reach_rate,

        # Kept for compatibility with existing code.
        "referral_reach_rate":
            eventual_referral_reach_rate,

        "nrc_completion_rate":
            safe_rate(
                completed_nrc,
                reached_nrc,
            ),


        # -------------------------------------------------
        # Attempt-level referral pathway
        # -------------------------------------------------

        "total_referral_attempts":
            total_referral_attempts,

        "successful_referral_attempts":
            successful_referral_attempts,

        "failed_referral_attempts":
            failed_referral_attempts,

        "unresolved_referral_attempts":
            unresolved_referral_attempts,

        "referral_attempt_completion_rate":
            referral_attempt_completion_rate,


        # -------------------------------------------------
        # Programme outcomes
        # -------------------------------------------------

        "recovery_rate":
            safe_rate(
                int(
                    outcomes.get(
                        "RECOVERED",
                        0,
                    )
                ),
                total,
            ),

        "dropout_rate":
            safe_rate(
                int(
                    outcomes.get(
                        "DROPOUT",
                        0,
                    )
                ),
                total,
            ),

        "non_responder_rate":
            safe_rate(
                int(
                    outcomes.get(
                        "NON_RESPONDER",
                        0,
                    )
                ),
                total,
            ),

        "active_mam_rate":
            safe_rate(
                int(
                    outcomes.get(
                        "ACTIVE_MAM",
                        0,
                    )
                ),
                total,
            ),

        "active_sam_rate":
            safe_rate(
                int(
                    outcomes.get(
                        "ACTIVE_SAM",
                        0,
                    )
                ),
                total,
            ),


        # -------------------------------------------------
        # Operational delays
        # -------------------------------------------------

        # Per-attempt metrics
        "mean_referral_delay_per_attempt":
            mean_referral_delay_per_attempt,

        "mean_arrival_delay_per_successful_attempt":
            mean_arrival_delay_per_successful_attempt,

        # Cumulative child-level metrics
        "mean_cumulative_referral_delay_days":
            mean_cumulative_referral_delay,

        "mean_cumulative_arrival_delay_days":
            mean_cumulative_arrival_delay,

        "mean_cumulative_operational_delay_days":
            mean_cumulative_operational_delay,

        # Old names retained so visualization/scenario
        # code does not break.
        "mean_referral_delay_days":
            mean_cumulative_referral_delay,

        "mean_arrival_delay_days":
            mean_cumulative_arrival_delay,

        "mean_operational_delay_days":
            mean_cumulative_operational_delay,

        "mean_queue_wait_days":
            mean_queue_wait,

        "max_queue_wait_days":
            max_queue_wait,

        "mean_missed_followup_days":
            float(
                children_df[
                    "missed_followup_days"
                ].mean()
            ),


        # -------------------------------------------------
        # Follow-up
        # -------------------------------------------------

        "total_followups_scheduled":
            total_followups_scheduled,

        "total_followups_attended":
            total_followups_attended,

        "overall_followup_attendance_rate":
            overall_followup_attendance_rate,

        "mean_followup_attendance_rate":
            mean_followup_attendance,


        # -------------------------------------------------
        # Programme duration
        # -------------------------------------------------

        "mean_programme_duration_days":
            mean_programme_duration,


        # -------------------------------------------------
        # NRC
        # -------------------------------------------------

        "mean_nrc_occupancy":
            mean_occupancy,

        "peak_nrc_occupancy":
            peak_occupancy,

        "peak_queue_length":
            peak_queue,

        "nrc_utilisation":
            utilisation,
    }