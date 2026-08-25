from __future__ import annotations

from typing import List
import pandas as pd

from config import SimulationConfig


def validate_outputs(
    children_df: pd.DataFrame,
    resource_df: pd.DataFrame,
    config: SimulationConfig,
) -> List[str]:
    """
    Check simulation outputs for logical and structural errors.

    Returns:
        List of validation errors.
        Empty list means no validation problem was detected.
    """

    errors: List[str] = []

    # =========================================================
    # POPULATION CHECKS
    # =========================================================

    if len(children_df) != config.number_of_children:
        errors.append(
            f"Expected {config.number_of_children} children, "
            f"found {len(children_df)}."
        )

    if children_df["child_id"].duplicated().any():
        errors.append("Duplicate child IDs detected.")

    if children_df["final_outcome"].isna().any():
        errors.append("At least one child has no final outcome.")


    # =========================================================
    # TIME / DELAY CHECKS
    # =========================================================

    delay_columns = [
        "referral_delay_days",
        "arrival_delay_days",
        "total_queue_wait",
        "missed_followup_days",
    ]

    for column in delay_columns:
        if (children_df[column] < 0).any():
            errors.append(
                f"Negative value detected in {column}."
            )


    # Treatment cannot end before it starts
    valid_treatment_times = (
        children_df["treatment_start_day"].notna()
        & children_df["treatment_end_day"].notna()
    )

    if (
        children_df.loc[
            valid_treatment_times,
            "treatment_end_day"
        ]
        <
        children_df.loc[
            valid_treatment_times,
            "treatment_start_day"
        ]
    ).any():

        errors.append(
            "Treatment end day occurs before treatment start day."
        )


    # Facility arrival cannot occur before referral generation
    valid_referral_times = (
        children_df["referral_generated_day"].notna()
        & children_df["facility_arrival_day"].notna()
    )

    if (
        children_df.loc[
            valid_referral_times,
            "facility_arrival_day"
        ]
        <
        children_df.loc[
            valid_referral_times,
            "referral_generated_day"
        ]
    ).any():

        errors.append(
            "Facility arrival occurs before referral generation."
        )


    # NRC admission cannot occur before facility arrival
    valid_nrc_times = (
        children_df["facility_arrival_day"].notna()
        & children_df["nrc_admission_day"].notna()
    )

    if (
        children_df.loc[
            valid_nrc_times,
            "nrc_admission_day"
        ]
        <
        children_df.loc[
            valid_nrc_times,
            "facility_arrival_day"
        ]
    ).any():

        errors.append(
            "NRC admission occurs before facility arrival."
        )


    # NRC discharge cannot occur before NRC admission
    valid_nrc_stays = (
        children_df["nrc_admission_day"].notna()
        & children_df["nrc_discharge_day"].notna()
    )

    if (
        children_df.loc[
            valid_nrc_stays,
            "nrc_discharge_day"
        ]
        <
        children_df.loc[
            valid_nrc_stays,
            "nrc_admission_day"
        ]
    ).any():

        errors.append(
            "NRC discharge occurs before NRC admission."
        )


    # =========================================================
    # FOLLOW-UP CHECKS
    # =========================================================

    if (
        children_df["followups_attended"]
        >
        children_df["followups_scheduled"]
    ).any():

        errors.append(
            "Follow-ups attended exceed follow-ups scheduled."
        )

    if (children_df["followups_scheduled"] < 0).any():
        errors.append(
            "Negative scheduled follow-up count detected."
        )

    if (children_df["followups_attended"] < 0).any():
        errors.append(
            "Negative attended follow-up count detected."
        )


    # =========================================================
    # CLINICAL OUTCOME CHECKS
    # =========================================================

    invalid_recovery = (
        (children_df["final_outcome"] == "RECOVERED")
        &
        (children_df["current_health_state"] != "RECOVERED")
    )

    if invalid_recovery.any():
        errors.append(
            "Recovered outcome is inconsistent "
            "with current health state."
        )


    # =========================================================
    # REFERRAL / NRC CHECKS
    # =========================================================

    nrc_without_referral = (
        children_df["nrc_admission_day"].notna()
        &
        (children_df["referral_attempts"] == 0)
    )

    if nrc_without_referral.any():
        errors.append(
            "A child entered NRC without a referral attempt."
        )


    discharge_without_admission = (
        children_df["nrc_discharge_day"].notna()
        &
        children_df["nrc_admission_day"].isna()
    )

    if discharge_without_admission.any():
        errors.append(
            "NRC discharge recorded without NRC admission."
        )


    # =========================================================
    # NRC RESOURCE CHECKS
    # =========================================================

    if not resource_df.empty:

        if (
            resource_df["occupied_beds"]
            >
            resource_df["capacity"]
        ).any():

            errors.append(
                "NRC occupancy exceeded configured capacity."
            )

        if (resource_df["occupied_beds"] < 0).any():
            errors.append(
                "Negative NRC occupancy detected."
            )

        if (resource_df["queue_length"] < 0).any():
            errors.append(
                "Negative NRC queue length detected."
            )


    return errors