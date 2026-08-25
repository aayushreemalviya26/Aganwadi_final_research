from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SimulationConfig:
    """
    Central configuration for the post-detection MAM/SAM
    referral-system Discrete Event Simulation.
    """

    # =========================================================
    # RUN CONTROL
    # =========================================================

    random_seed: int = 42

    # Detected children enter continuously over one programme year
    intake_days: int = 365

    # Extra follow-up time for children entering late in the year
    simulation_days: int = 545

    # Simulation cohort size
    number_of_children: int = 1000


    # =========================================================
    # DETECTED POPULATION MIX
    # =========================================================

    # Bokaro community screening:
    # 217 SAM among 1499 MAM + SAM children ≈ 14.5%
    sam_share: float = 0.145


    # =========================================================
    # INITIAL CLINICAL ROUTING
    # =========================================================

    # Approx. 10–15% of SAM children require inpatient care
    sam_complication_probability: float = 0.15

    # No reliable numeric estimate identified for MAM
    # complication prevalence at detection.
    mam_complication_probability: float = 0.00


    # =========================================================
    # OPERATIONAL SCENARIO PARAMETERS
    # =========================================================

    # Uncertain operational delays.
    # These will be varied through sensitivity/scenario analysis.
    referral_generation_delay: Tuple[int, int] = (1, 7)
    facility_arrival_delay: Tuple[int, int] = (1, 7)

    # No direct empirical estimate found for initial referral completion.
    # Treated as an operational scenario parameter.
    family_referral_compliance: float = 0.61

    # Baseline scenario value.
    # Literature reports programme-level follow-up completion,
    # not one universal per-visit probability.
    followup_attendance_probability: float = 0.80


    # =========================================================
    # NRC RESOURCE AND CARE
    # =========================================================

    nrc_beds: int = 12

    nrc_stay_days: Tuple[int, int] = (10, 14)


    # =========================================================
    # COMMUNITY CARE
    # =========================================================

    mam_followup_interval: int = 15
    sam_followup_interval: int = 7

    # General Indian CMAM framework: 12–16 weeks
    community_treatment_days: Tuple[int, int] = (84, 112)

    # National outpatient non-responder threshold
    non_responder_threshold_days: int = 120


    # =========================================================
    # CLINICAL TRANSITIONS
    # =========================================================

    # Derived modelling parameter:
    # approximately reproduces ~55% cumulative MAM recovery
    # over six fortnightly visits.
    mam_recovery_probability: float = 0.12

    # No reliable longitudinal MAM -> SAM transition coefficient
    # was identified. Excluded from baseline.
    mam_to_sam_probability: float = 0.00

    # Modelling assumption.
    # Must be tested in sensitivity analysis.
    sam_to_mam_probability: float = 0.10

    # Modelling assumption.
    # Published 21% is cumulative emergency referral,
    # not a per-follow-up probability.
    sam_complication_during_care_probability: float = 0.04


    # =========================================================
    # PROGRAMME EXIT / CONTINUUM OF CARE
    # =========================================================

    max_consecutive_missed_followups: int = 2

    # NRC discharge is stabilisation, not final recovery.
    # Child returns to community rehabilitation.
    post_nrc_state: str = "MAM"


    # =========================================================
    # RESOURCE MONITORING
    # =========================================================

    resource_monitor_interval: int = 1


    # =========================================================
    # VALIDATION
    # =========================================================

    def validate(self) -> None:

        probability_fields = {
            "sam_share": self.sam_share,
            "sam_complication_probability":
                self.sam_complication_probability,
            "mam_complication_probability":
                self.mam_complication_probability,
            "family_referral_compliance":
                self.family_referral_compliance,
            "followup_attendance_probability":
                self.followup_attendance_probability,
            "mam_recovery_probability":
                self.mam_recovery_probability,
            "mam_to_sam_probability":
                self.mam_to_sam_probability,
            "sam_to_mam_probability":
                self.sam_to_mam_probability,
            "sam_complication_during_care_probability":
                self.sam_complication_during_care_probability,
        }

        for name, value in probability_fields.items():
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )


        # Mutually exclusive outcomes at one MAM follow-up
        if (
            self.mam_recovery_probability
            + self.mam_to_sam_probability
            > 1
        ):
            raise ValueError(
                "MAM transition probabilities cannot exceed 1."
            )


        # Mutually exclusive outcomes at one SAM follow-up
        if (
            self.sam_to_mam_probability
            + self.sam_complication_during_care_probability
            > 1
        ):
            raise ValueError(
                "SAM transition probabilities cannot exceed 1."
            )


        if self.number_of_children <= 0:
            raise ValueError(
                "number_of_children must be positive."
            )

        if self.nrc_beds <= 0:
            raise ValueError(
                "nrc_beds must be positive."
            )

        if self.intake_days <= 0:
            raise ValueError(
                "intake_days must be positive."
            )

        if self.simulation_days <= self.intake_days:
            raise ValueError(
                "simulation_days must exceed intake_days."
            )

        if self.non_responder_threshold_days <= 0:
            raise ValueError(
                "non_responder_threshold_days must be positive."
            )

        if self.max_consecutive_missed_followups <= 0:
            raise ValueError(
                "max_consecutive_missed_followups must be positive."
            )

        if self.resource_monitor_interval <= 0:
            raise ValueError(
                "resource_monitor_interval must be positive."
            )

        if self.post_nrc_state not in {"MAM", "SAM"}:
            raise ValueError(
                "post_nrc_state must be 'MAM' or 'SAM'."
            )


        range_fields = {
            "referral_generation_delay":
                self.referral_generation_delay,

            "facility_arrival_delay":
                self.facility_arrival_delay,

            "nrc_stay_days":
                self.nrc_stay_days,

            "community_treatment_days":
                self.community_treatment_days,
        }

        for name, bounds in range_fields.items():

            low, high = bounds

            if low < 0:
                raise ValueError(
                    f"{name} cannot contain negative values."
                )

            if low > high:
                raise ValueError(
                    f"Invalid range for {name}: {bounds}"
                )


BASELINE_CONFIG = SimulationConfig()