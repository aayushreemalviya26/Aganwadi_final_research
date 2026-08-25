from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any


class HealthState(str, Enum):
    MAM = "MAM"
    SAM = "SAM"
    RECOVERED = "RECOVERED"


class TreatmentLocation(str, Enum):
    NONE = "NONE"
    COMMUNITY = "COMMUNITY"
    NRC = "NRC"


class TreatmentPhase(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    COMMUNITY_TREATMENT = "COMMUNITY_TREATMENT"
    REFERRAL_PENDING = "REFERRAL_PENDING"
    WAITING_FOR_NRC = "WAITING_FOR_NRC"
    NRC_TREATMENT = "NRC_TREATMENT"
    FOLLOW_UP = "FOLLOW_UP"
    COMPLETED = "COMPLETED"
    DROPOUT = "DROPOUT"
    NON_RESPONDER = "NON_RESPONDER"
    ACTIVE_AT_HORIZON = "ACTIVE_AT_HORIZON"


class ReferralStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    PENDING = "PENDING"
    DEFAULTED = "DEFAULTED"
    ARRIVED = "ARRIVED"
    COMPLETED = "COMPLETED"


class FinalOutcome(str, Enum):
    RECOVERED = "RECOVERED"
    DROPOUT = "DROPOUT"
    NON_RESPONDER = "NON_RESPONDER"
    ACTIVE_MAM = "ACTIVE_MAM"
    ACTIVE_SAM = "ACTIVE_SAM"


@dataclass
class Child:

    # -----------------------------
    # Basic child information
    # -----------------------------

    child_id: int
    entry_day: float

    initial_health_state: HealthState
    current_health_state: HealthState

    initial_medical_complication: bool
    medical_complication: bool


    # -----------------------------
    # Current system status
    # -----------------------------

    treatment_location: TreatmentLocation = TreatmentLocation.NONE
    treatment_phase: TreatmentPhase = TreatmentPhase.NOT_STARTED

    referral_status: ReferralStatus = ReferralStatus.NOT_REQUIRED

    final_outcome: Optional[FinalOutcome] = None


    # -----------------------------
    # Treatment timing
    # -----------------------------

    treatment_start_day: Optional[float] = None
    treatment_end_day: Optional[float] = None

    community_treatment_deadline: Optional[float] = None

    next_followup_day: Optional[float] = None


    # -----------------------------
    # Referral / NRC timestamps
    # -----------------------------

    referral_generated_day: Optional[float] = None
    facility_arrival_day: Optional[float] = None

    nrc_queue_enter_day: Optional[float] = None
    nrc_admission_day: Optional[float] = None
    nrc_discharge_day: Optional[float] = None


    # -----------------------------
    # Operational delays
    # -----------------------------

    referral_delay_days: float = 0.0
    arrival_delay_days: float = 0.0
    total_queue_wait: float = 0.0

    # Kept separate because missing a follow-up
    # does not necessarily mean receiving no treatment.
    missed_followup_days: float = 0.0


    # -----------------------------
    # Counters
    # -----------------------------

    referral_attempts: int = 0

    followups_scheduled: int = 0
    followups_attended: int = 0

    consecutive_missed_followups: int = 0


    # -----------------------------
    # Complete child event history
    # -----------------------------

    event_history: List[Dict[str, Any]] = field(default_factory=list)


    def log(
        self,
        day: float,
        event: str,
        **details: Any
    ) -> None:

        self.event_history.append({
            "day": round(float(day), 3),
            "event": event,
            **details,
        })


    def to_record(self) -> Dict[str, Any]:

        record = asdict(self)

        # Convert enums into plain strings for CSV output
        for field_name in [
            "initial_health_state",
            "current_health_state",
            "treatment_location",
            "treatment_phase",
            "referral_status",
            "final_outcome",
        ]:

            value = getattr(self, field_name)

            record[field_name] = (
                value.value
                if value is not None
                else None
            )

        # Useful summary variable
        record["event_count"] = len(self.event_history)

        # Event history will be exported separately
        record.pop("event_history", None)

        return record