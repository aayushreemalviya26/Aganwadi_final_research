from __future__ import annotations

from typing import Dict, List

import simpy

from config import SimulationConfig
from distributions import RandomSampler
from entities import (
    Child,
    FinalOutcome,
    HealthState,
    ReferralStatus,
    TreatmentLocation,
    TreatmentPhase,
)


class ReferralSystemModel:

    def __init__(self, config: SimulationConfig):

        config.validate()

        self.config = config
        self.env = simpy.Environment()

        self.sampler = RandomSampler(
            config.random_seed
        )

        self.nrc_beds = simpy.Resource(
            self.env,
            capacity=config.nrc_beds
        )

        self.children: List[Child] = []

        self.resource_log: List[Dict[str, float]] = []


    # =========================================================
    # RUN
    # =========================================================

    def run(self) -> None:

        self.env.process(
            self._generate_detected_children()
        )

        self.env.process(
            self._monitor_nrc()
        )

        self.env.run(
            until=self.config.simulation_days
        )

        for child in self.children:

            if child.final_outcome is None:

                child.treatment_phase = (
                    TreatmentPhase.ACTIVE_AT_HORIZON
                )

                if (
                    child.current_health_state
                    == HealthState.MAM
                ):

                    child.final_outcome = (
                        FinalOutcome.ACTIVE_MAM
                    )

                else:

                    child.final_outcome = (
                        FinalOutcome.ACTIVE_SAM
                    )

                child.log(
                    self.config.simulation_days,
                    "simulation_horizon_reached",
                    state=child.current_health_state.value,
                )


    # =========================================================
    # CHILD GENERATION
    # =========================================================

    def _generate_detected_children(self):

        entry_days = sorted(
            self.sampler.uniform(
                0,
                self.config.intake_days
            )
            for _ in range(
                self.config.number_of_children
            )
        )

        previous_day = 0.0

        for child_id, entry_day in enumerate(
            entry_days,
            start=1
        ):

            yield self.env.timeout(
                entry_day - previous_day
            )

            previous_day = entry_day

            state = (
                self.sampler.initial_health_state(
                    self.config
                )
            )

            complication = (
                self.sampler.initial_complication(
                    state,
                    self.config
                )
            )

            child = Child(

                child_id=child_id,

                entry_day=self.env.now,

                initial_health_state=state,

                current_health_state=state,

                initial_medical_complication=complication,

                medical_complication=complication,
            )

            child.log(
                self.env.now,
                "entered_post_detection_model",
                health_state=state.value,
                complication=complication,
            )

            self.children.append(child)

            self.env.process(
                self._child_pathway(child)
            )


    # =========================================================
    # INITIAL ROUTING
    # =========================================================

    def _child_pathway(
        self,
        child: Child
    ):

        if (
            child.current_health_state
            == HealthState.SAM
            and child.medical_complication
        ):

            yield self.env.process(
                self._referral_pathway(child)
            )

        else:

            yield self.env.process(
                self._community_pathway(child)
            )


    # =========================================================
    # COMMUNITY CARE
    # =========================================================

    def _community_pathway(
        self,
        child: Child
    ):

        child.treatment_location = (
            TreatmentLocation.COMMUNITY
        )

        if child.nrc_discharge_day is not None:

            child.treatment_phase = (
                TreatmentPhase.FOLLOW_UP
            )

        else:

            child.treatment_phase = (
                TreatmentPhase.COMMUNITY_TREATMENT
            )


        # -----------------------------------------------------
        # FIRST TREATMENT CONTACT
        # -----------------------------------------------------

        if child.treatment_start_day is None:

            child.treatment_start_day = (
                self.env.now
            )


        # -----------------------------------------------------
        # PROGRAMME CLOCK
        # -----------------------------------------------------

        if (
            child.community_treatment_deadline
            is None
        ):

            planned_duration = (
                self.sampler.integer_inclusive(
                    self.config.community_treatment_days
                )
            )

            child.community_treatment_deadline = (

                child.treatment_start_day

                + self.config.non_responder_threshold_days
            )

            child.log(
                self.env.now,
                "community_rehabilitation_programme_started",
                state=child.current_health_state.value,
                planned_duration=planned_duration,
                non_responder_threshold=(
                    self.config.non_responder_threshold_days
                ),
                deadline=round(
                    child.community_treatment_deadline,
                    3
                ),
            )

        else:

            child.log(
                self.env.now,
                "community_care_resumed",
                state=child.current_health_state.value,
                deadline=round(
                    child.community_treatment_deadline,
                    3
                ),
            )


        # =====================================================
        # FOLLOW-UP LOOP
        # =====================================================

        while (
            child.final_outcome is None
            and self.env.now
            < self.config.simulation_days
        ):


            # -------------------------------------------------
            # HARD NON-RESPONDER THRESHOLD
            # -------------------------------------------------

            if (
                self.env.now
                >= child.community_treatment_deadline
            ):

                self._finish(
                    child,
                    FinalOutcome.NON_RESPONDER
                )

                return


            # -------------------------------------------------
            # FOLLOW-UP INTERVAL
            # -------------------------------------------------

            if (
                child.current_health_state
                == HealthState.MAM
            ):

                interval = (
                    self.config.mam_followup_interval
                )

            else:

                interval = (
                    self.config.sam_followup_interval
                )


            next_followup_day = (
                self.env.now + interval
            )


            # =================================================
            # IMPORTANT FIX
            # =================================================
            #
            # If the next scheduled follow-up would occur
            # AFTER the 120-day programme deadline, do not
            # count that visit as scheduled.
            #
            # Instead, move directly to the deadline and
            # classify the unrecovered child as non-responder.

            if (
                next_followup_day
                > child.community_treatment_deadline
            ):

                wait_to_deadline = (
                    child.community_treatment_deadline
                    - self.env.now
                )

                yield self.env.timeout(
                    wait_to_deadline
                )

                if (
                    self.env.now
                    >= self.config.simulation_days
                ):

                    return

                self._finish(
                    child,
                    FinalOutcome.NON_RESPONDER
                )

                return


            # -------------------------------------------------
            # VALID FOLLOW-UP IS ACTUALLY SCHEDULED
            # -------------------------------------------------

            child.followups_scheduled += 1

            child.next_followup_day = (
                next_followup_day
            )

            yield self.env.timeout(
                interval
            )


            if (
                self.env.now
                >= self.config.simulation_days
            ):

                return


            # =================================================
            # FOLLOW-UP ATTENDANCE
            # =================================================

            attended = (
                self.sampler.bernoulli(
                    self.config.followup_attendance_probability
                )
            )


            if not attended:

                child.consecutive_missed_followups += 1

                child.missed_followup_days += (
                    interval
                )

                child.log(
                    self.env.now,
                    "followup_missed",
                    interval=interval,
                    consecutive_missed=(
                        child.consecutive_missed_followups
                    ),
                )

                if (
                    child.consecutive_missed_followups
                    >= self.config.max_consecutive_missed_followups
                ):

                    self._finish(
                        child,
                        FinalOutcome.DROPOUT
                    )

                    return

                continue


            # -------------------------------------------------
            # FOLLOW-UP ATTENDED
            # -------------------------------------------------

            child.followups_attended += 1

            child.consecutive_missed_followups = 0

            child.log(
                self.env.now,
                "followup_attended",
                state=(
                    child.current_health_state.value
                ),
            )


            # =================================================
            # PERSISTENT COMPLICATION
            # =================================================

            if (
                child.current_health_state
                == HealthState.SAM
                and child.medical_complication
            ):

                child.log(
                    self.env.now,
                    "persistent_complication_re_referral"
                )

                yield self.env.process(
                    self._referral_pathway(child)
                )

                return


            # =================================================
            # MAM TRANSITIONS
            # =================================================

            if (
                child.current_health_state
                == HealthState.MAM
            ):

                draw = (
                    self.sampler.rng.random()
                )

                if (
                    draw
                    < self.config.mam_recovery_probability
                ):

                    child.current_health_state = (
                        HealthState.RECOVERED
                    )

                    self._finish(
                        child,
                        FinalOutcome.RECOVERED
                    )

                    return


                elif (
                    draw
                    < self.config.mam_recovery_probability
                    + self.config.mam_to_sam_probability
                ):

                    child.current_health_state = (
                        HealthState.SAM
                    )

                    child.log(
                        self.env.now,
                        "deteriorated_mam_to_sam"
                    )


            # =================================================
            # SAM TRANSITIONS
            # =================================================

            elif (
                child.current_health_state
                == HealthState.SAM
            ):

                draw = (
                    self.sampler.rng.random()
                )

                if (
                    draw
                    < self.config.sam_to_mam_probability
                ):

                    child.current_health_state = (
                        HealthState.MAM
                    )

                    child.log(
                        self.env.now,
                        "improved_sam_to_mam"
                    )


                elif (
                    draw
                    < self.config.sam_to_mam_probability
                    + self.config.sam_complication_during_care_probability
                ):

                    child.medical_complication = True

                    child.log(
                        self.env.now,
                        "medical_complication_developed"
                    )

                    yield self.env.process(
                        self._referral_pathway(child)
                    )

                    return


            # -------------------------------------------------
            # If an attended visit occurred exactly on deadline
            # and the child still did not recover, classify as
            # non-responder after reassessment.
            # -------------------------------------------------

            if (
                self.env.now
                >= child.community_treatment_deadline
                and child.final_outcome is None
            ):

                self._finish(
                    child,
                    FinalOutcome.NON_RESPONDER
                )

                return


    # =========================================================
    # REFERRAL + NRC
    # =========================================================

    def _referral_pathway(
        self,
        child: Child
    ):

        child.referral_attempts += 1

        child.referral_status = (
            ReferralStatus.PENDING
        )

        child.treatment_phase = (
            TreatmentPhase.REFERRAL_PENDING
        )

        child.treatment_location = (
            TreatmentLocation.NONE
        )


        # -----------------------------------------------------
        # REFERRAL GENERATION
        # -----------------------------------------------------

        referral_delay = (
            self.sampler.integer_inclusive(
                self.config.referral_generation_delay
            )
        )

        child.referral_delay_days += (
            referral_delay
        )

        yield self.env.timeout(
            referral_delay
        )

        if (
            self.env.now
            >= self.config.simulation_days
        ):

            return


        child.referral_generated_day = (
            self.env.now
        )

        child.log(
            self.env.now,
            "referral_generated",
            attempt=child.referral_attempts,
            delay=referral_delay,
        )


        # -----------------------------------------------------
        # FAMILY COMPLIANCE
        # -----------------------------------------------------

        complies = (
            self.sampler.bernoulli(
                self.config.family_referral_compliance
            )
        )

        if not complies:

            child.referral_status = (
                ReferralStatus.DEFAULTED
            )

            child.log(
                self.env.now,
                "referral_not_completed_by_family"
            )

            yield self.env.process(
                self._community_pathway(child)
            )

            return


        # -----------------------------------------------------
        # FACILITY ARRIVAL
        # -----------------------------------------------------

        arrival_delay = (
            self.sampler.integer_inclusive(
                self.config.facility_arrival_delay
            )
        )

        child.arrival_delay_days += (
            arrival_delay
        )

        yield self.env.timeout(
            arrival_delay
        )

        if (
            self.env.now
            >= self.config.simulation_days
        ):

            return


        child.facility_arrival_day = (
            self.env.now
        )

        child.referral_status = (
            ReferralStatus.ARRIVED
        )

        child.treatment_phase = (
            TreatmentPhase.WAITING_FOR_NRC
        )

        child.nrc_queue_enter_day = (
            self.env.now
        )

        child.log(
            self.env.now,
            "arrived_at_nrc",
            arrival_delay=arrival_delay,
        )


        # -----------------------------------------------------
        # NRC BED
        # -----------------------------------------------------

        with self.nrc_beds.request() as request:

            yield request

            queue_wait = (
                self.env.now
                - child.nrc_queue_enter_day
            )

            child.total_queue_wait += (
                queue_wait
            )

            child.nrc_admission_day = (
                self.env.now
            )


            # Direct NRC child starts treatment here
            if (
                child.treatment_start_day
                is None
            ):

                child.treatment_start_day = (
                    self.env.now
                )


            # Start rehabilitation clock once only
            if (
                child.community_treatment_deadline
                is None
            ):

                planned_duration = (
                    self.sampler.integer_inclusive(
                        self.config.community_treatment_days
                    )
                )

                child.community_treatment_deadline = (

                    child.treatment_start_day

                    + self.config.non_responder_threshold_days
                )

                child.log(
                    self.env.now,
                    "rehabilitation_clock_started_at_nrc",
                    planned_duration=planned_duration,
                    non_responder_threshold=(
                        self.config.non_responder_threshold_days
                    ),
                    deadline=round(
                        child.community_treatment_deadline,
                        3
                    ),
                )


            child.treatment_location = (
                TreatmentLocation.NRC
            )

            child.treatment_phase = (
                TreatmentPhase.NRC_TREATMENT
            )

            child.log(
                self.env.now,
                "nrc_admitted",
                queue_wait=round(
                    queue_wait,
                    3
                ),
            )


            # -------------------------------------------------
            # NRC STAY
            # -------------------------------------------------

            stay = (
                self.sampler.integer_inclusive(
                    self.config.nrc_stay_days
                )
            )

            yield self.env.timeout(
                stay
            )

            if (
                self.env.now
                >= self.config.simulation_days
            ):

                return


            child.nrc_discharge_day = (
                self.env.now
            )

            child.referral_status = (
                ReferralStatus.COMPLETED
            )

            child.medical_complication = False

            child.current_health_state = HealthState(
                self.config.post_nrc_state
            )

            child.log(
                self.env.now,
                "nrc_discharged_to_community",
                stay_days=stay,
                post_nrc_state=(
                    child.current_health_state.value
                ),
            )


        # -----------------------------------------------------
        # CONTINUUM OF CARE
        # -----------------------------------------------------

        yield self.env.process(
            self._community_pathway(child)
        )


    # =========================================================
    # FINISH
    # =========================================================

    def _finish(
        self,
        child: Child,
        outcome: FinalOutcome
    ) -> None:

        child.final_outcome = (
            outcome
        )

        child.treatment_end_day = (
            self.env.now
        )

        child.next_followup_day = None


        if (
            outcome
            == FinalOutcome.RECOVERED
        ):

            child.current_health_state = (
                HealthState.RECOVERED
            )

            child.treatment_phase = (
                TreatmentPhase.COMPLETED
            )


        elif (
            outcome
            == FinalOutcome.DROPOUT
        ):

            child.treatment_phase = (
                TreatmentPhase.DROPOUT
            )


        elif (
            outcome
            == FinalOutcome.NON_RESPONDER
        ):

            child.treatment_phase = (
                TreatmentPhase.NON_RESPONDER
            )


        child.log(
            self.env.now,
            "programme_exit",
            outcome=outcome.value,
        )


    # =========================================================
    # NRC MONITOR
    # =========================================================

    def _monitor_nrc(self):

        while True:

            self.resource_log.append({

                "day":
                    float(self.env.now),

                "occupied_beds":
                    len(self.nrc_beds.users),

                "queue_length":
                    len(self.nrc_beds.queue),

                "capacity":
                    self.nrc_beds.capacity,
            })

            yield self.env.timeout(
                self.config.resource_monitor_interval
            )