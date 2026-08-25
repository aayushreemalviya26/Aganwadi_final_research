from dataclasses import replace

from config import BASELINE_CONFIG
from scenarios import run_single


tests = {
    "Baseline": {},

    "Huge NRC capacity": {
        "nrc_beds": 1000,
    },

    "Single NRC bed": {
        "nrc_beds": 1,
    },

    "Zero referral compliance": {
        "family_referral_compliance": 0.0,
    },

    "Full referral compliance": {
        "family_referral_compliance": 1.0,
    },
}


for name, overrides in tests.items():

    config = replace(
        BASELINE_CONFIG,
        **overrides
    )

    (
        _,
        children_df,
        resource_df,
        event_df,
        summary,
    ) = run_single(
        config,
        scenario_name=name
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("NRC capacity:", config.nrc_beds)
    print("Referral compliance:", config.family_referral_compliance)

    print("Referred children:",
          summary["referred_children"])

    print("Reached NRC:",
          summary["reached_nrc"])

    print("Referral reach rate:",
          summary["referral_reach_rate"])

    print("Mean queue wait:",
          summary["mean_queue_wait_days"])

    print("Peak queue:",
          summary["peak_queue_length"])

    print("NRC utilisation:",
          summary["nrc_utilisation"])

    print("Total referral attempts:",
          summary["total_referral_attempts"])

    print("Successful referral attempts:",
          summary["successful_referral_attempts"])

    print("Failed referral attempts:",
          summary["failed_referral_attempts"])

    print("Attempt completion rate:",
          summary["referral_attempt_completion_rate"])

    print("Overall follow-up attendance:",
          summary["overall_followup_attendance_rate"])

    print("Mean referral delay per attempt:",
          summary["mean_referral_delay_per_attempt"])