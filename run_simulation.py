from pathlib import Path
import json

import pandas as pd

from config import BASELINE_CONFIG

from scenarios import (
    run_single,
    run_scenarios,
    summarise_scenarios,
)

from sensitivity import (
    run_sensitivity_analysis,
    summarise_sensitivity,
)

from visualization import (
    plot_nrc_occupancy,
    plot_outcomes,
    plot_scenario_delays,
    plot_scenario_outcomes,
    plot_nrc_scenario_performance,
)


# =========================================================
# ML DATASET BUILDERS
# =========================================================

def build_outcome_ml_dataset(
    children_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    One row = one simulated child.

    Target:
        final_outcome
    """

    feature_columns = [

        "initial_health_state",
        "initial_medical_complication",
        "entry_day",

        "referral_attempts",
        "referral_delay_days",
        "arrival_delay_days",
        "total_queue_wait",

        "followups_scheduled",
        "followups_attended",
        "followup_attendance_rate",
        "missed_followup_days",

        "operational_delay_days",

        "ever_referred",
        "ever_reached_nrc",
        "ever_completed_nrc",

        "nrc_capacity",
        "family_referral_compliance_parameter",
        "followup_attendance_parameter",

        "scenario",
        "replication",
        "random_seed",

        "final_outcome",
    ]

    available_columns = [
        column
        for column in feature_columns
        if column in children_df.columns
    ]

    return children_df[
        available_columns
    ].copy()


def build_referral_ml_dataset(
    children_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Dataset containing only children who experienced
    at least one referral.

    Target:
        ever_reached_nrc
    """

    referred_df = children_df[
        children_df["ever_referred"] == 1
    ].copy()

    feature_columns = [

        "initial_health_state",
        "initial_medical_complication",
        "entry_day",

        "nrc_capacity",
        "family_referral_compliance_parameter",

        "scenario",
        "replication",
        "random_seed",

        "ever_reached_nrc",
    ]

    available_columns = [
        column
        for column in feature_columns
        if column in referred_df.columns
    ]

    return referred_df[
        available_columns
    ].copy()


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)


    # =====================================================
    # 1. BASELINE RUN
    # =====================================================

    print("\nRunning baseline simulation...")

    (
        _,
        baseline_children,
        baseline_resource,
        baseline_events,
        baseline_summary,
    ) = run_single(
        BASELINE_CONFIG,
        scenario_name="Baseline",
    )


    baseline_children.to_csv(
        output_dir / "children_baseline.csv",
        index=False,
    )

    baseline_resource.to_csv(
        output_dir / "nrc_resource_baseline.csv",
        index=False,
    )

    baseline_events.to_csv(
        output_dir / "event_log_baseline.csv",
        index=False,
    )

    with open(
        output_dir / "baseline_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            baseline_summary,
            file,
            indent=2,
        )


    # =====================================================
    # 2. SCENARIO ANALYSIS
    # =====================================================

    print("\nRunning scenario analysis...")

    (
        scenario_results,
        all_children,
        all_events,
    ) = run_scenarios(
        BASELINE_CONFIG,
        replications=100,
    )


    scenario_results.to_csv(
        output_dir / "scenario_replications.csv",
        index=False,
    )

    all_children.to_csv(
        output_dir / "children_all_scenarios.csv",
        index=False,
    )

    all_events.to_csv(
        output_dir / "event_log_all_scenarios.csv",
        index=False,
    )


    # =====================================================
    # 3. SCENARIO SUMMARY
    # =====================================================

    scenario_summary = summarise_scenarios(
        scenario_results
    )

    scenario_summary.to_csv(
        output_dir / "scenario_summary.csv",
        index=False,
    )


    # =====================================================
    # 4. ONE-WAY SENSITIVITY ANALYSIS
    # =====================================================

    print("\nRunning one-way sensitivity analysis...")

    sensitivity_results = run_sensitivity_analysis(
        BASELINE_CONFIG,
        replications=100,
    )

    sensitivity_results.to_csv(
        output_dir / "sensitivity_replications.csv",
        index=False,
    )


    sensitivity_summary = summarise_sensitivity(
        sensitivity_results
    )

    sensitivity_summary.to_csv(
        output_dir / "sensitivity_summary.csv",
        index=False,
    )

    print(
        "\nSensitivity analysis completed."
    )


    # =====================================================
    # 5. ML DATASETS
    # =====================================================

    print("\nGenerating ML datasets...")

    outcome_ml_df = build_outcome_ml_dataset(
        all_children
    )

    outcome_ml_df.to_csv(
        output_dir / "ml_dataset_outcome.csv",
        index=False,
    )


    referral_ml_df = build_referral_ml_dataset(
        all_children
    )

    referral_ml_df.to_csv(
        output_dir / "ml_dataset_referral.csv",
        index=False,
    )


    # =====================================================
    # 6. VISUALISATIONS
    # =====================================================

    print("\nGenerating plots...")

    plot_nrc_occupancy(
        baseline_resource,
        output_dir / "nrc_occupancy.png",
    )

    plot_outcomes(
        baseline_children,
        output_dir / "programme_outcomes.png",
    )

    plot_scenario_delays(
        scenario_results,
        output_dir / "scenario_delays.png",
    )

    plot_scenario_outcomes(
        scenario_results,
        output_dir / "scenario_outcomes.png",
    )

    plot_nrc_scenario_performance(
        scenario_results,
        output_dir / "nrc_capacity_effect.png",
    )


    # =====================================================
    # 7. BASELINE SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("BASELINE SIMULATION SUMMARY")
    print("=" * 60)

    for key, value in baseline_summary.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )


    # =====================================================
    # 8. SCENARIO SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("SCENARIO MEANS")
    print("=" * 60)

    display_columns = [

        "scenario",

        "mean_queue_wait_days_mean",

        "mean_operational_delay_days_mean",

        "recovery_rate_mean",

        "dropout_rate_mean",

        "non_responder_rate_mean",

        "nrc_utilisation_mean",
    ]

    available_display_columns = [

        column
        for column in display_columns

        if column
        in scenario_summary.columns
    ]


    print(
        scenario_summary[
            available_display_columns
        ].to_string(index=False)
    )


    # =====================================================
    # 9. SENSITIVITY SUMMARY
    # =====================================================

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    print(
        sensitivity_summary.to_string(
            index=False
        )
    )


    # =====================================================
    # 10. DATASET INFORMATION
    # =====================================================

    print("\n" + "=" * 60)
    print("GENERATED DATASETS")
    print("=" * 60)


    print(
        f"Baseline children: "
        f"{len(baseline_children)}"
    )


    print(
        f"Scenario child records: "
        f"{len(all_children)}"
    )


    print(
        f"Event records: "
        f"{len(all_events)}"
    )


    print(
        f"ML outcome dataset rows: "
        f"{len(outcome_ml_df)}"
    )


    print(
        f"ML referral dataset rows: "
        f"{len(referral_ml_df)}"
    )


    print(
        f"Sensitivity replications: "
        f"{len(sensitivity_results)}"
    )


    print(
        "\nOutputs saved in:",
        output_dir.resolve(),
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()