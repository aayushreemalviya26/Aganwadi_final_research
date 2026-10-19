import argparse
import json
from pathlib import Path

import pandas as pd

from config import BASELINE_CONFIG

from calibration import (
    CALIBRATION_BAND,
    run_calibrated_capacity_analysis,
    run_load_calibration,
    select_calibrated_load,
    summarise_calibrated_capacity,
    summarise_calibrated_outcomes,
    summarise_load_calibration,
)

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
    plot_calibrated_capacity_effect,
    plot_calibrated_capacity_utilisation,
    plot_calibrated_programme_outcomes,
    plot_load_vs_nrc_utilisation,
    plot_load_vs_queue_wait,
    plot_nrc_occupancy,
    plot_outcomes,
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run workload calibration and calibrated NRC capacity analysis."
    )
    parser.add_argument(
        "--search-replications",
        type=int,
        default=30,
        help="Replications per candidate load (default: 30).",
    )
    parser.add_argument(
        "--final-replications",
        type=int,
        default=100,
        help="Replications per calibrated capacity (default: 100).",
    )
    parser.add_argument(
        "--include-existing-analyses",
        action="store_true",
        help=(
            "Also regenerate the existing scenario, sensitivity, and ML CSVs; "
            "their legacy plots are intentionally not regenerated."
        ),
    )
    return parser.parse_args()


def run_existing_analyses(output_dir: Path) -> None:
    """Retain the verified scenario, sensitivity, and ML export workflow."""
    print("\nRunning existing scenario analysis and ML exports...")
    scenario_results, all_children, all_events = run_scenarios(
        BASELINE_CONFIG,
        replications=100,
    )
    scenario_results.to_csv(
        output_dir / "scenario_replications.csv", index=False
    )
    all_children.to_csv(
        output_dir / "children_all_scenarios.csv", index=False
    )
    all_events.to_csv(
        output_dir / "event_log_all_scenarios.csv", index=False
    )
    summarise_scenarios(scenario_results).to_csv(
        output_dir / "scenario_summary.csv", index=False
    )

    print("\nRunning existing one-way sensitivity analysis...")
    sensitivity_results = run_sensitivity_analysis(
        BASELINE_CONFIG,
        replications=100,
    )
    sensitivity_results.to_csv(
        output_dir / "sensitivity_replications.csv", index=False
    )
    summarise_sensitivity(sensitivity_results).to_csv(
        output_dir / "sensitivity_summary.csv", index=False
    )

    build_outcome_ml_dataset(all_children).to_csv(
        output_dir / "ml_dataset_outcome.csv", index=False
    )
    build_referral_ml_dataset(all_children).to_csv(
        output_dir / "ml_dataset_referral.csv", index=False
    )


def main() -> None:

    args = parse_args()

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)


    # =====================================================
    # 1. BASELINE RUN
    # =====================================================

    print("\nRunning low-demand / original baseline...")

    (
        _,
        baseline_children,
        baseline_resource,
        baseline_events,
        baseline_summary,
    ) = run_single(
        BASELINE_CONFIG,
        scenario_name="Low-demand / original baseline",
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
    # 2. ORIGINAL BASELINE VISUALISATIONS
    # =====================================================

    plot_nrc_occupancy(
        baseline_resource,
        output_dir / "nrc_occupancy.png",
    )

    plot_outcomes(
        baseline_children,
        output_dir / "programme_outcomes.png",
    )

    # =====================================================
    # 3. DEMAND LOAD CALIBRATION
    # =====================================================

    print(
        "\nRunning demand load calibration "
        f"({args.search_replications} replications per load)..."
    )
    calibration_results = run_load_calibration(
        BASELINE_CONFIG,
        replications=args.search_replications,
    )
    calibration_summary = summarise_load_calibration(calibration_results)
    calibration_results.to_csv(
        output_dir / "load_calibration_replications.csv",
        index=False,
        float_format="%.6f",
    )
    calibration_summary.to_csv(
        output_dir / "load_calibration_summary.csv",
        index=False,
        float_format="%.6f",
    )
    plot_load_vs_nrc_utilisation(
        calibration_summary,
        output_dir / "load_vs_nrc_utilisation.png",
        *CALIBRATION_BAND,
    )
    plot_load_vs_queue_wait(
        calibration_summary,
        output_dir / "load_vs_queue_wait.png",
    )

    calibrated_load = select_calibrated_load(
        calibration_summary,
        CALIBRATION_BAND,
    )
    original_row = calibration_summary.loc[
        calibration_summary["number_of_children"]
        == BASELINE_CONFIG.number_of_children
    ].iloc[0]

    if calibrated_load is None:
        print("\n" + "=" * 60)
        print("DEMAND LOAD CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"Original baseline load: {BASELINE_CONFIG.number_of_children}")
        print(
            "Original baseline active-window NRC utilisation: "
            f"{original_row['mean_active_window_nrc_utilisation']:.1%}"
        )
        print(
            "Original baseline full-horizon NRC utilisation: "
            f"{original_row['mean_full_horizon_nrc_utilisation']:.1%}"
        )
        print("Calibrated load selected: none")
        print("70-90% target band reached: No")
        print("Capacity analysis was not run because no tested load met the rule.")
        return

    # =====================================================
    # 4. CAPACITY ANALYSIS AT THE CALIBRATED WORKLOAD
    # =====================================================

    print(
        "\nRunning calibrated capacity analysis "
        f"({args.final_replications} replications per capacity)..."
    )
    capacity_results = run_calibrated_capacity_analysis(
        BASELINE_CONFIG,
        calibrated_load,
        replications=args.final_replications,
    )
    capacity_summary = summarise_calibrated_capacity(capacity_results)
    capacity_results.to_csv(
        output_dir / "calibrated_capacity_replications.csv",
        index=False,
        float_format="%.6f",
    )
    capacity_summary.to_csv(
        output_dir / "calibrated_capacity_summary.csv",
        index=False,
        float_format="%.6f",
    )
    outcome_summary = summarise_calibrated_outcomes(
        capacity_results,
        nrc_capacity=BASELINE_CONFIG.nrc_beds,
    )
    outcome_summary.to_csv(
        output_dir / "calibrated_programme_outcomes_summary.csv",
        index=False,
        float_format="%.6f",
    )
    plot_calibrated_capacity_effect(
        capacity_summary,
        output_dir / "calibrated_capacity_effect.png",
    )
    plot_calibrated_capacity_utilisation(
        capacity_summary,
        output_dir / "calibrated_capacity_utilisation.png",
    )
    plot_calibrated_programme_outcomes(
        outcome_summary,
        output_dir / "calibrated_programme_outcomes.png",
        calibrated_load,
    )

    if args.include_existing_analyses:
        run_existing_analyses(output_dir)

    calibrated_row = capacity_summary.loc[
        capacity_summary["nrc_capacity"] == BASELINE_CONFIG.nrc_beds
    ].iloc[0]
    search_row = calibration_summary.loc[
        calibration_summary["number_of_children"] == calibrated_load
    ].iloc[0]

    print("\n" + "=" * 60)
    print("DEMAND LOAD CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Original baseline load: {BASELINE_CONFIG.number_of_children}")
    print(
        "Original baseline active-window NRC utilisation: "
        f"{original_row['mean_active_window_nrc_utilisation']:.1%}"
    )
    print(
        "Original baseline full-horizon NRC utilisation: "
        f"{original_row['mean_full_horizon_nrc_utilisation']:.1%}"
    )
    print(f"Calibrated load selected: {calibrated_load}")
    print(
        "Calibrated active-window NRC utilisation (12 beds): "
        f"{calibrated_row['mean_active_window_nrc_utilisation']:.1%}"
    )
    print(
        "Calibrated full-horizon NRC utilisation (12 beds): "
        f"{calibrated_row['mean_full_horizon_nrc_utilisation']:.1%}"
    )
    print(
        "Calibrated mean NRC queue wait (12 beds): "
        f"{calibrated_row['mean_queue_wait_days']:.3f} days"
    )
    print(
        "Calibrated median NRC queue wait (12 beds): "
        f"{calibrated_row['median_queue_wait_days']:.3f} days"
    )
    print(
        "Calibrated 95th-percentile NRC queue wait (12 beds): "
        f"{calibrated_row['p95_queue_wait_days']:.3f} days"
    )
    print(
        "Calibrated mean peak queue (12 beds): "
        f"{calibrated_row['mean_peak_queue_length']:.2f} children"
    )
    print(
        "Proportion of NRC arrivals waiting >0 days (12 beds): "
        f"{calibrated_row['proportion_nrc_arrivals_waited_gt_0']:.1%}"
    )
    print("70-90% target band reached: Yes")
    print(
        "Selection-stage active-window utilisation: "
        f"{search_row['mean_active_window_nrc_utilisation']:.1%}"
    )
    print(
        "Internal plausibility sanity check: "
        + (
            "FLAGGED - " + calibrated_row["sanity_check_reasons"]
            if calibrated_row["sanity_check_flag"]
            else "No automatic warning flags"
        )
    )
    print(
        "\nInterpretation: the calibration matches simulated facility workload "
        "to a literature-informed occupancy range; it does not imply that the "
        "selected number of detected children is a national epidemiological estimate."
    )
    print("\nOutputs saved in:", output_dir.resolve())


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
