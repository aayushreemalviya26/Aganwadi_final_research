from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_load_vs_nrc_utilisation(
    calibration_summary: pd.DataFrame,
    output_path: Path,
    target_low: float = 0.70,
    target_high: float = 0.90,
) -> None:
    """Plot mean utilisation, replication SD, and the calibration band."""
    if calibration_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = calibration_summary["number_of_children"]
    y = calibration_summary["mean_nrc_utilisation"] * 100
    yerr = calibration_summary["sd_nrc_utilisation"].fillna(0) * 100
    ax.axhspan(
        target_low * 100,
        target_high * 100,
        color="#d8ead2",
        alpha=0.65,
        label="Literature-informed calibration band (70–90%)",
    )
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4, color="#24557a")
    ax.set_title("Detected Child Load and NRC Utilisation")
    ax.set_xlabel("Detected children per year")
    ax.set_ylabel("Mean NRC utilisation (%)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_load_vs_queue_wait(
    calibration_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if calibration_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        calibration_summary["number_of_children"],
        calibration_summary["mean_queue_wait_days"],
        marker="o",
        color="#9a3e2e",
    )
    ax.set_title("Detected Child Load and NRC Queue Waiting Time")
    ax.set_xlabel("Detected children per year")
    ax.set_ylabel("Mean NRC queue wait (days)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_calibrated_capacity_effect(
    capacity_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if capacity_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        capacity_summary["nrc_capacity"],
        capacity_summary["mean_queue_wait_days"],
        marker="o",
        color="#9a3e2e",
    )
    ax.set_title("NRC Capacity Effect at Calibrated Workload")
    ax.set_xlabel("NRC capacity (beds)")
    ax.set_ylabel("Mean NRC queue wait (days)")
    ax.set_xticks(capacity_summary["nrc_capacity"])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_calibrated_capacity_utilisation(
    capacity_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if capacity_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        capacity_summary["nrc_capacity"],
        capacity_summary["mean_nrc_utilisation"] * 100,
        marker="o",
        color="#24557a",
    )
    ax.set_title("NRC Capacity and Utilisation at Calibrated Workload")
    ax.set_xlabel("NRC capacity (beds)")
    ax.set_ylabel("Mean NRC utilisation (%)")
    ax.set_xticks(capacity_summary["nrc_capacity"])
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_nrc_occupancy(
    resource_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot NRC bed occupancy and queue length over simulation time.
    """

    if resource_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        resource_df["day"],
        resource_df["occupied_beds"],
        label="Occupied beds",
    )

    ax.plot(
        resource_df["day"],
        resource_df["queue_length"],
        label="Queue length",
    )

    ax.axhline(
        resource_df["capacity"].iloc[0],
        linestyle="--",
        label="NRC capacity",
    )

    ax.set_title("NRC Occupancy and Queue Over Time")
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Number of children")

    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_outcomes(
    children_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot final programme outcomes for one simulation run.
    """

    if children_df.empty:
        return

    outcome_order = [
        "RECOVERED",
        "DROPOUT",
        "NON_RESPONDER",
        "ACTIVE_MAM",
        "ACTIVE_SAM",
    ]

    counts = (
        children_df["final_outcome"]
        .value_counts()
        .reindex(outcome_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    counts.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title("Final Programme Outcomes")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Number of children")

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scenario_delays(
    scenario_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare operational delays across scenarios using
    replication-level mean results.
    """

    if scenario_results.empty:
        return

    grouped = (
        scenario_results
        .groupby("scenario")
        .agg(
            mean_referral_delay=(
                "mean_referral_delay_days",
                "mean",
            ),
            mean_arrival_delay=(
                "mean_arrival_delay_days",
                "mean",
            ),
            mean_queue_wait=(
                "mean_queue_wait_days",
                "mean",
            ),
        )
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    grouped.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Operational Delays Across Scenarios"
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean delay (days)")

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.legend(
        [
            "Referral generation",
            "Facility arrival",
            "NRC queue",
        ]
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scenario_outcomes(
    scenario_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare mean programme outcome rates across
    simulation scenarios.
    """

    if scenario_results.empty:
        return

    grouped = (
        scenario_results
        .groupby("scenario")
        .agg(
            recovery_rate=(
                "recovery_rate",
                "mean",
            ),
            dropout_rate=(
                "dropout_rate",
                "mean",
            ),
            non_responder_rate=(
                "non_responder_rate",
                "mean",
            ),
        )
    )

    # Convert proportions to percentages
    grouped = grouped * 100

    fig, ax = plt.subplots(figsize=(10, 5))

    grouped.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Programme Outcomes Across Scenarios"
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean outcome rate (%)")

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    ax.legend(
        [
            "Recovered",
            "Dropout",
            "Non-responder",
        ]
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_nrc_scenario_performance(
    scenario_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Compare NRC queue waiting time across capacity scenarios.
    This is particularly useful for bottleneck analysis.
    """

    if scenario_results.empty:
        return

    grouped = (
        scenario_results
        .groupby("scenario")
        .agg(
            mean_queue_wait=(
                "mean_queue_wait_days",
                "mean",
            )
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    grouped["mean_queue_wait"].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Effect of NRC Capacity on Queue Waiting Time"
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mean NRC queue wait (days)")

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
