from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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