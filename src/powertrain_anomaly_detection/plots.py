import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


TIME_COL = "time_s"
LABEL_COL = "anomaly"

# Default signals to visualize
DEFAULT_SIGNALS: List[Tuple[str, str]] = [
    ("VehicleSpeed_kph",          "Vehicle Speed (kph)"),
    ("EngineRPM",                 "Engine Speed (rpm)"),
    ("DriverTorqueRequest_Nm",    "Driver Torque Request (Nm)"),
    ("EngineTorque_Actual_Nm",    "Engine Torque Actual (Nm)"),
    ("MAP_kPa",                   "MAP (kPa)"),
    ("Lambda",                    "Lambda"),
]


def pick_example_drives_by_cycle(df_all: pd.DataFrame) -> Dict[str, int]:
    """
    For each cycle_name, pick one representative drive_id.
    Preference order:
        1) test_anom
        2) test_normal
        3) any drive in that cycle
    """
    example_ids: Dict[str, int] = {}

    for cycle_name, g in df_all.groupby("cycle_name"):
        cand = g[g["split"] == "test_anom"]["drive_id"].unique()
        if len(cand) == 0:
            cand = g[g["split"] == "test_normal"]["drive_id"].unique()
        if len(cand) == 0:
            cand = g["drive_id"].unique()

        if len(cand) > 0:
            example_ids[cycle_name] = int(cand[0])

    return example_ids


def plot_drive_signals_with_anomalies(
    df_drive: pd.DataFrame,
    signals: List[Tuple[str, str]] | None = None,
    title_prefix: str = "",
    figsize_scale: float = 2.5,
):
    """
    Plot a single drive with multiple signals and shaded anomaly regions.

    signals: list of (column_name, pretty_label)
    """
    if signals is None:
        signals = DEFAULT_SIGNALS

    time = df_drive[TIME_COL].values
    labels = df_drive[LABEL_COL].values
    split_vals = df_drive["split"].unique()
    cycle_name = df_drive["cycle_name"].iloc[0]
    drive_id = df_drive["drive_id"].iloc[0]

    n_signals = len(signals)
    n_rows = n_signals + 1  # +1 for anomaly timeline

    plt.figure(figsize=(14, figsize_scale * n_rows))
    plt.suptitle(
        f"{title_prefix}Cycle: {cycle_name} | drive_id={drive_id} | "
        f"split={', '.join(split_vals)}",
        fontsize=14,
        y=0.99,
    )

    # Plot each signal with shaded anomaly regions
    for row_idx, (col, nice_name) in enumerate(signals, start=1):
        sig = df_drive[col].values

        ax = plt.subplot(n_rows, 1, row_idx)
        ax.plot(time, sig)
        ax.set_ylabel(nice_name)
        if row_idx == 1:
            ax.set_title("Signals with anomaly regions")
        ax.grid(True, alpha=0.3)

        # Shade anomaly regions
        in_anom = False
        start = None
        for i in range(len(time)):
            if labels[i] == 1 and not in_anom:
                in_anom = True
                start = time[i]
            if labels[i] == 0 and in_anom:
                in_anom = False
                ax.axvspan(start, time[i], alpha=0.2)
        if in_anom:
            ax.axvspan(start, time[-1], alpha=0.2)

    # Bottom row: anomaly label timeline
    ax = plt.subplot(n_rows, 1, n_rows)
    ax.step(time, labels, where="post")
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("anomaly (0/1)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# plots.py
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_example_drives_by_cycle(df_all, save_dir=None, show=True):
    TIME_COL   = "time_s"
    LABEL_COL  = "anomaly"

    SIGNALS = [
        ("EngineRPM",              "Engine Speed (rpm)"),
        ("EngineTorque_Actual_Nm", "Engine Torque (Nm)"),
    ]

    # Pick one drive per cycle
    example_ids = {}
    for cycle_name, g in df_all.groupby("cycle_name"):
        cand = g[g["split"] == "test_anom"]["drive_id"].unique()
        if len(cand) == 0:
            cand = g[g["split"] == "test_normal"]["drive_id"].unique()
        if len(cand) == 0:
            cand = g["drive_id"].unique()
        if len(cand) > 0:
            example_ids[cycle_name] = int(cand[0])

    print("Example drive_ids by cycle:")
    for c, d in example_ids.items():
        print(f"  {c}: drive_id={d}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for cycle_name, drive_id in example_ids.items():
        df_drive = df_all[df_all["drive_id"] == drive_id].copy()
        time   = df_drive[TIME_COL].values
        labels = df_drive[LABEL_COL].values
        n_rows = len(SIGNALS) + 1

        fig = plt.figure(figsize=(14, 2.5 * n_rows))
        fig.suptitle(
            f"Cycle: {cycle_name} | drive_id={drive_id} | split={df_drive['split'].iloc[0]}",
            fontsize=14,
            y=0.99
        )

        # plot signals...
        for row_idx, (col, nice_name) in enumerate(SIGNALS, start=1):
            sig = df_drive[col].values
            ax = plt.subplot(n_rows, 1, row_idx)
            ax.plot(time, sig, label=nice_name)
            ax.set_ylabel(nice_name)
            ax.grid(True, alpha=0.3)

        # anomaly row
        ax = plt.subplot(n_rows, 1, n_rows)
        ax.step(time, labels, where="post")
        ax.set_ylim(-0.1, 1.2)
        ax.set_ylabel("anomaly")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"{cycle_name}_drive_{drive_id}.png")
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print("Saved:", out_path)

        if show:
            plt.show()
        else:
            plt.close(fig)
