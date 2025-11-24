#!/usr/bin/env python3
"""
Demo script: load a pre-trained TCN-AE, run anomaly detection on one drive,
and visualize true vs predicted anomalies.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from powertrain_anomaly_detection.config import CONFIG, numeric_cols, WINDOW_S
from powertrain_anomaly_detection.data import generate_synthetic_dataset
from powertrain_anomaly_detection.preprocessing import per_mode_standardize, make_windows
from powertrain_anomaly_detection.models import BaseConv1dAE  # adapt if class name differs
from powertrain_anomaly_detection.selector import (
    smooth_hysteresis,
    event_metrics,
    per_mode_quantiles,
)

CHECKPOINT_PATH = "checkpoints/tcn_ae_4.pt"  
WINDOW_S = 6

def pick_demo_drive(df_all):
    """Pick a single test drive with anomalies."""
    candidates = df_all[df_all["split"] == "test_anom"]["drive_id"].unique()
    if len(candidates) == 0:
        # fall back to any test drive
        candidates = df_all[df_all["split"].str.contains("test")]["drive_id"].unique()
    drive_id = int(candidates[0])
    df_drive = df_all[df_all["drive_id"] == drive_id].copy()
    return drive_id, df_drive

def build_tcn_model(in_dim: int) -> BaseConv1dAE:
    return BaseConv1dAE(
        in_dim=in_dim,
        hidden=CONFIG["AE_ARCH"]["hidden"],
        bottleneck=CONFIG["AE_ARCH"]["bottleneck"],
        arch="tcn",
        loss=CONFIG["AE_ARCH"]["loss"],
        dropout=CONFIG["AE_ARCH"]["dropout"],
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("\n=== Loading synthetic dataset ===")
    df_all = generate_synthetic_dataset()

    drive_id, df_demo = pick_demo_drive(df_all)
    print(f"Using drive_id={drive_id}, split={df_demo['split'].iloc[0]}, "
          f"cycle={df_demo['cycle_name'].iloc[0]}")

    # ---------------- Load model ----------------
    in_dim = len(numeric_cols)

    model = build_tcn_model(in_dim).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded TCN-AE from {CHECKPOINT_PATH}")

    # ---------------- Preprocess & window ----------------
    df_std = per_mode_standardize(df_demo, numeric_cols, per_mode=True)
    X, y, t0, modes = make_windows(
        df_std,
        numeric_cols,
        WINDOW_S,
        CONFIG["STRIDE_SEC"],
        CONFIG["HZ"],
    )
    print("Windowed shape:", X.shape)

    # ---------------- Compute scores ----------------
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        xr = model(X_t)
        if CONFIG["AE_ARCH"]["loss"] == "l1":
            scores = torch.mean(torch.abs(xr - X_t), dim=(1, 2)).cpu().numpy()
        else:
            scores = torch.mean((xr - X_t) ** 2, dim=(1, 2)).cpu().numpy()

    # Pick a demo threshold using a high quantile
    q = 95.0
    thr_map = per_mode_quantiles(scores, modes, q)
    smoothed_scores, flags = smooth_hysteresis(
        scores,
        modes,
        thr_map,
        CONFIG["SMOOTH_ROLL"],
        CONFIG["HYST_STEPS"],
        CONFIG["COOL_DOWN_STEPS"],
    )

    total_time_s = df_demo["time_s"].iloc[-1] - df_demo["time_s"].iloc[0]
    metrics = event_metrics(y.astype(int), flags, t0,
                            CONFIG["STRIDE_SEC"], total_time_s)
    print("\n=== Event-level metrics (demo drive) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    # ---------------- Plot ----------------
    time = df_demo["time_s"].values
    labels = df_demo["anomaly"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    # Top: a couple of key signals
    axes[0].plot(time, df_demo["EngineRPM"].values, label="EngineRPM")
    axes[0].plot(time, df_demo["EngineTorque_Actual_Nm"].values, label="EngineTorque")
    axes[0].set_ylabel("Signals")
    axes[0].legend()
    axes[0].set_title(f"Drive {drive_id} — signals with anomaly regions")

    # Shade true anomaly regions
    in_anom = False
    start = None
    for i in range(len(time)):
        if labels[i] == 1 and not in_anom:
            in_anom = True
            start = time[i]
        if labels[i] == 0 and in_anom:
            in_anom = False
            axes[0].axvspan(start, time[i], alpha=0.15)
    if in_anom:
        axes[0].axvspan(start, time[-1], alpha=0.15)

    # Middle: smoothed reconstruction error
    axes[1].plot(t0, smoothed_scores, label="Smoothed recon. error")
    # Plot a single global threshold for illustration (e.g., median of thr_map)
    global_thr = np.median(list(thr_map.values()))
    axes[1].axhline(global_thr, linestyle="--", label=f"Threshold (~{q}th pct)")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].set_title("Anomaly score over time")

    # Bottom: true vs predicted anomaly flags (window-level)
    axes[2].step(t0, y, where="post", label="True (window label)")
    axes[2].step(t0, flags, where="post", label="Predicted flag")
    axes[2].set_ylabel("0/1")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    axes[2].set_title("True vs predicted anomaly events (windowed)")

    plt.tight_layout()
    Path("demo_outputs").mkdir(exist_ok=True)
    plt.savefig("demo_outputs/demo_detection.png", dpi=200)
    plt.show()

    print("\nSaved plot to demo_outputs/demo_detection.png\n")


if __name__ == "__main__":
    main()
