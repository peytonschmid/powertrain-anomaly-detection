"""
powertrain_anomaly_detection: Temporal Conv1D / 
TCN autoencoder framework for synthetic powertrain 
anomaly detection.
"""

from .config import CONFIG, numeric_cols, WINDOW_S, EPA_CYCLE_URLS
from .data import generate_synthetic_dataset, build_runs_from_df
from .experiments import run_experiments
from .plots import (
    pick_example_drives_by_cycle,
    plot_drive_signals_with_anomalies,
    plot_example_drives_by_cycle,
)

__all__ = [
    "CONFIG",
    "numeric_cols",
    "WINDOW_S",
    "EPA_CYCLE_URLS",
    "generate_synthetic_dataset",
    "build_runs_from_df",
    "run_experiments",
    "pick_example_drives_by_cycle",
    "plot_drive_signals_with_anomalies",
    "plot_example_drives_by_cycle",
]
