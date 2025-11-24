"""
Generate and inspect synthetic EPA-based powertrain drives.
This script ONLY loads synthetic drives, prints dataset summaries,
and optionally saves or plots examples.
"""

from pathlib import Path
from powertrain_anomaly_detection.config import CONFIG, numeric_cols
from powertrain_anomaly_detection.data import generate_synthetic_dataset, build_runs_from_df
from powertrain_anomaly_detection.preprocessing import per_mode_standardize, make_windows
from powertrain_anomaly_detection.plots import plot_example_drives_by_cycle


def ensure_dir(path):
    Path(path).mkdir(exist_ok=True, parents=True)

def main(save_csv=False, plot_examples=False):
    print("\n=== Generating Synthetic Drives ===")

    # Load full synthetic dataset
    df_all = generate_synthetic_dataset()

    # Split into train/test runs (same as run_experiments)
    train_runs, test_runs = build_runs_from_df(df_all)

    print("\nLoaded synthetic drives:")
    print(f"Train runs: {len(train_runs)}")
    print(f"Test runs:  {len(test_runs)}")

    # Print preview of the first training run
    print("\nPreview of first training drive:")
    print(train_runs[0].head())

    # === Optional CSV output ===
    if save_csv:
        ensure_dir("data_out")
        train_runs[0].to_csv("data_out/synth_train_example.csv", index=False)
        test_runs[0].to_csv("data_out/synth_test_example.csv", index=False)
        print("\nSaved example CSVs to data_out/")

    print("\n=== Preprocessing example drive (normalization only) ===")
    sample_df = train_runs[0]

    df_std = per_mode_standardize(
        sample_df,
        numeric_cols,
        per_mode=True   # consistent with TCN experiment defaults
    )

    X, y, t0, md = make_windows(
        df_std,
        numeric_cols,
        CONFIG["WINDOW_S"],
        CONFIG["STRIDE_SEC"],
        CONFIG["HZ"]
    )

    print(f"Windowed shape: {X.shape} (windows x time x features)")

    if plot_examples:
        print("\n=== Plotting Sample Drives by EPA Cycle ===")
        plot_example_drives_by_cycle(df_all)

    print("\nDone.\n")
    return df_all, train_runs, test_runs

if __name__ == "__main__":
    main(save_csv=True, plot_examples=True)

