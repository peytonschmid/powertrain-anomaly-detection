
"""
Generate and inspect synthetic EPA-based powertrain drives.
This script ONLY loads synthetic drives, prints dataset summaries,
and optionally saves or plots examples.
"""

from pathlib import Path
from powertrain_anomaly_detection.config import CONFIG
from powertrain_anomaly_detection.data import generate_synthetic_dataset
from powertrain_anomaly_detection.preprocessing import preprocess_dataset
from powertrain_anomaly_detection.plots import plot_example_drives_by_cycle


def ensure_dir(path):
    Path(path).mkdir(exist_ok=True, parents=True)

def main(save_csv=False, plot_examples=False):
    print("\n=== Generating Synthetic Drives ===")

    # Load synthetic training/test drives BEFORE windowing
    df_train, df_test = generate_synthetic_dataset()

    print(f"\nLoaded synthetic drives:")
    print(f"Train rows: {len(df_train):,}")
    print(f"Test rows:  {len(df_test):,}")
    print("\nPreview of training data:")
    print(df_train.head())

    if save_csv:
        ensure_dir("data_out")
        df_train.to_csv("data_out/synth_train.csv", index=False)
        df_test.to_csv("data_out/synth_test.csv", index=False)
        print("\nSaved synthetic CSVs to data_out/")

    # OPTIONAL: run preprocessing but not windowing
    print("\n=== Preprocessing (Normalization Only) ===")
    train_np, test_np, scaler = preprocess_dataset(df_train, df_test, CONFIG)
    print(f"Processed shapes → train: {train_np.shape}, test: {test_np.shape}")

    if plot_examples:
        print("\n=== Plotting Sample Drives by EPA Cycle ===")
        plot_example_drives_by_cycle(df_train)

    print("\nDone.\n")
    return df_train, df_test

if __name__ == "__main__":
    # Set these True/False depending on what you want the script to do by default
    main(save_csv=True, plot_examples=True)
