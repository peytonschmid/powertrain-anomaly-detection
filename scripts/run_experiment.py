from powertrain_anomaly_detection.experiments import run_experiments
from powertrain_anomaly_detection.plots import plot_example_drives_by_cycle

if __name__ == "__main__":
    df_all, results_df = run_experiments()
    print("\nFinal summary (grouped by experiment):")
    summary = (
        results_df
        .groupby("exp")[["precision", "recall", "avg_delay_s", "FAPH", "AP", "AUC"]]
        .mean(numeric_only=True)
    )
    print(summary)

    plot_example_drives_by_cycle(df_all)
