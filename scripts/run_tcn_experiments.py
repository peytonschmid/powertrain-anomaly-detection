import os
import torch

from powertrain_anomaly_detection.experiments import run_tcn_experiments
from powertrain_anomaly_detection.plots import plot_example_drives_by_cycle

if __name__ == "__main__":
    df_all, results_df, tcn_models, qstar = run_tcn_experiments()
    if tcn_models is not None:
        ROOT = os.path.dirname(os.path.dirname(__file__))  # go up from scripts/
        ckpt_dir = os.path.join(ROOT, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        for i, m in enumerate(tcn_models):
            torch.save(m.state_dict(), f"checkpoints/tcn_ae_{i}.pt")
            print(f"[CKPT] Saved: checkpoints/tcn_ae_{i}.pt")
    print("\nFinal summary (grouped by experiment):")
    summary = (
        results_df
        .groupby("exp")[["precision", "recall", "avg_delay_s", "FAPH", "AP", "AUC"]]
        .mean(numeric_only=True)
    )
    print(summary)

    plot_example_drives_by_cycle(df_all)