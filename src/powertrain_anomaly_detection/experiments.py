import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .config import CONFIG, numeric_cols, WINDOW_S
from .data import generate_synthetic_dataset, build_runs_from_df
from .preprocessing import per_mode_standardize, make_windows
from .training import train_ensemble, score_ensemble
from .selector import select_operating_point, event_metrics, smooth_hysteresis


EXPERIMENTS = [
    {
        "name": "baseline_residual_per_mode",
        "arch": "residual",
        "per_mode": True,
        "description": "Residual Conv1d AE + per-mode (Mode,Gear) normalization",
    },
    {
        "name": "ablation_plain_per_mode",
        "arch": "plain",
        "per_mode": True,
        "description": "Plain Conv1d AE (no residual) + per-mode normalization",
    },
    {
        "name": "ablation_residual_global_norm",
        "arch": "residual",
        "per_mode": False,
        "description": "Residual Conv1d AE + global normalization (no per-mode)",
    },
    {
        "name": "tcn_residual_per_mode",
        "arch": "tcn",
        "per_mode": True,
        "description": "TCN-style dilated Conv1d AE + per-mode norm",
    },
]


def evaluate_run_with_q(models, td: pd.DataFrame, q: float, per_mode: bool, label: str):
    td_std = per_mode_standardize(td, numeric_cols, per_mode=per_mode)
    Xt, yt, tt, md = make_windows(
        td_std, numeric_cols,
        WINDOW_S, CONFIG["STRIDE_SEC"], CONFIG["HZ"]
    )
    sc = score_ensemble(models, Xt)

    thr = {m: np.percentile(sc[md == m], q) if (md == m).any() else np.percentile(sc, q)
           for m in np.unique(md)}

    _, flags = smooth_hysteresis(
        sc, md, thr,
        CONFIG["SMOOTH_ROLL"],
        CONFIG["HYST_STEPS"],
        CONFIG["COOL_DOWN_STEPS"],
    )

    total_time_s = td["time_s"].iloc[-1] - td["time_s"].iloc[0]
    ev = event_metrics(yt.astype(int), flags, tt,
                       CONFIG["STRIDE_SEC"], total_time_s)
    ap = average_precision_score(yt, sc)
    auc = roc_auc_score(yt, sc) if len(np.unique(yt)) > 1 else float("nan")
    ev.update({"AP": ap, "AUC": auc})
    return ev

def run_experiments():
    print("Generating synthetic dataset...")
    df_all = generate_synthetic_dataset()
    train_runs, test_runs = build_runs_from_df(df_all)

    print("# TRAIN_NORMAL_RUNS:", len(train_runs))
    print("# TEST_RUNS:", len(test_runs))
    print("Any anomalies in TEST_RUNS?:",
          any(td["anomaly"].sum() > 0 for td in test_runs))

    all_results = []
    all_models = {}
    start_time = time.time()

    for exp in EXPERIMENTS:
        name = exp["name"]
        arch = exp["arch"]
        per_mode = exp["per_mode"]
        desc = exp["description"]

        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(desc)
        print("=" * 80)

        X_train_list = []
        for df in train_runs:
            df_std = per_mode_standardize(df, numeric_cols, per_mode=per_mode)
            X, y, t0, md = make_windows(
                df_std, numeric_cols,
                WINDOW_S, CONFIG["STRIDE_SEC"], CONFIG["HZ"]
            )
            X_train_list.append(X)
        X_train = np.concatenate(X_train_list, axis=0)
        print("Train windows:", X_train.shape)

        models = train_ensemble(
            X_train,
            arch=arch,
            per_mode_name=name,
            num_models=CONFIG["ENSEMBLE"]["num_models"],
        )

        all_models[name] = models

        calib_idx = next(
            (i for i, td in enumerate(test_runs) if td["anomaly"].sum() > 0),
            None
        )
        if calib_idx is None:
            raise ValueError("No anomalous test drives found for selector calibration.")

        choice = select_operating_point(
            models,
            test_runs[calib_idx],
            window_s=WINDOW_S,
            per_mode=per_mode,
            label=name,
        )
        print(f"\nChosen operating point for {name}: {choice}")
        q_star = choice["q"]

        rows = []
        for i, td in enumerate(test_runs):
            r = evaluate_run_with_q(
                models, td, q=q_star, per_mode=per_mode, label=name
            )
            r["run"] = i
            r["exp"] = name
            rows.append(r)
            print(
                f"[{name}] Run {i:02d}: "
                f"P={r['precision']:.2f} R={r['recall']:.2f} "
                f"FAPH={r['FAPH']:.2f} delay={r['avg_delay_s']:.2f}s | "
                f"AP={r['AP']:.2f} AUC={r['AUC']:.2f}"
            )

        df_exp = pd.DataFrame(rows)
        print("\nMEANS (event-level) for", name)
        print(df_exp[["precision", "recall", "avg_delay_s",
                      "FAPH", "AP", "AUC"]].mean(numeric_only=True))
        all_results.append(df_exp)

    exec_time = time.time() - start_time
    print("Execution Time:", exec_time)

    results_df = pd.concat(all_results, ignore_index=True)
    return df_all, results_df, all_models

def run_tcn_experiments():
    df_all = generate_synthetic_dataset()
    train_runs, test_runs = build_runs_from_df(df_all)

    all_tcn_results = []
    start_time = time.time()

    exp = {
        "name": "tcn_residual_per_mode",
        "arch": "tcn",
        "per_mode": True,
        "description": "TCN-style dilated Conv1d AE + per-mode norm",
    }
    name = exp["name"]
    arch = exp["arch"]
    per_mode = exp["per_mode"]
    desc = exp["description"]

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {name}")
    print(desc)
    print("=" * 80)

    X_train_list = []
    for df in train_runs:
        df_std = per_mode_standardize(df, numeric_cols, per_mode=per_mode)
        X, y, t0, md = make_windows(
            df_std, numeric_cols, WINDOW_S, CONFIG["STRIDE_SEC"], CONFIG["HZ"]
        )
        X_train_list.append(X)
    X_train = np.concatenate(X_train_list, axis=0)
    print("Train windows:", X_train.shape)

    models = train_ensemble(
        X_train,
        arch=arch,
        per_mode_name=name,
        num_models=CONFIG["ENSEMBLE"]["num_models"],
    )

    calib_idx = next(
        (i for i, td in enumerate(test_runs) if td["anomaly"].sum() > 0),
        None
    )
    if calib_idx is None:
        raise ValueError("No anomalous test drives found for selector calibration.")

    choice = select_operating_point(
        models,
        test_runs[calib_idx],
        window_s=WINDOW_S,
        per_mode=per_mode,
        label=name,
    )
    print(f"\nChosen operating point for {name}: {choice}")
    q_star = choice["q"]

    rows = []
    for i, td in enumerate(test_runs):
        r = evaluate_run_with_q(models, td, q=q_star, per_mode=per_mode, label=name)
        r["run"] = i
        r["exp"] = name
        rows.append(r)
        print(
            f"[{name}] Run {i:02d}: "
            f"P={r['precision']:.2f} R={r['recall']:.2f} "
            f"FAPH={r['FAPH']:.2f} delay={r['avg_delay_s']:.2f}s | "
            f"AP={r['AP']:.2f} AUC={r['AUC']:.2f}"
        )

    df_exp = pd.DataFrame(rows)
    print("\nMEANS (event-level) for", name)
    print(df_exp[["precision", "recall", "avg_delay_s",
                    "FAPH", "AP", "AUC"]].mean(numeric_only=True))
    all_tcn_results.append(df_exp)

    exec_time = time.time() - start_time
    print("Execution Time:", exec_time)

    results_df = pd.concat(all_tcn_results, ignore_index=True)
    return df_all, results_df, models, q_star


