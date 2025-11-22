import numpy as np
import pandas as pd

from .config import CONFIG, numeric_cols
from .preprocessing import per_mode_standardize, make_windows
from .training import score_ensemble

def windows_to_events(mask: np.ndarray):
    ev = []
    i = 0
    while i < len(mask):
        if mask[i] == 1:
            j = i
            while j+1 < len(mask) and mask[j+1] == 1:
                j += 1
            ev.append((i, j))
            i = j + 1
        else:
            i += 1
    return ev


def event_metrics(true: np.ndarray,
                  pred: np.ndarray,
                  tt: np.ndarray,
                  stride_s: int,
                  total_time_s: float):
    gt = windows_to_events(true)
    pr = windows_to_events(pred)

    def ov(a, b):
        (s1, e1), (s2, e2) = a, b
        return max(0, min(e1, e2) - max(s1, s2) + 1)

    TP = []
    FP = []
    FN = []
    used = set()

    for g in gt:
        cand = [k for k, p in enumerate(pr) if ov(g, p) > 0]
        if cand:
            k = max(cand, key=lambda kk: ov(g, pr[kk]))
            used.add(k)
            delay = max(0, pr[k][0] - g[0]) * stride_s
            TP.append({"delay": delay})
        else:
            FN.append({})

    for k, p in enumerate(pr):
        if k not in used:
            FP.append({})

    rec = len(TP)/(len(TP)+len(FN)) if (len(TP)+len(FN)) > 0 else 0.0
    prec = len(TP)/(len(TP)+len(FP)) if (len(TP)+len(FP)) > 0 else 0.0
    delay = float(np.mean([x["delay"] for x in TP])) if TP else float("nan")
    hours = max(total_time_s/3600.0, 1e-9)
    faph = len(FP) / hours

    return {"recall": rec, "precision": prec, "avg_delay_s": delay, "FAPH": faph}


def smooth_hysteresis(scores: np.ndarray,
                      modes: np.ndarray,
                      thr_map: dict,
                      roll: int,
                      k: int,
                      cool_down_steps: int):
    ss = pd.Series(scores).rolling(roll, center=True).mean().fillna(method="bfill").fillna(method="ffill").values
    raw = np.array([1 if ss[i] > thr_map.get(modes[i], np.inf) else 0 for i in range(len(ss))])

    flags = []
    streak = 0
    cooldown = 0

    for r in raw:
        if cooldown > 0:
            flags.append(0)
            cooldown -= 1
            continue
        streak = streak + 1 if r == 1 else 0
        fire = 1 if streak >= k else 0
        flags.append(fire)
        if fire == 1:
            cooldown = cool_down_steps

    return ss, np.array(flags)


def fbeta(prec, rec, beta=2.0):
    b2 = beta * beta
    if prec == 0 and rec == 0:
        return 0.0
    return (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12)


def per_mode_quantiles(scores, modes, q):
    thr = {}
    for md in np.unique(modes):
        vals = scores[modes == md]
        thr[md] = np.percentile(vals, q) if len(vals) > 0 else np.percentile(scores, q)
    return thr


def select_operating_point(models,
                           td: pd.DataFrame,
                           window_s: int,
                           per_mode: bool,
                           label: str):
    """
    Run selector on ONE validation/test drive (td) to pick best q.
    """
    td_std = per_mode_standardize(td, numeric_cols, per_mode=per_mode)
    Xv, yv, tv, md = make_windows(td_std, numeric_cols,
                                   window_s, CONFIG["STRIDE_SEC"], CONFIG["HZ"])
    scores = score_ensemble(models, Xv)

    best = None
    beta = CONFIG["SELECTOR"]["BETA"]

    for q in CONFIG["SELECTOR"]["Q_GRID"]:
        thr = per_mode_quantiles(scores, md, q)
        _, flags = smooth_hysteresis(
            scores, md, thr,
            CONFIG["SMOOTH_ROLL"],
            CONFIG["HYST_STEPS"],
            CONFIG["COOL_DOWN_STEPS"]
        )
        total_time_s = td["time_s"].iloc[-1] - td["time_s"].iloc[0]
        em = event_metrics(yv.astype(int), flags, tv,
                           CONFIG["STRIDE_SEC"], total_time_s)

        modal = pd.Series(md).mode().iloc[0] if len(md) > 0 else "DEFAULT"
        if "IDLE" in str(modal):
            p_floor = CONFIG["SELECTOR"]["P_FLOOR_IDLE"]
        else:
            p_floor = CONFIG["SELECTOR"]["P_FLOOR_DEFAULT"]

        ok = (em["precision"] >= p_floor) and (em["FAPH"] <= CONFIG["SELECTOR"]["FAPH_BUDGET"])
        score = fbeta(em["precision"], em["recall"], beta=beta)

        print(f"[SEL | {label}] q={q:.1f} | P={em['precision']:.2f} R={em['recall']:.2f} "
              f"FAPH={em['FAPH']:.2f} delay={em['avg_delay_s']:.2f}s | floor={p_floor:.2f} ok={ok}")

        row = {"q": q, **em, "F_beta": score, "p_floor_used": p_floor}

        if ok and (best is None or row["F_beta"] > best["F_beta"]):
            best = row

    if best is None:
        print(f"[WARN | {label}] No q met constraints; falling back to unconstrained best by F_beta")
        fallback = None
        for q in CONFIG["SELECTOR"]["Q_GRID"]:
            thr = per_mode_quantiles(scores, md, q)
            _, flags = smooth_hysteresis(
                scores, models, thr,
                CONFIG["SMOOTH_ROLL"],
                CONFIG["HYST_STEPS"],
                CONFIG["COOL_DOWN_STEPS"]
            )
            total_time_s = td["time_s"].iloc[-1] - td["time_s"].iloc[0]
            em = event_metrics(yv.astype(int), flags, tv,
                               CONFIG["STRIDE_SEC"], total_time_s)
            row = {"q": q, **em, "F_beta": fbeta(em["precision"], em["recall"], beta=beta),
                   "p_floor_used": None}
            if (fallback is None) or (row["F_beta"] > fallback["F_beta"]):
                fallback = row
        best = fallback

    return best