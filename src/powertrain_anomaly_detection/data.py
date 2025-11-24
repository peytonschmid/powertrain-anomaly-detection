import re
import numpy as np
import pandas as pd
import requests

from .config import EPA_CYCLE_URLS, numeric_cols

def load_epa_cycle_from_url(url: str, target_dt: float = 0.02) -> pd.DataFrame:
    """
    Load an EPA drive cycle stored as a single text line:
      "... Time (sec), Speed (mph) 0 0 1 0 2 0 ..."

    Returns a DataFrame with:
      - time_s (float)
      - VehicleSpeed_kph (float)
    resampled to uniform target_dt.
    """
    # Download raw text
    resp = requests.get(url)
    resp.raise_for_status()
    txt = resp.text

    # Extract all numeric tokens (ints / floats / scientific notation)
    # [t0, v0, t1, v1, t2, v2, ...]
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", txt)
    vals = np.array(nums, dtype=float)

    if len(vals) < 4:
        raise ValueError(f"Not enough numeric values found in {url}")

    # If odd count, drop the last one
    if len(vals) % 2 == 1:
        vals = vals[:-1]

    t = vals[0::2]       # even indices → time
    v_mph = vals[1::2]   # odd indices → speed (mph)

    # Make a uniform time grid at target_dt
    t_min, t_max = t.min(), t.max()
    t_uniform = np.arange(t_min, t_max + 1e-9, target_dt)

    # Interpolate speed onto uniform grid
    v_uniform_mph = np.interp(t_uniform, t, v_mph)
    v_uniform_kph = v_uniform_mph * 1.60934

    df = pd.DataFrame({
        "time_s": t_uniform,
        "VehicleSpeed_kph": v_uniform_kph
    })
    return df

def simulate_drive_from_speed(df_speed: pd.DataFrame,
                              drive_id: int,
                              split: str,
                              cycle_name: str,
                              rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Take a speed trace (time_s, VehicleSpeed_kph) and synthesize
    the rest of your numeric_cols + metadata (anomaly, drive_id, split).
    """
    if rng is None:
        rng = np.random.default_rng(10_000 + drive_id)

    df = df_speed.copy()

    # Simple gear logic based on speed buckets
    speed = df["VehicleSpeed_kph"]
    # Tuneable thresholds
    bins = [0, 5, 20, 40, 60, 90, 130, 300]
    labels = [1, 2, 3, 4, 5, 6, 7]  # up to 7 gears
    df["Gear"] = pd.cut(speed, bins=bins, labels=labels, right=False).astype(float)

    # Engine RPM roughly from speed + gear
    # rpm ≈ speed_kph * factor / gear
    base_factor = 120.0  # tweakable overall ratio
    gear_safe = df["Gear"].replace(0, 1.0)
    df["EngineRPM"] = (speed * base_factor / gear_safe).clip(600, 6000)

    # Approximate accel from speed derivative
    dt_series = df["time_s"].diff().fillna(0.02)
    accel_kph_s = speed.diff().fillna(0) / dt_series.replace(0, 0.02)
    # Pedal mostly follows positive accel + baseline
    df["AccelPedalPos_pct"] = np.clip(15.0 * accel_kph_s.clip(lower=0) + 5.0 * (speed > 1.0), 0, 100)

    # Throttle command / actual
    df["ThrottlePos_Command_pct"] = (
        df["AccelPedalPos_pct"]
        .rolling(10, min_periods=1)
        .mean()
        .clip(0, 100)
    )
    df["ThrottlePos_Actual_pct"] = np.clip(
        df["ThrottlePos_Command_pct"] + rng.normal(0, 1.5, size=len(df)),
        0,
        100
    )

    # Baro, MAP, MAF, temps
    df["Baro_kPa"] = 100.0 + rng.normal(0, 0.5, size=len(df))
    load_frac = df["ThrottlePos_Actual_pct"] / 100.0
    df["MAP_kPa"] = np.clip(30.0 + 70.0 * load_frac, 20.0, 250.0)

    # Intake air temp: slightly rising with speed, plus some noise
    df["IntakeAirTemp_C"] = 25.0 + 0.02 * speed + rng.normal(0, 1.0, size=len(df))

    # Coolant temp: warm-up from 40 C to ~95 C over first 600 s
    warmup = np.minimum(df["time_s"], 600.0) / 600.0
    df["CoolantTemp_C"] = 40.0 + 55.0 * warmup + rng.normal(0, 0.5, size=len(df))

    # MAF: roughly proportional to RPM * load
    df["MAF_gps"] = np.clip(0.02 * df["EngineRPM"] * load_frac, 0, 300.0)

    # Lambda target & measured lambda + trims
    # Slightly rich at idle/low speed, ~stoich at cruise
    df["Lambda_Target"] = 1.0 - 0.05 * (speed < 5.0).astype(float)
    df["Lambda"] = df["Lambda_Target"] + rng.normal(0, 0.01, size=len(df))

    df["STFT_B1_pct"] = rng.normal(0, 3.0, size=len(df))
    df["LTFT_B1_pct"] = rng.normal(0, 5.0, size=len(df))

    # Spark advance & knock
    df["SparkAdvance_deg"] = (
        10.0 + 0.06 * speed - 0.002 * df["EngineRPM"] + rng.normal(0, 1.0, size=len(df))
    )
    df["KnockRetard_deg"] = np.clip(rng.normal(0, 0.5, size=len(df)), 0, None)

    # Torque request & actual
    df["DriverTorqueRequest_Nm"] = 20.0 + 2.0 * load_frac * (speed + 5.0)
    df["EngineTorque_Actual_Nm"] = df["DriverTorqueRequest_Nm"] + rng.normal(0, 5.0, size=len(df))

    # Error channels
    df["ThrottleErr_pct"] = df["ThrottlePos_Command_pct"] - df["ThrottlePos_Actual_pct"]
    df["TorqueErr_Nm"] = df["DriverTorqueRequest_Nm"] - df["EngineTorque_Actual_Nm"]

    # Derived / engineered features
    df["dRPM"] = df["EngineRPM"].diff().fillna(0.0)
    df["dMAP"] = df["MAP_kPa"].diff().fillna(0.0)
    df["LambdaErr"] = df["Lambda"] - df["Lambda_Target"]
    df["MAP_over_Baro"] = df["MAP_kPa"] / df["Baro_kPa"].replace(0, np.nan)
    df["MAF_over_RPM"] = df["MAF_gps"] / df["EngineRPM"].replace(0, np.nan)
    df["MAF_over_RPM"] = df["MAF_over_RPM"].fillna(0.0)
    df["TqRatio"] = df["EngineTorque_Actual_Nm"] / df["DriverTorqueRequest_Nm"].replace(0, np.nan)
    df["TqRatio"] = df["TqRatio"].fillna(0.0)

    # Simple ignition / PRND 
    df["IgnitionState"] = np.where(df["VehicleSpeed_kph"] > 0.5, "ON", "OFF")
    df["PRND"] = np.where(df["VehicleSpeed_kph"] > 0.5, "D", "P")

    # Metadata
    df["anomaly"] = 0  # inject anomalies later
    df["drive_id"] = drive_id
    df["split"] = split
    df["cycle_name"] = cycle_name

    # Make sure all numeric_cols exist
    missing = [c for c in numeric_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected numeric columns: {missing}")

    return df

def inject_anomalies_into_drive(df_drive: pd.DataFrame,
                                rng: np.random.Generator,
                                max_events: int = 3) -> pd.DataFrame:
    """
    Inject anomalies into a single drive (all rows for one drive_id).
    Returns a modified copy with 'anomaly' set to 1 in anomalous regions.
    """
    # Copy AND reset index so 0..n-1 matches idx positions
    df = df_drive.copy().reset_index(drop=True)
    n = len(df)
    if n < 100:
        return df  # too short, skip

    # Ensure anomaly column exists and starts at 0
    df["anomaly"] = 0

    # How many distinct anomaly segments in this drive?
    n_events = int(rng.integers(1, max_events + 1))

    for _ in range(n_events):
        # Choose a random segment length between ~3–12 seconds (0.02s step)
        seg_len = int(rng.integers(150, 600))  # 150*0.02=3s to 600*0.02=12s
        if seg_len >= n:
            seg_len = max(50, n // 3)

        start_idx = int(rng.integers(0, max(1, n - seg_len)))
        end_idx = start_idx + seg_len
        idx = np.arange(start_idx, end_idx)

        anomaly_type = rng.choice([
            "rich_lambda_misfire",
            "boost_leak_low_map",
            "sensor_stuck_map",
            "oscillating_throttle",
            "torque_loss"
        ])

        if anomaly_type == "rich_lambda_misfire":
            # Lambda suddenly rich + trims go crazy + RPM jitter
            df.loc[idx, "Lambda"] = np.clip(
                df.loc[idx, "Lambda_Target"] - 0.15, 0.7, 1.0
            )
            df.loc[idx, "STFT_B1_pct"] = (
                df.loc[idx, "STFT_B1_pct"] + rng.normal(15, 5, size=len(idx))
            )
            df.loc[idx, "LTFT_B1_pct"] = (
                df.loc[idx, "LTFT_B1_pct"] + rng.normal(10, 3, size=len(idx))
            )
            df.loc[idx, "EngineRPM"] = np.clip(
                df.loc[idx, "EngineRPM"] + rng.normal(0, 250, size=len(idx)),
                600, 6000
            )

        elif anomaly_type == "boost_leak_low_map":
            # MAP much lower than expected for given load, torque under-delivery
            df.loc[idx, "MAP_kPa"] = np.clip(
                df.loc[idx, "MAP_kPa"] * 0.6, 20.0, None
            )
            df.loc[idx, "EngineTorque_Actual_Nm"] = (
                df.loc[idx, "EngineTorque_Actual_Nm"] * 0.6
            )
            df.loc[idx, "TorqueErr_Nm"] = (
                df.loc[idx, "DriverTorqueRequest_Nm"]
                - df.loc[idx, "EngineTorque_Actual_Nm"]
            )
            df.loc[idx, "TqRatio"] = (
                df.loc[idx, "EngineTorque_Actual_Nm"]
                / df.loc[idx, "DriverTorqueRequest_Nm"].replace(0, np.nan)
            )
            df["TqRatio"] = df["TqRatio"].fillna(0.0)

        elif anomaly_type == "sensor_stuck_map":
            # MAP flat-lines (stuck sensor) while other signals change
            stuck_value = float(df.loc[idx, "MAP_kPa"].iloc[0])
            df.loc[idx, "MAP_kPa"] = stuck_value
            df.loc[idx, "EngineRPM"] = np.clip(
                df.loc[idx, "EngineRPM"] + rng.normal(0, 150, size=len(idx)),
                600, 6000
            )
            df.loc[idx, "ThrottlePos_Actual_pct"] = np.clip(
                df.loc[idx, "ThrottlePos_Actual_pct"] +
                rng.normal(0, 3, size=len(idx)),
                0, 100
            )

        elif anomaly_type == "oscillating_throttle":
            # Throttle hunts up/down rapidly, with error vs command
            t_local = np.linspace(0, 4 * np.pi, len(idx))
            osc = 10 * np.sin(t_local)  # +/- 10%
            df.loc[idx, "ThrottlePos_Actual_pct"] = np.clip(
                df.loc[idx, "ThrottlePos_Command_pct"]
                + osc
                + rng.normal(0, 2, size=len(idx)),
                0, 100
            )
            df.loc[idx, "ThrottleErr_pct"] = (
                df.loc[idx, "ThrottlePos_Command_pct"]
                - df.loc[idx, "ThrottlePos_Actual_pct"]
            )

        elif anomaly_type == "torque_loss":
            # Torque suddenly collapses at given pedal / RPM
            df.loc[idx, "EngineTorque_Actual_Nm"] = (
                df.loc[idx, "EngineTorque_Actual_Nm"] * 0.3
            )
            df.loc[idx, "TorqueErr_Nm"] = (
                df.loc[idx, "DriverTorqueRequest_Nm"]
                - df.loc[idx, "EngineTorque_Actual_Nm"]
            )
            df.loc[idx, "TqRatio"] = (
                df.loc[idx, "EngineTorque_Actual_Nm"]
                / df.loc[idx, "DriverTorqueRequest_Nm"].replace(0, np.nan)
            )
            df["TqRatio"] = df["TqRatio"].fillna(0.0)

        # Recompute derived signals in the anomalous segment
        df_full_dRPM = df["EngineRPM"].diff().fillna(0.0)
        df_full_dMAP = df["MAP_kPa"].diff().fillna(0.0)

        df.loc[idx, "dRPM"] = df_full_dRPM.iloc[idx].values
        df.loc[idx, "dMAP"] = df_full_dMAP.iloc[idx].values

        df["LambdaErr"] = df["Lambda"] - df["Lambda_Target"]
        df["MAP_over_Baro"] = (
            df["MAP_kPa"] / df["Baro_kPa"].replace(0, np.nan)
        ).fillna(0.0)
        df["MAF_over_RPM"] = (
            df["MAF_gps"] / df["EngineRPM"].replace(0, np.nan)
        ).fillna(0.0)

        # Mark anomaly labels
        df.loc[idx, "anomaly"] = 1

    return df

def generate_synthetic_dataset(
    reps_per_cycle_train: int = 15,
    reps_per_cycle_test: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate df_all across all EPA cycles and inject anomalies."""
    all_drives = []
    drive_id = 0

    for cycle_name, url in EPA_CYCLE_URLS.items():
        df_speed = load_epa_cycle_from_url(url, target_dt=0.02)

        # training: normal only
        for _ in range(reps_per_cycle_train):
            df_drive = simulate_drive_from_speed(
                df_speed, drive_id=drive_id,
                split="train_normal", cycle_name=cycle_name,
                rng=np.random.default_rng(1000 + drive_id),
            )
            all_drives.append(df_drive)
            drive_id += 1

        # test: normal before injection
        for _ in range(reps_per_cycle_test):
            df_drive = simulate_drive_from_speed(
                df_speed, drive_id=drive_id,
                split="test_normal", cycle_name=cycle_name,
                rng=np.random.default_rng(2000 + drive_id),
            )
            all_drives.append(df_drive)
            drive_id += 1

    df_all = pd.concat(all_drives, ignore_index=True)

    # anomaly injection
    rng = np.random.default_rng(seed)
    df_list = []
    for drive_id, df_drive in df_all.groupby("drive_id"):
        split = df_drive["split"].iloc[0]

        if split == "train_normal":
            df_list.append(df_drive)
            continue

        if split == "test_normal" and rng.random() < 0.9:
            df_anom = inject_anomalies_into_drive(df_drive, rng, max_events=3)
            df_anom["split"] = "test_anom"
            df_list.append(df_anom)
        else:
            df_list.append(df_drive)

    return pd.concat(df_list, ignore_index=True)

def build_runs_from_df(df: pd.DataFrame) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Split df_all into per-drive training and test runs."""
    train_runs, test_runs = [], []
    df = df.sort_values(["drive_id", "time_s"])

    for drive_id, g in df.groupby("drive_id"):
        split = g["split"].iloc[0]
        if split == "train_normal":
            train_runs.append(g)
        else:
            test_runs.append(g)

    return train_runs, test_runs

