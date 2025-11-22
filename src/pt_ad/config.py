import numpy as np  # not strictly needed yet, but fine

CONFIG = {
    "HZ": 50,
    "STRIDE_SEC": 2,
    "SEED": 23,

    # AE architecture config
    "AE_ARCH": {
        "hidden": 128,
        "bottleneck": 64,
        "dropout": 0.0,
        "loss": "l2",  # "l2" (MSE) or "l1"
    },

    # AE training
    "AE_TRAIN": {
        "epochs": 500,
        "lr": 5e-4,
        "batch_size": 256,
        "noise_std": 0.01,
    },

    # Ensemble
    "ENSEMBLE": {
        "num_models": 5,
    },

    # Smoothing / hysteresis
    "SMOOTH_ROLL": 3,
    "HYST_STEPS": 3,
    "COOL_DOWN_STEPS": 5,

    # Selector
    "SELECTOR": {
        "P_FLOOR_DEFAULT": 0.30,
        "P_FLOOR_IDLE": 0.30,
        "FAPH_BUDGET": 3.0,
        "Q_GRID": [80.0, 85.0, 90.0, 92.0, 95.0, 97.0, 99.0],
        "BETA": 2.0,
    },
}

EPA_CYCLE_URLS = {
    "udds": "https://www.epa.gov/sites/default/files/2015-10/uddscol.txt",
    "ftp":  "https://www.epa.gov/sites/default/files/2015-10/ftpcol.txt",
    "hwy":  "https://www.epa.gov/sites/default/files/2015-10/hwycol.txt",
    "nyc":  "https://www.epa.gov/system/files/other-files/2025-03/epa-new-york-city-cycle.txt",
    "us06": "https://www.epa.gov/system/files/other-files/2025-03/us06col.txt",
}

numeric_cols = [
    "VehicleSpeed_kph","Gear","EngineRPM","AccelPedalPos_pct",
    "ThrottlePos_Command_pct","ThrottlePos_Actual_pct","MAF_gps","MAP_kPa","Baro_kPa",
    "IntakeAirTemp_C","CoolantTemp_C","Lambda","Lambda_Target","STFT_B1_pct","LTFT_B1_pct",
    "SparkAdvance_deg","KnockRetard_deg","DriverTorqueRequest_Nm","EngineTorque_Actual_Nm",
    "ThrottleErr_pct","TorqueErr_Nm","dRPM","dMAP","LambdaErr",
    "MAP_over_Baro","MAF_over_RPM","TqRatio",
]

WINDOW_S = 6  # seconds per window
