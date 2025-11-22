
# Temporal Convolutional Autoencoder for Powertrain Anomaly Detection

#### ProductPrototype (Conv1d / TCN + Ensemble + Ablations)

This notebook implements:

- Synthetic **engine/powertrain drive cycles** with labeled anomalies.
- Per-mode **normalization** by (Mode, Gear), with an ablation to turn it off.
- A family of **1D Conv autoencoders**:
  - `plain` Conv1d AE
  - `residual` Conv1d AE (output skip: x̂ + x)
  - `tcn` Conv1d AE with **dilated** Conv blocks and residuals
- An **ensemble** of AEs (N models; mean score).
- **Mode-aware thresholds**, smoothing, hysteresis, cool-down.
- A **selector** that chooses an operating point (quantile q) under:
  - Precision floor
  - False alarms per hour (FAPH) budget
- Event-level metrics:
  - precision, recall, avg delay, FAPH, AP, ROC AUC
- Four experiments:
  1. **Baseline**: residual AE + per-mode norm
  2. **Ablation A**: plain AE + per-mode norm
  3. **Ablation B**: residual AE + global norm (no per-mode)
  4. **TCN Baseline**: Temporal Convolutional Network AE

## Install

```bash
pip install -r requirements.txt
