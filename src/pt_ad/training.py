import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from .config import CONFIG
from .models import BaseConv1dAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_ae_single(X: np.ndarray,
                  arch: str,
                  per_mode_name: str,
                  epochs: int | None = None,
                  lr: float | None = None,
                  batch_size: int | None = None,
                  noise_std: float | None = None,
                  verbose: bool = True) -> BaseConv1dAE:
    if epochs is None: epochs = CONFIG["AE_TRAIN"]["epochs"]
    if lr is None: lr = CONFIG["AE_TRAIN"]["lr"]
    if batch_size is None: batch_size = CONFIG["AE_TRAIN"]["batch_size"]
    if noise_std is None: noise_std = CONFIG["AE_TRAIN"]["noise_std"]

    in_dim = X.shape[-1]
    model = BaseConv1dAE(in_dim=in_dim, arch=arch).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            if noise_std and noise_std > 0:
                noise = noise_std * torch.randn_like(xb)
                xb_noisy = xb + noise
            else:
                xb_noisy = xb

            xr = model(xb_noisy)
            L = model.loss(xr, xb)
            opt.zero_grad()
            L.backward()
            opt.step()
            total += L.item() * len(xb)

        if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
            print(f"[{per_mode_name} | {arch}] ep{ep:03d} loss={total/len(ds):.6f}")

    return model


def score_windows_single(model: BaseConv1dAE, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xp = torch.from_numpy(X).to(device)
        xr = model(Xp)
        if CONFIG["AE_ARCH"]["loss"] == "l1":
            err = torch.mean(torch.abs(xr - Xp), dim=(1, 2))
        else:
            err = torch.mean((xr - Xp) ** 2, dim=(1, 2))
    return err.cpu().numpy()


def train_ensemble(X_train: np.ndarray,
                   arch: str,
                   per_mode_name: str,
                   num_models: int | None = None) -> List[BaseConv1dAE]:
    if num_models is None:
        num_models = CONFIG["ENSEMBLE"]["num_models"]

    models = []
    base_seed = CONFIG["SEED"]

    for i in range(num_models):
        this_seed = base_seed + 100 * i
        random.seed(this_seed)
        np.random.seed(this_seed)
        torch.manual_seed(this_seed)
        torch.cuda.manual_seed_all(this_seed)

        print(f"\n=== Training AE model {i+1}/{num_models} ({arch}, {per_mode_name}) ===")
        m = fit_ae_single(X_train, arch=arch, per_mode_name=per_mode_name)
        models.append(m)

    return models


def score_ensemble(models: list[BaseConv1dAE], X: np.ndarray) -> np.ndarray:
    all_scores = [score_windows_single(m, X) for m in models]
    return np.stack(all_scores, axis=0).mean(axis=0)
