import torch
import torch.nn as nn

from .config import CONFIG


class BaseConv1dAE(nn.Module):
    """
    arch:
      - "plain":    no skip
      - "residual": decoder output + input skip
      - "tcn":      dilated convs (TCN-style)
    """
    def __init__(self,
                 in_dim: int,
                 hidden: int | None = None,
                 bottleneck: int | None = None,
                 arch: str = "residual",
                 loss: str | None = None,
                 dropout: float | None = None):
        super().__init__()
        self.arch = arch
        self.loss_type = loss or CONFIG["AE_ARCH"]["loss"]

        hidden = hidden if hidden is not None else CONFIG["AE_ARCH"]["hidden"]
        bottleneck = bottleneck if bottleneck is not None else CONFIG["AE_ARCH"]["bottleneck"]
        if dropout is None:
            dropout = CONFIG["AE_ARCH"].get("dropout", 0.0)

        if arch == "tcn":
            self.enc = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, bottleneck, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.dec = nn.Sequential(
                nn.Conv1d(bottleneck, hidden, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, in_dim, kernel_size=3, padding=1, dilation=1),
            )
        else:
            self.enc = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, bottleneck, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.dec = nn.Sequential(
                nn.Conv1d(bottleneck, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, in_dim, kernel_size=5, padding=2),
            )

        if self.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T) for Conv1d
        x_in = x.transpose(1, 2)
        z = self.enc(x_in)
        xr = self.dec(z)

        if self.arch == "residual":
            if xr.shape == x_in.shape:
                xr = xr + x_in
            else:
                proj = nn.Conv1d(x_in.shape[1], xr.shape[1], kernel_size=1).to(x_in.device)
                xr = xr + proj(x_in)

        return xr.transpose(1, 2)

    def loss(self, xr, x):
        return self.loss_fn(xr, x)
