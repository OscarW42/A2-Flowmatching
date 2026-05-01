import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d: int = 128):
        super().__init__()
        self.d = d
        k = d // 2
        # ωi = exp(-i · ln(10000) / (k-1)), i = 0, ..., k-1
        i = torch.arange(k, dtype=torch.float32)
        omegas = torch.exp(-i * math.log(10000) / (k - 1))
        self.register_buffer("omegas", omegas)  # shape: (k,)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) scalar values in [0, 1]
        t = t.unsqueeze(-1)                    # (B, 1)
        angles = t * self.omegas               # (B, k)
        et = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, d)
        return et


class Denoiser(nn.Module):
    def __init__(self, D: int, hidden: int = 256):
        """
        Args:
            D:        data dimensionality (input/output size)
            hidden:   hidden layer width (default 256)
        """
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(d=128)
        self.h_embed = SinusoidalTimeEmbedding(d=128)

        # 5 hidden layers (Linear → ReLU) + 1 final linear, 6 total
        self.net = nn.Sequential(
            # Layer 1: (D + 256) → 256
            nn.Linear(D + 256, hidden), nn.ReLU(),
            # Layers 2-5: 256 → 256
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            # Layer 6: 256 → D  (no activation)
            nn.Linear(hidden, D),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) noisy input
            t: (B,)   continuous time in [0, 1]
            h: (B,)   continuous horizon in [0, 1]
        Returns:
            (B, D) predicted noise or velocity, depending on training objective
        """
        et = self.time_embed(t)               # (B, 128)
        eh = self.h_embed(h)                  # (B, 128)
        x  = torch.cat([z, et, eh], dim=-1)   # (B, D+256)
        return self.net(x)                    # (B, D)


# --- quick sanity check ---
if __name__ == "__main__":
    B, D = 4, 16
    model = Denoiser(D=D)
    z = torch.randn(B, D)
    t = torch.rand(B)
    h = torch.rand(B)
    out = model(z, t, h)
    print(f"Input shape : {z.shape}")   # (4, 16)
    print(f"Output shape: {out.shape}") # (4, 16)
    print(f"\nModel architecture:\n{model}")