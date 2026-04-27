import torch
import torch.nn as nn
from pathlib import Path

from model import Denoiser
from dataloader import get_dataloader

# ── Training loop ─────────────────────────────────────────────────────────────
def train(
    dataset_name: str = "swiss_roll",
    dim: int = 2,
    # data_dir: Path = Path("data"),
    n_steps: int = 25000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    log_every: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Data
    loader = get_dataloader(name=dataset_name, dim=dim,
                            batch_size=batch_size)
    data_iter = iter(loader)

    # Model, optimiser, loss
    model = Denoiser(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    model.train()
    for step in range(1, n_steps + 1):

        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)

        x = x.to(device)

        t = torch.rand(1).item()
        t_tensor = torch.full((batch_size,), t).to(device)
        t_expand = t_tensor.unsqueeze(-1) 

        epsilon = torch.randn_like(x)

        z_t = (1 - t_expand) * x + t_expand * epsilon

        v_pred = model(z_t, t_tensor)
        loss = mse(v_pred, epsilon - x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"step {step:>6} / {n_steps} | loss {loss.item():.4f}")

    torch.save(model.state_dict(), "denoiser.pt")
    print("Model saved to denoiser.pt")

    return model

if __name__ == "__main__":
    model = train(dataset_name="circles", dim=2)