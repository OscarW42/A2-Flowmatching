import torch
import torch.nn as nn
from pathlib import Path
import wandb

from meanflow_model import Denoiser
from dataloader import get_dataloader

# ── Training loop ─────────────────────────────────────────────────────────────
def train(
    dataset_name: str = "swiss_roll",
    # pred_quantity: str = "v",  # "x" or "v"
    # loss_type: str = "v", # "x" or "v"
    dim: int = 2,
    ratio_h: float = 0.5,
    n_steps: int = 25000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    log_every: int = 500,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Initialize Weights & Biases for logging
    wandb.init(
        project="flow-matching",
        config={
            "dataset": dataset_name,
            "dim": dim,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "lr": lr,
        }
    )

    # Data
    loader = get_dataloader(name=dataset_name, dim=dim,
                            batch_size=batch_size)
    data_iter = iter(loader)

    # Model, optimiser, loss
    model = Denoiser(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=0.1*lr)

    running_loss = 0.0

    model.train()
    for step in range(1, n_steps + 1):

        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x = next(data_iter)

        x = x.to(device)

        t = torch.rand(1).item()
        if torch.rand(1).item() < ratio_h:
            r = torch.rand(1).item()
            t, r = max(t, r), min(t, r)  # ensure t >= r
        else:
            r = t
        h = t-r

        t_tensor = torch.full((batch_size,), t, device=device)
        # r_tensor = torch.full((batch_size,), r, device=device)
        h_tensor = torch.full((batch_size,), h, device=device)

        t_expand = t_tensor.unsqueeze(-1)
        h_expand = h_tensor.unsqueeze(-1)

        epsilon = torch.randn_like(x)

        z_t = (1 - t_expand) * x + t_expand * epsilon
        v_t = epsilon - x
        u, dudt = torch.func.jvp(
            lambda z, t, h: model(z, t, h),
            (z_t, t_tensor, h_tensor),
            (v_t, torch.ones_like(t_tensor), torch.ones_like(h_tensor)),
        )

        with torch.no_grad():
            u_tgt = v_t - h_expand * dudt

        error = u - u_tgt
        loss = (error ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            wandb.log({
                "loss": avg_loss, 
                "lr": scheduler.get_last_lr()[0], 
                "step": step,
            })
            print(f"step {step:>6} / {n_steps} | loss {avg_loss:.4f}")
            running_loss = 0.0

    torch.save(model.state_dict(), "denoiser.pt")
    print("Model saved to denoiser.pt")

    return model

if __name__ == "__main__":
    model = train(dataset_name="swiss_roll", dim=2)

        # if pred_quantity == "v" and loss_type == "v":
        #     loss = mse(pred, epsilon - x)
        # elif pred_quantity == "v" and loss_type == "x":
        #     x_pred = z_t - t_expand * pred
        #     loss = mse(x_pred, x)
        # elif pred_quantity == "x" and loss_type == "x":
        #     loss = mse(pred, x)
        # elif pred_quantity == "x" and loss_type == "v":
        #     v_pred = (z_t - pred) / t_expand
        #     loss = mse(v_pred, epsilon - x)
        # else:
        #     raise ValueError(f"Unknown combination of pred_quantity '{pred_quantity}' and loss_type '{loss_type}'. Choose 'x' or 'v' for both.")