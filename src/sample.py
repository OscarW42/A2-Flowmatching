import torch
import matplotlib.pyplot as plt
from dataloader import ToyDiffusionDataset
from model import Denoiser

def sample(model, pred_quantity, dim=2, n_steps=50, batch_size=8192):
    delta_t = -1.0 / n_steps
    t = 1.0
    z = torch.randn(batch_size, dim)
    while t > 0.0:
        with torch.no_grad():
            pred = model(z, torch.full((batch_size,), t))
            if pred_quantity == "v":
                v = pred
            elif pred_quantity == "x":
                v = (z - pred) / t
            else:
                raise ValueError(f"Unknown pred_quantity '{pred_quantity}'. Choose 'x' or 'v'.")
        z = z + v * delta_t
        t += delta_t
    return z

if __name__ == "__main__":
    dim = 2
    model = Denoiser(dim)
    model.load_state_dict(torch.load("denoiser.pt", map_location="cpu"))
    model.eval()  # switch off dropout/batchnorm training behaviour
    
    z = sample(model, "x", dim=dim).numpy()
    ds = ToyDiffusionDataset(name="gaussians", dim=dim)
    if dim > 2:
        z = ds.to_2d(z)

    plt.scatter(z[:, 0], z[:, 1], s=1, alpha=0.5)
    plt.title("Sample")
    plt.show()