import torch
import matplotlib.pyplot as plt
from dataloader import ToyDiffusionDataset, get_dataloader
from model import Denoiser

def sample(model, n_steps, dim=2, batch_size=1024):
    delta_t = -1.0 / n_steps
    t = 1.0
    z = torch.randn(batch_size, dim)
    while t > 0.0:
        with torch.no_grad():
            v = model(z, torch.full((batch_size,), t))
        z = z + v * delta_t
        t += delta_t
    return z

if __name__ == "__main__":
    dim = 2
    model = Denoiser(dim)
    model.load_state_dict(torch.load("denoiser.pt", map_location="cpu"))
    model.eval()  # switch off dropout/batchnorm training behaviour
    
    z = sample(model, 50, dim=dim, batch_size=8192).numpy()
    
    plt.scatter(z[:, 0], z[:, 1], s=1, alpha=0.5)
    plt.title("Sample")
    plt.show()