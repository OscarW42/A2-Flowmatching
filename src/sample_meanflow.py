import torch
import matplotlib.pyplot as plt
from dataloader import ToyDiffusionDataset
from meanflow_model import Denoiser

def sample(model, dim=2, n_steps=2, batch_size=8192):
    h = 1.0 / n_steps
    z = torch.randn(batch_size, dim)
    
    for i in range(n_steps):
        t = 1.0 - i * h

        t_tensor = torch.full((batch_size,), t)
        h_tensor = torch.full((batch_size,), h)

        with torch.no_grad():
            u = model(z, t_tensor, h_tensor)
        
        z = z - h * u
    
    return z

if __name__ == "__main__":
    dim = 2
    model = Denoiser(dim)
    model.load_state_dict(torch.load("denoiser.pt", map_location="cpu"))
    model.eval()  # switch off dropout/batchnorm training behaviour
    
    z = sample(model, dim=dim, n_steps=5).numpy()
    ds = ToyDiffusionDataset(name="swiss_roll", dim=dim)
    if dim > 2:
        z = ds.to_2d(z)

    plt.scatter(z[:, 0], z[:, 1], s=1, alpha=0.5)
    plt.title("Sample")
    plt.show()