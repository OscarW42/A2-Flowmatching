import torch
import matplotlib.pyplot as plt
from dataloader import ToyDiffusionDataset, get_dataloader
from model import Denoiser

dim = 2
model = Denoiser(dim)
model.load_state_dict(torch.load("denoiser.pt", map_location="cpu"))
model.eval()  # switch off dropout/batchnorm training behaviour



delta_t = -0.01
t = 1.0
z = torch.randn(1024, dim)
while t > 0.0:
    with torch.no_grad():
        v = model(z, torch.tensor([t]))
    z = z + v * delta_t
    t += delta_t

