"""
Dataloader for swiss_roll and gaussians projected to various dims.
NO normalization — raw projected data.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AVAILABLE_DATASETS = ("swiss_roll", "gaussians", "circles")
AVAILABLE_DIMS = (2, 8, 32)


class ToyDiffusionDataset(Dataset):
    def __init__(self, name: str = "swiss_roll", dim: int = 2,
                 data_dir: Path | None = None):
        if name not in AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset '{name}'. Choose from {AVAILABLE_DATASETS}")
        if dim not in AVAILABLE_DIMS:
            raise ValueError(f"dim={dim} not available. Choose from {AVAILABLE_DIMS}")

        root = Path(data_dir) if data_dir else DATA_DIR
        npz = np.load(root / f"{name}.npz")

        self.data = torch.from_numpy(npz[f"{dim}d"])
        self.dim = dim
        self.name = name
        self.P = npz[f"P_{dim}"] if dim > 2 else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def to_2d(self, samples: np.ndarray) -> np.ndarray:
        if self.P is not None:
            return samples @ self.P.T
        return samples


def get_dataloader(name: str = "swiss_roll", dim: int = 2, batch_size: int = 1024,
                   shuffle: bool = True, data_dir: Path | None = None) -> DataLoader:
    ds = ToyDiffusionDataset(name=name, dim=dim, data_dir=data_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

import matplotlib.pyplot as plt

dl = get_dataloader(name="swiss_roll", dim=2, batch_size=8192)
points = next(iter(dl))  # shape: (8192, 2)

points_2d = points.numpy()  # convert to numpy for matplotlib

# ds = ToyDiffusionDataset(name="swiss_roll", dim=32)
# loader = DataLoader(ds, batch_size=8192, shuffle=False)

# points_32d = next(iter(loader)).numpy()  # (8192, 32)
# points_2d = ds.to_2d(points_32d)         # (8192, 2) via P.T

plt.scatter(points_2d[:, 0], points_2d[:, 1], s=1, alpha=0.5)
plt.title("Swiss Roll 2d")
plt.show()