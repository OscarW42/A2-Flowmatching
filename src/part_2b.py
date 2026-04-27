import matplotlib.pyplot as plt

from dataloader import ToyDiffusionDataset
from train import train
from sample import sample

for dim in [2, 8, 32]:
    for ds_name in ["swiss_roll", "gaussians", "circles"]:
        for pred_quantity in ["x", "v"]:
            for loss_type in ["x", "v"]:
                print(f"Training on {ds_name} with dim={dim}, pred_quantity={pred_quantity}, loss_type={loss_type}")
                ds = ToyDiffusionDataset(name=ds_name, dim=dim)
                model = train(dataset_name=ds_name, dim=dim, pred_quantity=pred_quantity, loss_type=loss_type)
                print(f"Sampling on {ds_name} with dim={dim}, pred_quantity={pred_quantity}, loss_type={loss_type}")
                z = sample(model, pred_quantity, dim=dim).numpy()
                z_2d = ds.to_2d(z)
                plt.scatter(z_2d[:, 0], z_2d[:, 1], s=1, alpha=0.5)
                plt.title(f"{ds_name} | {dim}D | {pred_quantity}-pred | {loss_type}-loss")
                plt.savefig(f"results/{ds_name}_{dim}D_{pred_quantity}pred_{loss_type}loss.png")