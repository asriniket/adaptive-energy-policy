from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch

from utils import EnergyNetwork, GaussianMixtureDataset


def visualize_ot_flow(num_samples=100, grid_size=100, padding_factor=0.5):
    obs = dataset[0]["obs"].unsqueeze(0).to(device)
    data = torch.stack([dataset[i]["action"] for i in range(num_samples)]).to(device)
    noise = torch.randn_like(data)

    # Compute OT matching
    M = torch.cdist(noise, data, p=2).pow(2).cpu().numpy()
    coupling = ot.emd([], [], M)
    matching = torch.from_numpy(coupling).argmax(dim=0).to(device)
    matched_noise = noise[matching]

    # Compute energy landscape
    points = torch.cat([matched_noise, data]).cpu().numpy()
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    x = np.linspace(
        points[:, 0].min() - padding_factor * x_range,
        points[:, 0].max() + padding_factor * x_range,
        grid_size,
    )
    y = np.linspace(
        points[:, 1].min() - padding_factor * y_range,
        points[:, 1].max() + padding_factor * y_range,
        grid_size,
    )
    X, Y = np.meshgrid(x, y)

    grid = torch.from_numpy(np.stack([X.flatten(), Y.flatten()], 1)).float().to(device)
    with torch.no_grad():
        energy = (
            network(obs.expand(len(grid), -1), grid)
            .cpu()
            .numpy()
            .reshape(grid_size, grid_size)
        )

    # Generate plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, energy, levels=50)
    plt.colorbar(label="Energy")

    noise_np = matched_noise.cpu().numpy()
    data_np = data.cpu().numpy()
    for i in range(len(noise_np)):
        plt.arrow(
            noise_np[i, 0],
            noise_np[i, 1],
            (data_np[i, 0] - noise_np[i, 0]),
            (data_np[i, 1] - noise_np[i, 1]),
            color="white",
            alpha=0.5,
        )

    plt.scatter(
        noise_np[:, 0], noise_np[:, 1], c="white", edgecolors="black", label="Noise"
    )
    plt.scatter(
        data_np[:, 0], data_np[:, 1], c="black", edgecolors="black", label="Data"
    )
    plt.legend()
    plt.title("OT Flow + Energy Landscape")
    plt.savefig("ot_flow.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    checkpoint_path = Path("checkpoints") / "energy_matching_phase2.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = GaussianMixtureDataset()
    network = EnergyNetwork(
        obs_dim=2,
        action_dim=2,
        hidden_dim=256,
        enc_output_dim=128,
        output_scale=1000.0,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint["network_state_dict"])
    network = network.to(device)
    network.eval()

    visualize_ot_flow(num_samples=100)
