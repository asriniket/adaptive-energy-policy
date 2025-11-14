import copy
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def mlp(input_dim, hidden_dims, output_dim, output_activation=None, dropout=0.0):
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]

    dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Mish())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        elif output_activation:
            layers.append(output_activation)
    return nn.Sequential(*layers)


class RobosuiteDataset(torch.utils.data.Dataset):
    def __init__(self, task):
        demos = self.load_robosuite_demo(task)
        self.obs = []
        for demo in demos:
            for i in range(len(demo["obs"])):
                self.obs.append(demo["obs"][i])

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx]

    @staticmethod
    def load_robosuite_demo(task):
        demo_path = Path(f"datasets/robosuite/{task}.hdf5")
        data = []
        with h5py.File(demo_path, "r") as f:
            data_group = f["data"]
            for demo_id in data_group.keys():
                demo = data_group[demo_id]
                data.append(RobosuiteDataset._process_robosuite_demo(demo))
        return data

    @staticmethod
    def _process_robosuite_demo(demo):
        obs = torch.from_numpy(demo["states"][()]).float()
        actions = torch.from_numpy(demo["actions"][()]).float()
        rewards = torch.from_numpy(demo["rewards"][()]).float()
        dones = torch.from_numpy(demo["dones"][()]).float()
        return {"obs": obs, "action": actions, "reward": rewards, "terminated": dones}


class EnergyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, enc_output_dim, output_scale=1.0):
        super().__init__()
        self.encoder = mlp(input_dim, [hidden_dim] * 4, enc_output_dim)
        self.energy = mlp(enc_output_dim, [hidden_dim] * 2, 1)
        self.output_scale = output_scale

    def forward(self, obs):
        z = self.encoder(obs)
        energy = self.energy(z)
        return energy.squeeze(-1) * self.output_scale

    def velocity(self, x):
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            energy = self.forward(x)
            grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
            return -grad


class Trainer(ABC):
    def __init__(
        self,
        network,
        optimizer_cls,
        dataset,
        *,
        batch_size,
        lr,
        ema_decay,
        device,
    ):
        self.network = network.to(device)
        self.ema_network = copy.deepcopy(network).to(device)
        self.ema_network.eval()
        self.ema_decay = ema_decay

        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        self.optimizer = optimizer_cls(network.parameters(), lr=lr)
        self.device = device

        sample = dataset[0]
        self.obs_dim = sample.shape[0]

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_network.parameters(), self.network.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )

    def save_checkpoint(self, path):
        checkpoint = {
            "network_state_dict": self.network.state_dict(),
            "ema_network_state_dict": self.ema_network.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.ema_network.load_state_dict(checkpoint["ema_network_state_dict"])
        print(f"Checkpoint loaded from {path}")

    @abstractmethod
    def train(self, iterations):
        pass

    @abstractmethod
    def sample(self, obs, **kwargs):
        pass


class EqMTrainer(Trainer):
    def __init__(
        self,
        network,
        dataset,
        *,
        batch_size,
        lr,
        ema_decay,
        decay_type,
        decay_a,
        decay_b,
        gradient_multiplier,
        num_sampling_steps,
        sampling_step_size,
        device,
    ):
        super().__init__(
            network,
            torch.optim.AdamW,
            dataset,
            batch_size=batch_size,
            lr=lr,
            ema_decay=ema_decay,
            device=device,
        )

        self.decay_type = decay_type
        self.decay_a = decay_a
        self.decay_b = decay_b
        self.gradient_multiplier = gradient_multiplier
        self.num_sampling_steps = num_sampling_steps
        self.sampling_step_size = sampling_step_size

    def c_decay(self, gamma):
        if self.decay_type == "linear":
            return (1 - gamma) * self.gradient_multiplier
        elif self.decay_type == "truncated":
            c = torch.where(
                gamma <= self.decay_a,
                torch.ones_like(gamma),
                (1 - gamma) / (1 - self.decay_a),
            )
            return c * self.gradient_multiplier
        elif self.decay_type == "piecewise":
            c = torch.where(
                gamma <= self.decay_a,
                self.decay_b - ((self.decay_b - 1) / self.decay_a) * gamma,
                (1 - gamma) / (1 - self.decay_a),
            )
            return c * self.gradient_multiplier
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

    def compute_eqm_loss(self, obs):
        B, obs_dim = obs.shape

        gamma = torch.rand(B, 1, device=self.device)
        noise = torch.randn(B, obs_dim, device=self.device)

        interpolated = gamma * obs + (1 - gamma) * noise
        c_gamma = self.c_decay(gamma)

        target = (noise - obs) * c_gamma
        predicted = -self.network.velocity(interpolated)
        return F.mse_loss(predicted, target)

    def train(self, iterations):
        self.network.train()
        data_iter = iter(self.data_loader)
        info = {"total_loss": []}

        pbar = tqdm(range(iterations), desc="Training EqM")
        for _ in pbar:
            try:
                obs = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                obs = next(data_iter)
            obs = obs.to(self.device)
            self.optimizer.zero_grad()
            loss = self.compute_eqm_loss(obs)
            loss.backward()
            self.optimizer.step()
            self.update_ema()

            info["total_loss"].append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return info

    @torch.no_grad()
    def sample(self, num_samples=1):
        samples = torch.randn(num_samples, self.obs_dim, device=self.device)
        for _ in range(self.num_sampling_steps):
            grad = -self.ema_network.velocity(samples)
            samples = samples - self.sampling_step_size * grad
        return samples


def train():
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("Square")
    network = EnergyNetwork(
        input_dim=45,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    trainer = EqMTrainer(
        network,
        dataset,
        batch_size=256,
        lr=1e-4,
        ema_decay=0.9999,
        decay_type="truncated",
        decay_a=0.8,
        decay_b=1.0,
        gradient_multiplier=4.0,
        num_sampling_steps=250,
        sampling_step_size=0.003,
        device=device,
    )
    trainer.train(iterations=100_000)
    trainer.save_checkpoint(checkpoints_dir / "eqm_state.pt")


if __name__ == "__main__":
    train()
