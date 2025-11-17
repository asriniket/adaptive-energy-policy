import copy
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


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
        self.obs_dim = sample["obs"].shape[0]
        self.action_dim = sample["action"].shape[0]

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
