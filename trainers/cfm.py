import torch
import torch.nn.functional as F
from tqdm import tqdm

from .trainer import Trainer


class CFMTrainer(Trainer):
    def __init__(
        self,
        network,
        dataset,
        *,
        batch_size,
        lr,
        ema_decay,
        num_sampling_steps,
        device,
    ):
        super().__init__(
            network,
            torch.optim.Adam,
            dataset,
            batch_size=batch_size,
            lr=lr,
            ema_decay=ema_decay,
            device=device,
        )
        self.num_sampling_steps = num_sampling_steps

    def compute_cfm_loss(self, obs, actions):
        B, action_dim = actions.shape

        t = torch.rand(B, device=self.device)
        noise = torch.randn(B, action_dim, device=self.device)
        actions_t = (1 - t.unsqueeze(-1)) * noise + t.unsqueeze(-1) * actions

        target_velocity = actions - noise
        predicted_velocity = self.network(obs, actions_t, t)
        return F.mse_loss(predicted_velocity, target_velocity)

    def train(self, iterations):
        self.network.train()
        data_iter = iter(self.data_loader)
        info = {"total_loss": []}

        pbar = tqdm(range(iterations), desc="Training CFM")
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)

            obs = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)

            self.optimizer.zero_grad()
            loss = self.compute_cfm_loss(obs, actions)
            loss.backward()
            self.optimizer.step()
            self.update_ema()

            info["total_loss"].append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        return info

    @torch.no_grad()
    def sample(self, obs, num_samples=1):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0).repeat(num_samples, 1)

        B = obs.shape[0]
        actions = torch.randn(B, self.action_dim, device=self.device)
        dt = 1.0 / self.num_sampling_steps
        for step in range(self.num_sampling_steps):
            t = torch.full((B,), step / self.num_sampling_steps, device=self.device)
            velocity = self.ema_network(obs, actions, t)
            actions = actions + velocity * dt
        return actions
