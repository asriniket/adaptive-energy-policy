import math

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .trainer import Trainer


class EqMContrastiveTrainer(Trainer):
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
        num_langevin_steps,
        dt,
        eps_max,
        lambda_cd,
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

        self.num_langevin_steps = num_langevin_steps
        self.dt = dt
        self.eps_max = eps_max
        self.lambda_cd = lambda_cd

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

    def compute_eqm_loss(self, obs, actions):
        B, action_dim = actions.shape

        gamma = torch.rand(B, 1, device=self.device)
        noise = torch.randn(B, action_dim, device=self.device)

        c_actions = gamma * actions + (1 - gamma) * noise
        c_gamma = self.c_decay(gamma)

        target = (noise - actions) * c_gamma
        predicted = -self.network.velocity(obs, c_actions)
        return F.mse_loss(predicted, target)

    def langevin_dynamics(self, obs, init_actions=None):
        B = obs.shape[0]
        obs = obs.detach()

        if init_actions is not None:
            actions = init_actions.clone()
            use_constant_temp = True
        else:
            actions = torch.randn(B, self.action_dim, device=self.device)
            use_constant_temp = False

        for step in range(self.num_langevin_steps):
            actions = actions.requires_grad_(True)
            energy = self.network(obs, actions)
            grad = torch.autograd.grad(energy.sum(), actions)[0]

            if use_constant_temp:
                epsilon_t = self.eps_max
            else:
                epsilon_t = self.eps_max * (step + 1) / self.num_langevin_steps

            with torch.no_grad():
                noise = torch.randn_like(actions)
                actions = (
                    actions
                    - self.dt * grad
                    + math.sqrt(2 * self.dt * epsilon_t) * noise
                )
        return actions.detach()

    def compute_contrastive_loss(self, obs, actions):
        B = obs.shape[0]
        half_batch = B // 2

        neg_actions_data = self.langevin_dynamics(
            obs[:half_batch], init_actions=actions[:half_batch]
        )
        neg_actions_noise = self.langevin_dynamics(obs[half_batch:], init_actions=None)
        neg_actions = torch.cat([neg_actions_data, neg_actions_noise], dim=0)

        pos_energy = self.network(obs, actions) / self.eps_max
        neg_energy = self.network(obs, neg_actions) / self.eps_max
        loss_cd = pos_energy.mean() - neg_energy.mean()
        return loss_cd

    def train(self, iterations):
        self.network.train()
        data_iter = iter(self.data_loader)
        info = {"total_loss": [], "eqm_loss": [], "cd_loss": []}

        pbar = tqdm(range(iterations), desc="Training EqM Contrastive")
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)

            obs = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)

            self.optimizer.zero_grad()

            loss_eqm = self.compute_eqm_loss(obs, actions)
            loss_cd = self.compute_contrastive_loss(obs, actions)
            loss = loss_eqm + self.lambda_cd * loss_cd

            loss.backward()
            self.optimizer.step()
            self.update_ema()

            info["total_loss"].append(loss.item())
            info["eqm_loss"].append(loss_eqm.item())
            info["cd_loss"].append(loss_cd.item())
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
        for _ in range(self.num_sampling_steps):
            grad = -self.ema_network.velocity(obs, actions)
            actions = actions - self.sampling_step_size * grad
        return actions
