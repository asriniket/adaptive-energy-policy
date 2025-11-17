import torch
import torch.nn.functional as F
from tqdm import tqdm

from .trainer import Trainer


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

    def compute_eqm_loss(self, obs, actions):
        B, action_dim = actions.shape

        gamma = torch.rand(B, 1, device=self.device)
        noise = torch.randn(B, action_dim, device=self.device)

        c_actions = gamma * actions + (1 - gamma) * noise
        c_gamma = self.c_decay(gamma)

        target = (noise - actions) * c_gamma
        predicted = -self.network.velocity(obs, c_actions)
        return F.mse_loss(predicted, target)

    def train(self, iterations):
        self.network.train()
        data_iter = iter(self.data_loader)
        info = {"total_loss": []}

        pbar = tqdm(range(iterations), desc="Training EqM")
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)

            obs = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)

            self.optimizer.zero_grad()
            loss = self.compute_eqm_loss(obs, actions)
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
        for _ in range(self.num_sampling_steps):
            grad = -self.ema_network.velocity(obs, actions)
            actions = actions - self.sampling_step_size * grad
        return actions


class EqMStateTrainer(Trainer):
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

        pbar = tqdm(range(iterations), desc="Training EqM State")
        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.data_loader)
                batch = next(data_iter)
            obs = batch["obs"].to(self.device)
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
