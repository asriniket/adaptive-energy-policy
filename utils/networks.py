import math

import torch
import torch.nn as nn


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


class EnergyNetwork(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim, enc_output_dim, output_scale=1.0
    ):
        super().__init__()
        self.encoder = mlp(obs_dim, [hidden_dim] * 4, enc_output_dim)
        self.energy = mlp(enc_output_dim + action_dim, [hidden_dim] * 2, 1)
        self.output_scale = output_scale

    def forward(self, obs, action):
        z = self.encoder(obs)
        energy = self.energy(torch.cat([z, action], dim=-1))
        return energy.squeeze(-1) * self.output_scale

    def velocity(self, obs, action):
        with torch.enable_grad():
            action = action.detach().clone().requires_grad_(True)
            energy = self.forward(obs.detach(), action)
            grad = torch.autograd.grad(energy.sum(), action, create_graph=True)[0]
            return -grad


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device)
            / half_dim
        )
        args = t * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class VelocityNetwork(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim, enc_output_dim, time_embed_dim=256
    ):
        super().__init__()
        self.encoder = mlp(obs_dim, [hidden_dim] * 4, enc_output_dim)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.velocity = mlp(
            enc_output_dim + action_dim + time_embed_dim, [hidden_dim] * 2, action_dim
        )

    def forward(self, obs, action, t):
        obs_enc = self.encoder(obs)
        t_emb = self.time_embed(t)

        x = torch.cat([obs_enc, action, t_emb], dim=-1)
        velocity = self.velocity(x)
        return velocity
