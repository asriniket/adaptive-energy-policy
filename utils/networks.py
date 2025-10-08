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
        self.encoder = mlp(obs_dim, hidden_dim, enc_output_dim)
        self.energy = mlp(enc_output_dim + action_dim, [hidden_dim, hidden_dim], 1)
        self.output_scale = output_scale

    def forward(self, obs, action):
        z = self.encoder(obs)
        energy = self.energy(torch.cat([z, action], dim=-1))
        return energy.squeeze(-1) * self.output_scale

    def velocity(self, obs, action):
        action = action.clone().requires_grad_(True)
        energy = self.forward(obs.detach(), action)
        grad = torch.autograd.grad(energy.sum(), action, create_graph=True)[0]
        return -grad
