from pathlib import Path

import torch

from utils import (
    CFMTrainer,
    EnergyMatchingTrainer,
    EnergyNetwork,
    EqMContrastiveTrainer,
    EqMTrainer,
    RobosuiteDataset,
    VelocityNetwork,
    plot_losses,
)


def train_cfm():
    network = VelocityNetwork(
        obs_dim=45, action_dim=7, hidden_dim=512, enc_output_dim=256
    )
    trainer = CFMTrainer(
        network,
        dataset,
        batch_size=256,
        lr=1e-4,
        ema_decay=0.9999,
        num_sampling_steps=200,
        device=device,
    )
    info = trainer.train(iterations=100_000)
    trainer.save_checkpoint(checkpoints_dir / "cfm.pt")
    plot_losses(info, save_path=checkpoints_dir / "cfm_losses.png")


def train_energy_matching():
    network = EnergyNetwork(
        obs_dim=45,
        action_dim=7,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    trainer = EnergyMatchingTrainer(
        network,
        dataset,
        batch_size=128,
        lr=1.2e-3,
        ema_decay=0.9999,
        tau_star=1.0,
        num_langevin_steps=200,
        dt=0.01,
        eps_max=0.01,
        lambda_cd=1e-3,
        device=device,
    )

    info = trainer.pretrain(iterations=87_500)
    trainer.save_checkpoint(checkpoints_dir / "energy_matching_phase1.pt")
    plot_losses(info, save_path=checkpoints_dir / "energy_matching_phase1_losses.png")

    trainer.ema_decay = 0.99
    info = trainer.train(iterations=12_500)
    trainer.save_checkpoint(checkpoints_dir / "energy_matching_phase2.pt")
    plot_losses(info, save_path=checkpoints_dir / "energy_matching_phase2_losses.png")


def train_eqm():
    network = EnergyNetwork(
        obs_dim=45,
        action_dim=7,
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
    info = trainer.train(iterations=100_000)
    trainer.save_checkpoint(checkpoints_dir / "eqm.pt")
    plot_losses(info, save_path=checkpoints_dir / "eqm_losses.png")


def train_eqm_contrastive():
    network = EnergyNetwork(
        obs_dim=45,
        action_dim=7,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    trainer = EqMContrastiveTrainer(
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
        num_langevin_steps=200,
        dt=0.01,
        eps_max=0.01,
        lambda_cd=1e-3,
        device=device,
    )
    info = trainer.train(iterations=100_000)
    trainer.save_checkpoint(checkpoints_dir / "eqm_contrastive.pt")
    plot_losses(info, save_path=checkpoints_dir / "eqm_contrastive_losses.png")


if __name__ == "__main__":
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("Square")

    train_cfm()
    train_energy_matching()
    train_eqm()
    train_eqm_contrastive()
