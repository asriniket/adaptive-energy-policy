from pathlib import Path

import torch

from utils import (
    EnergyMatchingTrainer,
    EnergyNetwork,
    EqMContrastiveTrainer,
    EqMTrainer,
    RobosuiteDataset,
    plot_losses,
)


def train_energy_matching():
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

    info = trainer.pretrain(iterations=10000)
    trainer.save_checkpoint(checkpoints_dir / "energy_matching_phase1.pt")
    plot_losses(info, save_path=checkpoints_dir / "energy_matching_phase1_losses.png")

    trainer.ema_decay = 0.99
    info = trainer.train(iterations=2500)
    trainer.save_checkpoint(checkpoints_dir / "energy_matching_phase2.pt")
    plot_losses(info, save_path=checkpoints_dir / "energy_matching_phase2_losses.png")


def train_eqm_contrastive():
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
    info = trainer.train(iterations=12500)
    trainer.save_checkpoint(checkpoints_dir / "eqm_contrastive.pt")
    plot_losses(info, save_path=checkpoints_dir / "eqm_contrastive_losses.png")


def train_eqm():
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
    info = trainer.train(iterations=100000)
    trainer.save_checkpoint(checkpoints_dir / "eqm.pt")
    plot_losses(info, save_path=checkpoints_dir / "eqm_losses.png")


if __name__ == "__main__":
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("PickPlaceCan")
    network = EnergyNetwork(
        obs_dim=71,
        action_dim=7,
        hidden_dim=256,
        enc_output_dim=128,
        output_scale=1000.0,
    )

    # train_energy_matching()
    # train_eqm_contrastive()
    train_eqm()
