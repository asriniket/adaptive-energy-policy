from pathlib import Path

import torch

from trainers import (
    CFMTrainer,
    EnergyMatchingTrainer,
    EqMContrastiveTrainer,
    EqMStateTrainer,
    EqMTrainer,
)
from utils import (
    RobosuiteDataset,
    StateActionEnergyNetwork,
    StateActionVelocityNetwork,
    StateEnergyNetwork,
    plot_losses,
    set_seed,
)


def train_cfm():
    network = StateActionVelocityNetwork(
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
    plot_losses(info, save_path=results_dir / "cfm_losses.png")


def train_energy_matching():
    network = StateActionEnergyNetwork(
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
    plot_losses(info, save_path=results_dir / "energy_matching_phase1_losses.png")

    trainer.ema_decay = 0.99
    info = trainer.train(iterations=12_500)
    trainer.save_checkpoint(checkpoints_dir / "energy_matching_phase2.pt")
    plot_losses(info, save_path=results_dir / "energy_matching_phase2_losses.png")


def train_eqm():
    policy_network = StateActionEnergyNetwork(
        obs_dim=45,
        action_dim=7,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    policy_trainer = EqMTrainer(
        policy_network,
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
    policy_info = policy_trainer.train(iterations=100_000)
    policy_trainer.save_checkpoint(checkpoints_dir / "eqm_policy.pt")
    plot_losses(policy_info, save_path=results_dir / "eqm_policy_losses.png")

    state_network = StateEnergyNetwork(
        obs_dim=45,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    state_trainer = EqMStateTrainer(
        state_network,
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
    state_info = state_trainer.train(iterations=100_000)
    state_trainer.save_checkpoint(checkpoints_dir / "eqm_state.pt")
    plot_losses(state_info, save_path=results_dir / "eqm_state_losses.png")


def train_eqm_contrastive():
    network = StateActionEnergyNetwork(
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
    plot_losses(info, save_path=results_dir / "eqm_contrastive_losses.png")


if __name__ == "__main__":
    set_seed(0)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("Square")

    # train_cfm()
    # train_energy_matching()
    train_eqm()
    # train_eqm_contrastive()
