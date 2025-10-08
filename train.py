import torch

from utils import EnergyNetwork, FlowMatchingEnergyTrainer, RobosuiteDataset


def main():
    dataset = RobosuiteDataset("Lift")
    network = EnergyNetwork(
        obs_dim=32,
        action_dim=7,
        hidden_dim=256,
        enc_output_dim=128,
        output_scale=1000.0,
    )
    trainer = FlowMatchingEnergyTrainer(
        network,
        dataset,
        batch_size=128,
        lr=1.2e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ema_decay=0.9999,
        tau_star=1.0,
        num_langevin_steps=200,
        dt=0.01,
        eps_max=0.01,
        lambda_cd=1e-3,
    )

    trainer.pretrain(iterations=145000)
    trainer.save_checkpoint("phase1.pt")

    trainer.ema_decay = 0.99
    trainer.train(iterations=2000)
    trainer.save_checkpoint("phase2.pt")


if __name__ == "__main__":
    main()
