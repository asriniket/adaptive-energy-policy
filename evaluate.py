import json
from pathlib import Path

import imageio
import numpy as np
import robosuite as suite
import torch
from tqdm import tqdm

from trainers import (
    CFMTrainer,
    EnergyMatchingTrainer,
    EqMContrastiveTrainer,
    EqMTrainer,
)
from utils import (
    RobosuiteDataset,
    StateActionEnergyNetwork,
    StateActionVelocityNetwork,
    set_seed,
)


def evaluate(trainer, save_name, sample_kwargs):
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    ep_rewards, ep_successes = [], []
    for ep in range(num_eval_episodes):
        obs = env.reset()
        ep_reward = 0
        frames = []

        for _ in tqdm(range(num_episode_steps), desc=f"{save_name} Episode {ep + 1}"):
            actions = trainer.sample(env.sim.get_state().flatten(), **sample_kwargs)

            if action_selection == "min":
                obs_tensor = (
                    torch.tensor(
                        env.sim.get_state().flatten(),
                        dtype=torch.float32,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(actions.shape[0], -1)
                )
                energies = trainer.ema_network(obs_tensor, actions)
                action = actions[energies.argmin()].cpu().numpy()
            elif action_selection == "random":
                rand_idx = torch.randint(0, actions.shape[0], (1,)).item()
                action = actions[rand_idx].cpu().numpy()
            elif action_selection == "mean":
                action = actions.mean(dim=0).cpu().numpy()
            elif action_selection == "weighted_mean":
                obs_tensor = (
                    torch.tensor(
                        env.sim.get_state().flatten(),
                        dtype=torch.float32,
                        device=device,
                    )
                    .unsqueeze(0)
                    .expand(actions.shape[0], -1)
                )
                energies = trainer.ema_network(obs_tensor, actions)
                weights = torch.softmax(-energies, dim=0)
                action = (
                    (actions * weights.unsqueeze(-1)).sum(dim=0).detach().cpu().numpy()
                )
            else:
                raise ValueError(f"Unknown action_selection: {action_selection}")

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            frames.append(obs["agentview_image"][::-1])
            if done:
                break

        video_path = videos_dir / f"{save_name}_ep{ep + 1}.mp4"
        imageio.mimsave(video_path, frames, fps=20)
        print(
            f"{save_name} Episode {ep + 1}: reward={ep_reward:.1f}, saved to {video_path}"
        )
        ep_rewards.append(ep_reward)
        ep_successes.append(info.get("success", 0))

    mean_reward = np.mean(ep_rewards)
    mean_success = np.mean(ep_successes)
    print(f"{save_name} Average Reward: {mean_reward:.1f}")
    print(f"{save_name} Average Success: {mean_success:.1f}")
    results = {
        "episode_rewards": ep_rewards,
        "episode_successes": ep_successes,
        "mean_reward": float(mean_reward),
        "mean_success": float(mean_success),
        "num_episodes": num_eval_episodes,
        "action_selection": action_selection,
        "sample_kwargs": sample_kwargs,
    }

    results_path = results_dir / f"{save_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


def eval_cfm():
    network = StateActionVelocityNetwork(
        obs_dim=45, action_dim=7, hidden_dim=512, enc_output_dim=256
    )
    trainer = CFMTrainer(
        network,
        dataset,
        batch_size=256,
        lr=1e-4,
        ema_decay=0.9999,
        num_sampling_steps=250,
        device=device,
    )
    trainer.load_checkpoint(checkpoints_dir / "cfm.pt")
    evaluate(trainer, "cfm", sample_kwargs={"num_samples": 64})


def eval_energy_matching():
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
    trainer.load_checkpoint(checkpoints_dir / "energy_matching_phase2.pt")
    evaluate(
        trainer,
        "energy_matching_phase2",
        sample_kwargs={"tau_s": 3.25, "num_samples": 64},
    )


def eval_eqm():
    network = StateActionEnergyNetwork(
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
    trainer.load_checkpoint(checkpoints_dir / "eqm_policy.pt")
    evaluate(trainer, "eqm", sample_kwargs={"num_samples": 64})


def eval_eqm_contrastive():
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
    trainer.load_checkpoint(checkpoints_dir / "eqm_contrastive.pt")
    evaluate(trainer, "eqm_contrastive", sample_kwargs={"num_samples": 64})


if __name__ == "__main__":
    set_seed(0)

    checkpoints_dir = Path("checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("Square")
    env = suite.make(env_name="NutAssemblySquare", robots="Panda")

    num_eval_episodes = 10
    num_episode_steps = 200
    action_selection = "mean"

    # eval_cfm()
    # eval_energy_matching()
    eval_eqm()
    # eval_eqm_contrastive()

    env.close()
