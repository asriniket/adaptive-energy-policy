from pathlib import Path

import h5py
import numpy as np
import robosuite as suite
import torch
from tqdm import tqdm

from utils import EnergyNetwork, EqMTrainer, RobosuiteDataset


def collect_rollouts():
    success_obs, success_actions = [], []
    failure_obs, failure_actions = [], []
    num_successes = 0
    num_failures = 0

    pbar = tqdm(range(num_episodes), desc="Collecting rollouts")
    for _ in pbar:
        env.reset()
        ep_data = []
        ep_reward = 0.0

        for _ in range(num_episode_steps):
            state = env.sim.get_state().flatten()
            actions = trainer.sample_actions(state, num_samples=64)
            action = actions.mean(dim=0).cpu().numpy()

            ep_data.append((state, action))
            _, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break

        if ep_reward > 0.0:
            num_successes += 1
            for state, action in ep_data:
                success_obs.append(state)
                success_actions.append(action)
        else:
            num_failures += 1
            for state, action in ep_data:
                failure_obs.append(state)
                failure_actions.append(action)

        pbar.set_postfix({"success": num_successes, "failure": num_failures})
    env.close()

    save_path = Path(f"datasets/robosuite/Square_rollouts.hdf5")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(save_path, "w") as f:
        if success_obs:
            f.create_dataset(
                "success/obs", data=np.array(success_obs, dtype=np.float32)
            )
            f.create_dataset(
                "success/actions", data=np.array(success_actions, dtype=np.float32)
            )
        if failure_obs:
            f.create_dataset(
                "failure/obs", data=np.array(failure_obs, dtype=np.float32)
            )
            f.create_dataset(
                "failure/actions", data=np.array(failure_actions, dtype=np.float32)
            )

    print(f"\nResults:")
    print(
        f"\tSuccess rate: {num_successes}/{num_episodes} ({100 * num_successes / num_episodes:.1f}%)"
    )
    print(f"\tSuccessful transitions: {len(success_actions)}")
    print(f"\tFailed transitions: {len(failure_actions)}")
    print(f"\tSaved to: {save_path}")
    return save_path


if __name__ == "__main__":
    checkpoints_dir = Path("checkpoints")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("Square")
    env = suite.make(env_name="NutAssemblySquare", robots="Panda", hard_reset=False)

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
        num_sampling_steps=100,
        sampling_step_size=0.003,
        device=device,
    )
    trainer.load_checkpoint(checkpoints_dir / "eqm.pt")

    num_episodes = 200
    num_episode_steps = 200
    collect_rollouts()
