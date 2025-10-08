from pathlib import Path

import imageio
import numpy as np
import robosuite as suite
import torch
from tqdm import tqdm

from utils import EnergyNetwork, FlowMatchingEnergyTrainer, RobosuiteDataset


def evaluate(checkpoint_path, save_name, num_episodes=3, tau_s=3.25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = EnergyNetwork(
        obs_dim=32,
        action_dim=7,
        hidden_dim=256,
        enc_output_dim=128,
        output_scale=1000.0,
    )
    dataset = RobosuiteDataset("Lift")
    trainer = FlowMatchingEnergyTrainer(
        network,
        dataset,
        device=device,
        dt=0.01,
        eps_max=0.01,
        tau_star=1.0,
    )
    trainer.load_checkpoint(checkpoint_path)

    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
    )

    for ep in range(num_episodes):
        obs = env.reset()
        frames = []
        total_reward = 0

        for _ in tqdm(range(200), desc=f"{save_name} Episode {ep + 1}"):
            state = env.sim.get_state().flatten()
            actions = trainer.sample_actions(state, tau_s=tau_s)
            state_tensor = (
                torch.tensor(state, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .expand(actions.shape[0], -1)
            )
            energies = trainer.network(state_tensor, actions)  # Shape: [num_samples]
            min_idx = energies.argmin()
            action = actions[min_idx].cpu().numpy()
            action = np.clip(action, -1.0, 1.0)

            obs, reward, done, _ = env.step(action)
            total_reward += reward
            frames.append(obs["agentview_image"][::-1])
            if done:
                break

        Path("videos").mkdir(exist_ok=True)
        video_path = f"videos/{save_name}_ep{ep + 1}.mp4"
        imageio.mimsave(video_path, frames, fps=20)
        print(
            f"{save_name} Episode {ep + 1}: reward={total_reward:.3f}, saved to {video_path}"
        )
    env.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evaluating Phase 1 (tau_s=3.25)")
    print("=" * 60)
    evaluate("phase1.pt", "phase1", num_episodes=3, tau_s=3.25)

    print("\n" + "=" * 60)
    print("Evaluating Phase 2 (tau_s=3.25)")
    print("=" * 60)
    evaluate("phase2.pt", "phase2", num_episodes=3, tau_s=3.25)

    print("\n" + "=" * 60)
    print("All evaluations complete! Videos saved to ./videos/")
    print("=" * 60)
