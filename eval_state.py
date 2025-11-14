import json
from pathlib import Path

import imageio
import numpy as np
import robosuite as suite
import torch
from tqdm import tqdm

from train_state import EnergyNetwork as StateEnergyNetwork
from utils import EnergyNetwork, EqMTrainer, RobosuiteDataset


def rollout_trajectory(env, action_trainer, state_trainer, initial_action, initial_state, num_steps=5, num_action_samples=32):
    cumulative_energy = 0.0
    
    # Set to initial state and execute first action
    env.sim.set_state(initial_state)
    env.sim.forward()
    
    env.step(initial_action.cpu().numpy())
    current_state = env.sim.get_state().flatten()
    
    # Compute energy of first next state
    state_tensor = torch.tensor(current_state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        energy = state_trainer.ema_network(state_tensor).item()
    cumulative_energy += energy
    
    # Rollout remaining steps
    for step in range(1, num_steps):
        # Sample candidate actions
        actions = action_trainer.sample_actions(current_state, num_samples=num_action_samples)
        
        # Evaluate each action
        next_state_energies = []
        saved_state = env.sim.get_state()
        
        for i in range(actions.shape[0]):
            env.step(actions[i].cpu().numpy())
            next_state = env.sim.get_state().flatten()
            
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                energy = state_trainer.ema_network(next_state_tensor).item()
            next_state_energies.append(energy)
            
            env.sim.set_state(saved_state)
            env.sim.forward()
        
        # Select best action and execute it
        min_idx = np.argmin(next_state_energies)
        selected_action = actions[min_idx].cpu().numpy()
        
        env.step(selected_action)
        current_state = env.sim.get_state().flatten()
        cumulative_energy += next_state_energies[min_idx]
    return cumulative_energy


def eval_eqm():
    videos_dir = Path("videos")
    videos_dir.mkdir(exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    ep_rewards = []
    ep_successes = []
    ep_mean_energies = []
    
    for ep in range(num_eval_episodes):
        obs = env.reset()
        ep_reward = 0
        frames = []
        episode_energies = []

        for step in tqdm(
            range(num_episode_steps), desc=f"Episode {ep + 1}/{num_eval_episodes}"
        ):
            current_state = env.sim.get_state().flatten()
            saved_state = env.sim.get_state()

            # Sample candidate initial actions
            actions = action_trainer.sample_actions(
                current_state, num_samples=num_action_samples
            )
            
            # Evaluate each candidate action with multi-step rollout
            cumulative_energies = []
            for i in range(actions.shape[0]):
                cumulative_energy = rollout_trajectory(
                    env, 
                    action_trainer, 
                    state_trainer, 
                    actions[i], 
                    saved_state,
                    num_steps=lookahead_steps,
                    num_action_samples=lookahead_action_samples
                )
                cumulative_energies.append(cumulative_energy)
                
                # Restore to original state
                env.sim.set_state(saved_state)
                env.sim.forward()

            # Select action with minimum cumulative energy
            min_idx = np.argmin(cumulative_energies)
            selected_action = actions[min_idx].cpu().numpy()
            min_cumulative_energy = cumulative_energies[min_idx]
            
            # Execute selected action in actual environment
            obs, reward, done, info = env.step(selected_action)
            ep_reward += reward
            frames.append(obs["agentview_image"][::-1])
            episode_energies.append(min_cumulative_energy)
            
            if done:
                break

        video_path = videos_dir / f"{save_name}_ep{ep + 1}.mp4"
        imageio.mimsave(video_path, frames, fps=20)

        ep_rewards.append(ep_reward)
        ep_successes.append(info.get("success", 0))
        mean_energy = np.mean(episode_energies)
        ep_mean_energies.append(mean_energy)
        print(
            f"Episode {ep + 1}: reward={ep_reward:.1f}, success={info.get('success', 0)}, "
            f"mean_cumulative_energy={mean_energy:.2f}, video saved to {video_path}"
        )

    mean_reward = np.mean(ep_rewards)
    std_reward = np.std(ep_rewards)
    mean_success = np.mean(ep_successes)
    mean_energy_overall = np.mean(ep_mean_energies)

    print("\n" + "=" * 60)
    print(f"Results for {save_name}")
    print("=" * 60)
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {mean_success:.2%}")
    print(f"Mean Cumulative Energy ({lookahead_steps}-step): {mean_energy_overall:.2f}")
    print("=" * 60)

    # Save results to JSON
    results = {
        "method": save_name,
        "episode_rewards": ep_rewards,
        "episode_successes": ep_successes,
        "episode_mean_energies": ep_mean_energies,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_success": float(mean_success),
        "mean_energy": float(mean_energy_overall),
        "num_episodes": num_eval_episodes,
        "num_action_samples": num_action_samples,
        "lookahead_steps": lookahead_steps,
        "lookahead_action_samples": lookahead_action_samples,
        "action_selection": f"min_cumulative_energy_{lookahead_steps}step",
    }

    results_path = results_dir / f"{save_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    env.close()


if __name__ == "__main__":
    checkpoints_dir = Path("checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RobosuiteDataset("Square")

    action_network = EnergyNetwork(
        obs_dim=45,
        action_dim=7,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    action_trainer = EqMTrainer(
        action_network,
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
    action_trainer.load_checkpoint(checkpoints_dir / "eqm.pt")
    print("Loaded action policy from eqm.pt")
    
    state_network = StateEnergyNetwork(
        input_dim=45,
        hidden_dim=512,
        enc_output_dim=256,
        output_scale=1000.0,
    )
    state_trainer = EqMTrainer(
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
    state_trainer.load_checkpoint(checkpoints_dir / "eqm_state.pt")
    print("Loaded state energy model from eqm_state.pt")

    env = suite.make(env_name="NutAssemblySquare", robots="Panda", ignore_done=True)

    num_eval_episodes = 10
    num_episode_steps = 200
    num_action_samples = 8
    lookahead_steps = 3
    lookahead_action_samples = 1
    save_name = f"eqm_min_cumulative_energy_{lookahead_steps}step"
    eval_eqm()