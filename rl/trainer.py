import os
import csv
import time
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt


class RLTrainer:
    """
    Generic RL trainer for text-based environments.

    Responsibilities:
    - Environment interaction
    - Logging
    - Saving tables and figures
    - Delegating learning to the algorithm (e.g. PPO)

    This class DOES NOT:
    - Define policy updates
    - Define reward functions
    """

    def __init__(
        self,
        agent,
        env,
        algorithm,
        output_dir: str,
        device: str = "cpu",
        max_steps_per_episode: int = 200,
    ):
        self.agent = agent.to(device)
        self.env = env
        self.algorithm = algorithm
        self.device = device
        self.max_steps = max_steps_per_episode

        self.output_dir = output_dir
        self.fig_dir = os.path.join(output_dir, "figures")
        self.table_dir = os.path.join(output_dir, "tables")

        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

        # Logs
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.loss_logs: List[Dict[str, float]] = []

    def train(self, num_episodes: int):
        print(f"[Trainer] Starting training for {num_episodes} episodes")

        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            obs = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0

            self.algorithm.reset_buffer()

            for step in range(self.max_steps):
                obs_tensor = self.env.encode_observation(obs).to(self.device)

                with torch.no_grad():
                    action_dict = self.agent.act(obs_tensor)

                action = action_dict["action"].item()
                next_obs, reward, done, info = self.env.step(action)

                self.algorithm.store_transition(
                    obs_tensor=obs_tensor,
                    action=action,
                    reward=reward,
                    log_prob=action_dict["log_prob"],
                    value=action_dict["value"],
                    done=done,
                )

                episode_reward += reward
                episode_steps += 1
                obs = next_obs

                if done:
                    break

            # Update policy
            loss_info = self.algorithm.update(self.agent)
            self.loss_logs.append(loss_info)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)

            if episode % 10 == 0:
                print(
                    f"[Episode {episode:04d}] "
                    f"Reward={episode_reward:.2f} "
                    f"Steps={episode_steps} "
                    f"Loss={loss_info}"
                )

        elapsed = time.time() - start_time
        print(f"[Trainer] Training finished in {elapsed:.2f}s")

        self._save_tables()
        self._save_figures()

    # =======================
    # Logging & Visualization
    # =======================

    def _save_tables(self):
        print("[Trainer] Saving tables...")

        # Episode metrics
        episode_table_path = os.path.join(self.table_dir, "episode_metrics.csv")
        with open(episode_table_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length"])
            for i, (r, l) in enumerate(zip(self.episode_rewards, self.episode_lengths)):
                writer.writerow([i + 1, r, l])

        # Loss metrics
        loss_table_path = os.path.join(self.table_dir, "loss_metrics.csv")
        if len(self.loss_logs) > 0:
            keys = self.loss_logs[0].keys()
            with open(loss_table_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["episode"] + list(keys))
                writer.writeheader()
                for i, log in enumerate(self.loss_logs):
                    row = {"episode": i + 1}
                    row.update(log)
                    writer.writerow(row)

        print(f"[✓] Tables saved to {self.table_dir}")

    def _save_figures(self):
        print("[Trainer] Saving figures...")

        # Reward curve
        plt.figure()
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Reward Curve")
        plt.grid(True)
        reward_fig_path = os.path.join(self.fig_dir, "reward_curve.png")
        plt.savefig(reward_fig_path)
        plt.close()

        # Episode length curve
        plt.figure()
        plt.plot(self.episode_lengths)
        plt.xlabel("Episode")
        plt.ylabel("Episode Length")
        plt.title("Episode Length Curve")
        plt.grid(True)
        length_fig_path = os.path.join(self.fig_dir, "episode_length.png")
        plt.savefig(length_fig_path)
        plt.close()

        # Loss curves (if scalar)
        if len(self.loss_logs) > 0:
            for key in self.loss_logs[0].keys():
                values = [log[key] for log in self.loss_logs]

                if np.isscalar(values[0]):
                    plt.figure()
                    plt.plot(values)
                    plt.xlabel("Episode")
                    plt.ylabel(key)
                    plt.title(f"{key} over Training")
                    plt.grid(True)
                    fig_path = os.path.join(self.fig_dir, f"{key}.png")
                    plt.savefig(fig_path)
                    plt.close()

        print(f"[✓] Figures saved to {self.fig_dir}")
