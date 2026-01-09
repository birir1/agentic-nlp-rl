import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List


class RolloutBuffer:
    """
    Stores trajectories for PPO updates
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()


class PPO:
    """
    Proximal Policy Optimization (PPO)

    Compatible with:
    - Text-based environments
    - LLM or neural policies
    - Value + policy heads
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = device

        self.buffer = RolloutBuffer()
        self.optimizer = None
        self.lr = lr

    # ======================
    # Buffer Interaction
    # ======================

    def reset_buffer(self):
        self.buffer.clear()

    def store_transition(
        self,
        obs_tensor,
        action,
        reward,
        log_prob,
        value,
        done,
    ):
        self.buffer.obs.append(obs_tensor)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.log_probs.append(log_prob.detach())
        self.buffer.values.append(value.detach())
        self.buffer.dones.append(done)

    # ======================
    # Core PPO Logic
    # ======================

    def _compute_returns_and_advantages(self):
        rewards = self.buffer.rewards
        values = self.buffer.values + [torch.tensor(0.0).to(self.device)]
        dones = self.buffer.dones

        advantages = []
        returns = []

        gae = 0.0
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = (
                delta
                + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            )
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        advantages = torch.stack(advantages).detach()
        returns = torch.stack(returns).detach()

        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        return returns, advantages

    def update(self, agent) -> Dict[str, float]:
        """
        Performs PPO update step
        """
        if self.optimizer is None:
            self.optimizer = optim.Adam(agent.parameters(), lr=self.lr)

        returns, advantages = self._compute_returns_and_advantages()

        obs = torch.stack(self.buffer.obs).to(self.device)
        actions = torch.tensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device)

        policy_losses = []
        value_losses = []
        entropies = []

        num_samples = len(actions)
        indices = np.arange(num_samples)

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Forward pass
                action_dict = agent.evaluate_actions(
                    batch_obs, batch_actions
                )

                log_probs = action_dict["log_prob"]
                entropy = action_dict["entropy"]
                values = action_dict["value"]

                # PPO objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_eps, 1 + self.clip_eps
                )

                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages,
                ).mean()

                value_loss = nn.functional.mse_loss(
                    values.squeeze(), batch_returns
                )

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }
