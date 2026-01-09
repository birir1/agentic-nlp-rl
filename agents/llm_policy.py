import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple


class LLMPolicy(nn.Module):
    """
    Policy + Value network for text / agentic RL

    This module is intentionally generic:
    - Accepts pre-embedded observations (text, symbolic, hybrid)
    - Compatible with PPO, A2C, DQN-style algorithms
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device

        # ==========================
        # Shared Encoder
        # ==========================
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ==========================
        # Policy Head
        # ==========================
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # ==========================
        # Value Head
        # ==========================
        self.value_head = nn.Linear(hidden_dim, 1)

        self.to(device)

    # ==========================
    # Forward Pass
    # ==========================

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: action logits
            value: state value
        """
        features = self.encoder(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    # ==========================
    # Acting
    # ==========================

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Sample an action from the policy

        Returns:
            action
            log_prob
            value
        """
        obs = obs.to(self.device)

        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value.squeeze(-1),
        }

    # ==========================
    # PPO Evaluation
    # ==========================

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Used during PPO updates
        """
        obs = obs.to(self.device)
        actions = actions.to(self.device)

        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return {
            "log_prob": log_probs,
            "entropy": entropy,
            "value": value.squeeze(-1),
        }
