import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any, Tuple


class BaseAgent(nn.Module):
    """
    Stateless baseline RL agent for text-based environments.

    Observation: dict with at least {"text": str}
    Action: discrete index (mapped to text externally)

    This agent DOES NOT:
    - plan
    - use tools
    - store memory

    It is a true baseline.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs_tensor: shape [batch_size, obs_dim]

        Returns:
            logits: action logits
            value: state value
        """
        logits = self.policy_net(obs_tensor)
        value = self.value_net(obs_tensor).squeeze(-1)
        return logits, value

    def act(
        self,
        obs_tensor: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        Sample or select an action.

        Returns:
            {
                "action": Tensor[int],
                "log_prob": Tensor[float],
                "value": Tensor[float]
            }
        """
        logits, value = self.forward(obs_tensor)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
        }

    def evaluate_actions(
        self,
        obs_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log-probs and entropy for PPO updates.
        """
        logits, value = self.forward(obs_tensor)
        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()

        return {
            "log_prob": log_prob,
            "entropy": entropy,
            "value": value,
        }
