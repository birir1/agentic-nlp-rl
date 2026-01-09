import torch
import torch.nn as nn
import torch.distributions as D


class AffectivePolicy(nn.Module):
    """
    Simple policy for affect-aware control.
    """

    def __init__(self, obs_dim=4, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Continuous affect control
        self.affect_head = nn.Linear(hidden_dim, 3)

        # Discrete strategy selection
        self.strategy_head = nn.Linear(hidden_dim, 3)

    def forward(self, obs):
        h = self.net(obs)

        affect_mean = torch.tanh(self.affect_head(h))
        affect_dist = D.Normal(affect_mean, 0.1)

        strategy_logits = self.strategy_head(h)
        strategy_dist = D.Categorical(logits=strategy_logits)

        return affect_dist, strategy_dist
