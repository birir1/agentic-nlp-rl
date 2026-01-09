import torch
import numpy as np

from utils.policy import AffectivePolicy


class LearningAffectiveAgent:
    """
    Policy-gradient learning agent.
    """

    def __init__(self, lr=3e-4):
        self.policy = AffectivePolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.reset_episode()

    def reset_episode(self):
        self.log_probs = []
        self.rewards = []

    def act(self, obs):
        state = torch.tensor([
            obs["affective_state"]["valence"],
            obs["affective_state"]["arousal"],
            obs["affective_state"]["dominance"],
            obs["timestep"] / 10.0,
        ], dtype=torch.float32)

        affect_dist, strategy_dist = self.policy(state)

        affect_action = affect_dist.sample()
        strategy = strategy_dist.sample()

        log_prob = (
            affect_dist.log_prob(affect_action).sum()
            + strategy_dist.log_prob(strategy)
        )

        self.log_probs.append(log_prob)

        message = [
            "Let's outline energy usage patterns.",
            "We should coordinate device scheduling.",
            "Optimizing peak loads will help.",
        ][strategy.item()]

        return {
            "message": message,
            "reflection": message,
            "affect_delta": {
                "valence": affect_action[0].item(),
                "arousal": affect_action[1].item(),
                "dominance": affect_action[2].item(),
            },
        }

    def observe(self, reward):
        self.rewards.append(reward)

    def update(self, gamma=0.99):
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -torch.sum(torch.stack(self.log_probs) * returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_episode()
