"""
EA-MARL Agent for Multi-Agent NLP + Affective RL
"""

import numpy as np
from typing import Dict
from agents.baselines import BaseAgent

class EAMARLAgent(BaseAgent):
    """
    Evolutionary Affective Multi-Agent Reinforcement Learning Agent.

    Key features:
    - Learns cooperative policies over text + affect.
    - Maintains internal affective state.
    - Updates action probabilities via reward + valence feedback.
    """

    def __init__(self, agent_id: str, n_actions: int = 4, alpha: float = 0.2):
        """
        Args:
            agent_id: unique ID (e.g., 'agent_0')
            n_actions: number of discrete text actions
            alpha: learning rate for policy update
        """
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.alpha = alpha

        # Initialize uniform policy over actions
        self.policy = np.ones(n_actions) / n_actions

        # Internal affective state
        self.valence = 0.0
        self.arousal = 0.0
        self.dominance = 0.0

        # Action library
        self.actions = [
            "Let's optimize energy usage.",
            "We could schedule appliances efficiently.",
            "Divide tasks among agents.",
            "Consider renewable energy sources."
        ]

    def act(self, observation: Dict) -> Dict:
        # Sample action based on policy
        action_idx = np.random.choice(self.n_actions, p=self.policy)
        message = self.actions[action_idx]

        # Compute affect delta heuristically
        affect_delta = self.compute_affect_delta(action_idx)

        # Update internal affective state
        self.valence += affect_delta["valence"]
        self.arousal += affect_delta["arousal"]
        self.dominance += affect_delta["dominance"]

        return {
            "message": message,
            "reflection": f"EA-MARL step, policy idx {action_idx}",
            "affect_delta": affect_delta,
            "action_idx": action_idx
        }

    def compute_affect_delta(self, action_idx: int) -> Dict:
        """
        Map actions to affective changes (heuristic or learned later)
        """
        # Simple heuristic: cooperative actions increase valence
        valence_map = [0.1, 0.12, 0.08, 0.05]
        arousal_map = [0.0, 0.0, 0.01, 0.02]
        dominance_map = [0.02, 0.01, 0.03, 0.0]

        return {
            "valence": valence_map[action_idx],
            "arousal": arousal_map[action_idx],
            "dominance": dominance_map[action_idx],
        }

    def update_policy(self, reward: float, inferred_valence: float, action_idx: int):
        """
        Update policy using EA-MARL rule:
        - Reward + valence boost
        - Normalize probabilities
        """
        delta = self.alpha * (reward + inferred_valence)
        self.policy[action_idx] += delta

        # Ensure probabilities stay valid
        self.policy = np.clip(self.policy, 0.01, None)
        self.policy /= self.policy.sum()
