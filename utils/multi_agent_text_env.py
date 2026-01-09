"""
Multi-Agent Text Environment with Affective State and Shared Memory
"""

import random
from typing import Dict, List

from utils.emotion_mapping import emotion_to_valence


class MultiAgentTextEnv:
    def __init__(
        self,
        task_description: str,
        agent_ids: List[str] = None,
        max_steps: int = 10,
        seed: int = 0,
    ):
        random.seed(seed)

        self.task_description = task_description
        self.agent_ids = agent_ids or ["agent_0", "agent_1"]
        self.max_steps = max_steps

        self.timestep = 0
        self.shared_memory = []
        self.affective_states = {
            aid: {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            for aid in self.agent_ids
        }

    def reset(self):
        self.timestep = 0
        self.shared_memory = []
        for aid in self.agent_ids:
            self.affective_states[aid] = {
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
            }

        return self._get_obs()

    def _get_obs(self):
        return {
            "task_description": self.task_description,
            "shared_context": "The task has just started."
            if self.timestep == 0
            else "Ongoing collaboration.",
            "affective_states": self.affective_states,
            "shared_memory": self.shared_memory,
            "timestep": self.timestep,
        }

    def step(self, actions: Dict[str, Dict]):
        """
        actions = {
            agent_id: {
                "message": str,
                "affect_delta": {valence, arousal, dominance}
            }
        }
        """

        rewards = {}
        self.timestep += 1

        for agent_id, action in actions.items():
            # Store message
            self.shared_memory.append({
                "sender": agent_id,
                "content": action.get("message", ""),
                "timestep": self.timestep,
            })

            # Update affective state
            delta = action.get("affect_delta", {})
            for k in ["valence", "arousal", "dominance"]:
                self.affective_states[agent_id][k] += delta.get(k, 0.0)

            # Simulated emotion model output
            emotion_out = {
                "probs": {
                    "joy": random.uniform(0.2, 0.6),
                    "neutral": random.uniform(0.1, 0.4),
                    "sadness": random.uniform(0.0, 0.2),
                }
            }

            inferred_valence = emotion_to_valence(emotion_out["probs"])
            self.affective_states[agent_id]["valence"] += inferred_valence * 0.1

            # Reward = cooperation + positivity
            rewards[agent_id] = round(
                0.5 + max(0.0, self.affective_states[agent_id]["valence"]), 3
            )

        done = self.timestep >= self.max_steps

        obs = self._get_obs()
        info = {
            "shared_memory": self.shared_memory,
            "affective_states": self.affective_states,
        }

        # ✅ CRITICAL: return FOUR values
        return obs, rewards, done, info
