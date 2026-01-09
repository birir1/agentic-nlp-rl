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
        num_agents: int = 2,
        max_steps: int = 10,
        seed: int = 0,
    ):
        """
        Multi-agent text environment.

        Args:
            task_description: string describing the task.
            agent_ids: optional list of agent IDs.
            num_agents: if agent_ids not provided, generate this many agents.
            max_steps: maximum timesteps in an episode.
            seed: random seed for reproducibility.
        """
        random.seed(seed)

        # ✅ Define agent IDs dynamically if not provided
        if agent_ids:
            self.agent_ids = agent_ids
        else:
            self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

        self.num_agents = len(self.agent_ids)
        self.task_description = task_description
        self.max_steps = max_steps

        self.timestep = 0
        self.shared_memory = []

        # Initialize affective states for each agent
        self.affective_states = {
            aid: {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            for aid in self.agent_ids
        }

    def reset(self):
        """Reset environment for a new episode."""
        self.timestep = 0
        self.shared_memory = []

        for aid in self.agent_ids:
            self.affective_states[aid] = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

        return self._get_obs()

    def _get_obs(self):
        """Return current observation."""
        return {
            "task_description": self.task_description,
            "shared_context": (
                "The task has just started."
                if self.timestep == 0
                else "Ongoing collaboration."
            ),
            "affective_states": {aid: self.affective_states[aid].copy() for aid in self.agent_ids},
            "shared_memory": self.shared_memory.copy(),
            "timestep": self.timestep,
        }

    def step(self, actions: Dict[str, Dict]):
        """
        Perform one environment step.

        Args:
            actions: dict mapping agent_id -> {"message": str, "affect_delta": dict}

        Returns:
            obs: current observation
            rewards: dict of rewards for each agent
            done: boolean if episode is finished
            info: extra info dict (shared_memory, affective_states)
        """
        rewards = {}
        self.timestep += 1

        for agent_id in self.agent_ids:
            action = actions.get(agent_id, {})

            # Store message in shared memory
            self.shared_memory.append({
                "sender": agent_id,
                "content": action.get("message", ""),
                "timestep": self.timestep,
            })

            # Update affective state
            delta = action.get("affect_delta", {})
            for dim in ["valence", "arousal", "dominance"]:
                self.affective_states[agent_id][dim] += delta.get(dim, 0.0)
                # Clip affective values between -1 and 1
                self.affective_states[agent_id][dim] = max(-1.0, min(1.0, self.affective_states[agent_id][dim]))

            # Simulated emotion model output (probabilities)
            emotion_out = {
                "probs": {
                    "joy": random.uniform(0.2, 0.6),
                    "neutral": random.uniform(0.1, 0.4),
                    "sadness": random.uniform(0.0, 0.2),
                }
            }

            # Convert emotion probs to valence
            inferred_valence = emotion_to_valence(emotion_out["probs"])
            self.affective_states[agent_id]["valence"] += inferred_valence * 0.1

            # Reward: positivity + cooperation
            rewards[agent_id] = round(0.5 + max(0.0, self.affective_states[agent_id]["valence"]), 3)

        done = self.timestep >= self.max_steps

        obs = self._get_obs()
        info = {
            "shared_memory": self.shared_memory.copy(),
            "affective_states": {aid: self.affective_states[aid].copy() for aid in self.agent_ids},
        }

        # ✅ Must return 4 values for compatibility with all simulation scripts
        return obs, rewards, done, info
