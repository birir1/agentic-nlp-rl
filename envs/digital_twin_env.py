# envs/digital_twin_env.py
import numpy as np
import random


class DigitalTwinEnv:
    """
    Digital Twin Environment for Multi-Agent Coordination
    """

    def __init__(
        self,
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=42,
        max_steps=50,
        communication_enabled=True,
    ):
        self.num_agents = num_agents
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.max_steps = max_steps
        self.communication_enabled = communication_enabled

        random.seed(seed)
        np.random.seed(seed)

        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]

        self.reset()

    def reset(self):
        self.timestep = 0
        self.positions = {aid: np.random.rand(2) for aid in self.agent_ids}
        self.valence = {aid: random.uniform(0.2, 0.4) for aid in self.agent_ids}
        self.arousal = {aid: random.uniform(0.2, 0.4) for aid in self.agent_ids}
        self.weights = {aid: 0.5 for aid in self.agent_ids}

        self.communications = []
        self.coop_scores = []

    def step(self):
        self.timestep += 1

        step_positions = {}
        step_valence = {}
        step_arousal = {}
        step_weights = {}
        step_comm = []

        for i, aid in enumerate(self.agent_ids):
            direction = self.goal_pos - self.positions[aid]
            dist = np.linalg.norm(direction) + 1e-6
            direction = direction / dist

            # Base movement
            step = 0.08 * direction * self.weights[aid]
            noise = np.random.normal(0, 0.01, size=2)

            # Communication-induced coordination
            if self.communication_enabled:
                for j, other in enumerate(self.agent_ids):
                    if other != aid and random.random() < 0.5:
                        step_comm.append((aid, other))
                        self.valence[aid] += 0.02
                        self.valence[other] += 0.01

            self.positions[aid] = np.clip(
                self.positions[aid] + step + noise, 0.0, 1.0
            )

            # Affect → control coupling
            self.valence[aid] = np.clip(1.0 - dist, 0.0, 1.0)
            self.arousal[aid] = 0.5 + 0.5 * self.valence[aid]
            self.weights[aid] = 0.4 + 0.6 * self.valence[aid]

            step_positions[aid] = self.positions[aid].tolist()
            step_valence[aid] = self.valence[aid]
            step_arousal[aid] = self.arousal[aid]
            step_weights[aid] = self.weights[aid]

        coop_score = np.mean(list(self.valence.values()))
        self.coop_scores.append(coop_score)
        self.communications.append(step_comm)

        return {
            "positions": step_positions,
            "valence": step_valence,
            "arousal": step_arousal,
            "weights": step_weights,
            "communications": step_comm,
            "coop_score": coop_score,
        }
