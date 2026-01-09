"""
Text-Based Agentic Task Environment with Affective Dynamics

This environment simulates:
- Multi-step language-based tasks
- Agent-to-agent communication
- Internal cognition (reflection)
- Affective state evolution
- Reward signals (task + affective)

Designed for:
- Agentic NLP + RL research
- Communication analysis
- Emotion-aware reinforcement learning
"""

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# ---------------------------
# Data Structures
# ---------------------------

@dataclass
class AffectiveState:
    """
    Simple continuous affective representation.
    Values are bounded in [-1, 1].
    """
    valence: float = 0.0   # negative ↔ positive
    arousal: float = 0.0   # calm ↔ excited
    dominance: float = 0.0 # submissive ↔ confident

    def update(self, delta: Dict[str, float]):
        for k, v in delta.items():
            if hasattr(self, k):
                current = getattr(self, k)
                setattr(self, k, max(-1.0, min(1.0, current + v)))


@dataclass
class Message:
    sender: str
    content: str
    timestep: int


@dataclass
class StepResult:
    observation: Dict
    reward: float
    done: bool
    info: Dict = field(default_factory=dict)


# ---------------------------
# Environment
# ---------------------------

class TextTaskEnv:
    """
    A minimal agentic text environment.

    Observation:
        - task_description
        - shared_context
        - affective_state
        - message_history

    Action:
        {
            "message": str,
            "reflection": str,
            "affect_delta": {valence, arousal, dominance}
        }

    Reward:
        - task engagement
        - affective alignment
    """

    def __init__(
        self,
        task_description: str = (
            "Collaboratively plan a solution to reduce energy consumption in a smart home."
        ),
        max_steps: int = 10,
        target_valence: float = 0.5,
        seed: int | None = None,
    ):
        if seed is not None:
            random.seed(seed)

        self.task_description = task_description
        self.max_steps = max_steps
        self.target_valence = target_valence

        self.reset()

    # ---------------------------
    # Core API
    # ---------------------------

    def reset(self) -> Dict:
        self.timestep = 0
        self.done = False

        self.affective_state = AffectiveState()
        self.message_history: List[Message] = []
        self.shared_context = "The task has just started."

        return self._get_observation()

    def step(self, action: Dict) -> StepResult:
        if self.done:
            raise RuntimeError("Environment already terminated. Call reset().")

        self.timestep += 1

        # --- Communication ---
        msg_text = action.get("message", "")
        if msg_text:
            self.message_history.append(
                Message(
                    sender="agent",
                    content=msg_text,
                    timestep=self.timestep,
                )
            )

        # --- Affective update ---
        affect_delta = action.get("affect_delta", {})
        self.affective_state.update(affect_delta)

        # --- Reflection / shared context ---
        reflection = action.get("reflection", "")
        if reflection:
            self.shared_context = reflection

        # --- Reward ---
        reward, reward_info = self._compute_reward(msg_text)

        # --- Termination ---
        if self.timestep >= self.max_steps:
            self.done = True

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=self.done,
            info=reward_info,
        )

    # ---------------------------
    # Helpers
    # ---------------------------

    def _get_observation(self) -> Dict:
        return {
            "task_description": self.task_description,
            "shared_context": self.shared_context,
            "affective_state": {
                "valence": self.affective_state.valence,
                "arousal": self.affective_state.arousal,
                "dominance": self.affective_state.dominance,
            },
            "message_history": [
                {
                    "sender": m.sender,
                    "content": m.content,
                    "timestep": m.timestep,
                }
                for m in self.message_history
            ],
            "timestep": self.timestep,
        }

    def _compute_reward(self, message: str) -> Tuple[float, Dict]:
        """
        Reward components:
        1. Language engagement reward
        2. Affective alignment reward
        """

        # --- Engagement reward ---
        engagement_reward = min(len(message.split()) / 20.0, 1.0)

        # --- Affective alignment ---
        affect_error = abs(self.affective_state.valence - self.target_valence)
        affect_reward = 1.0 - affect_error

        total_reward = 0.5 * engagement_reward + 0.5 * affect_reward

        return total_reward, {
            "engagement_reward": engagement_reward,
            "affect_reward": affect_reward,
            "valence": self.affective_state.valence,
        }


# ---------------------------
# Smoke Test
# ---------------------------

if __name__ == "__main__":
    env = TextTaskEnv()

    obs = env.reset()
    print("Initial Observation:")
    print(obs)

    for step in range(3):
        action = {
            "message": "I suggest we start by analyzing peak usage times.",
            "reflection": "We are identifying key factors influencing energy use.",
            "affect_delta": {"valence": 0.1, "dominance": 0.05},
        }

        result = env.step(action)
        print(f"\nStep {step + 1}")
        print("Reward:", result.reward)
        print("Affective State:", result.observation["affective_state"])
