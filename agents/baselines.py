import random
from typing import Dict


# -------------------------
# Base Agent
# -------------------------
class BaseAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def act(self, observation: dict) -> dict:
        """Return an action dict with 'message' and 'affect_delta'"""
        raise NotImplementedError


# -------------------------
# Random Agent
# -------------------------
class RandomAgent(BaseAgent):
    def act(self, observation: dict) -> dict:
        messages = [
            "Divide tasks among agents.",
            "Consider renewable energy sources.",
            "Let's optimize energy usage.",
            "We could schedule appliances efficiently.",
        ]
        msg = random.choice(messages)
        affect_delta = {
            "valence": random.uniform(-0.1, 0.1),
            "arousal": random.uniform(-0.1, 0.1),
            "dominance": 0.0,
        }
        return {"message": msg, "affect_delta": affect_delta}


# -------------------------
# EA-MARL Agent
# -------------------------
class EAMARLAgent(BaseAgent):
    def act(self, observation: dict) -> dict:
        # Example: simple heuristic - encourage positive valence
        avg_valence = sum(
            state["valence"] for state in observation["affective_states"].values()
        ) / len(observation["affective_states"])
        delta = 0.1 if avg_valence >= 0 else -0.1
        return {
            "message": "Coordinating energy optimization.",
            "affect_delta": {"valence": delta, "arousal": 0.0, "dominance": 0.0},
        }


# -------------------------
# Frozen LLM Agent (placeholder)
# -------------------------
class FrozenLLMAgent(BaseAgent):
    def act(self, observation: dict) -> dict:
        # Placeholder: deterministic message
        return {
            "message": "Following LLM instruction.",
            "affect_delta": {"valence": 0.05, "arousal": 0.0, "dominance": 0.0},
        }


# -------------------------
# Rule-Based Affective Agent (placeholder)
# -------------------------
class RuleBasedAffectiveAgent(BaseAgent):
    def act(self, observation: dict) -> dict:
        avg_valence = sum(
            state["valence"] for state in observation["affective_states"].values()
        ) / len(observation["affective_states"])
        delta = 0.1 if avg_valence > 0 else -0.1
        return {
            "message": "Following rule-based affect.",
            "affect_delta": {"valence": delta, "arousal": 0.0, "dominance": 0.0},
        }
