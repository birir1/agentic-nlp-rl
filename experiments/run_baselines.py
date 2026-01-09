"""
Run baseline agents on the agentic text environment.
"""

import numpy as np
from tqdm import trange

from envs.text_task_env import TextTaskEnv
from agents.baselines import (
    RandomAgent,
    RuleBasedAffectiveAgent,
    FrozenLLMAgent,
)

from utils.metrics import MetricsLogger


# ---------------------------
# Configuration
# ---------------------------

TASK_DESCRIPTION = (
    "Collaboratively plan a solution to reduce energy consumption "
    "in a smart home."
)

AGENTS = {
    "random": RandomAgent,
    "rule_affective": RuleBasedAffectiveAgent,
    "frozen_llm": FrozenLLMAgent,
}

SEEDS = [0, 1, 2]
MAX_STEPS = 10


# ---------------------------
# Experiment Logic
# ---------------------------

def run_agent(agent_cls, seed):
    """
    Run a single agent for one episode and return episode-level metrics.
    """
    env = TextTaskEnv(
        task_description=TASK_DESCRIPTION,
        max_steps=MAX_STEPS,
        seed=seed,
    )

    agent = agent_cls()
    obs = env.reset()
    done = False

    logger = MetricsLogger()
    trajectory = []
    total_reward = 0.0

    while not done:
        action = agent.act(obs)
        result = env.step(action)

        obs = result.observation
        done = result.done
        total_reward += result.reward

        trajectory.append({
            "timestep": obs["timestep"],
            "reward": result.reward,
            "valence": obs["affective_state"]["valence"],
            "arousal": obs["affective_state"]["arousal"],
            "dominance": obs["affective_state"]["dominance"],
        })

    # Persist trajectory for later analysis / plotting
    logger.log_episode(agent_cls.__name__, seed, trajectory)

    return {
        "total_reward": total_reward,
        "final_valence": obs["affective_state"]["valence"],
    }


def main():
    print("\n=== Running Baselines ===\n")

    for agent_name, agent_cls in AGENTS.items():
        all_metrics = []

        for seed in SEEDS:
            episode_metrics = run_agent(agent_cls, seed)
            all_metrics.append(episode_metrics)

        mean_reward = np.mean([m["total_reward"] for m in all_metrics])
        mean_valence = np.mean([m["final_valence"] for m in all_metrics])

        print(
            f"{agent_name:15s} | "
            f"Reward: {mean_reward:.3f} | "
            f"Valence: {mean_valence:.3f}"
        )


if __name__ == "__main__":
    main()
