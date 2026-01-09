"""
EA-MARL Multi-Agent Training Script
"""

import numpy as np
from tqdm import trange

from envs.multi_agent_text_env import MultiAgentTextEnv
from agents.baselines import BaseAgent, RandomAgent, RuleBasedAffectiveAgent, FrozenLLMAgent
from agents.ea_marl_agent import EAMARLAgent

# Task description for the multi-agent environment
TASK_DESCRIPTION = "Collaboratively plan a solution to reduce energy consumption in a smart home."

# Baseline agents to compare against EA-MARL agent
AGENTS = {
    "random": RandomAgent,
    "rule_affective": RuleBasedAffectiveAgent,
    "frozen_llm": FrozenLLMAgent,
    "ea_marl": EAMARLAgent,
}

EPISODES = 50
MAX_STEPS = 10
SEEDS = [0, 1, 2]

def run_episode(agent_cls, seed):
    """
    Run a single episode for a given agent class and random seed.
    """
    # Initialize multi-agent environment
    env = MultiAgentTextEnv(task_description=TASK_DESCRIPTION, seed=seed)

    # Initialize agents with agent_id
    agents = {aid: agent_cls(aid) if agent_cls == EAMARLAgent else agent_cls(aid) 
              for aid in env.agent_ids}

    obs = env.reset()
    done = False
    total_rewards = {aid: 0.0 for aid in env.agent_ids}

    while not done:
        # Collect actions from all agents
        actions = {aid: agents[aid].act(obs) for aid in env.agent_ids}

        # Step environment (note: env.step returns 4 values)
        obs, rewards, done, _ = env.step(actions)

        # Accumulate rewards
        for aid in env.agent_ids:
            total_rewards[aid] += rewards[aid]

    # Compute mean metrics across agents
    mean_reward = sum(total_rewards.values()) / len(total_rewards)
    mean_valence = sum(obs["affective_states"][aid]["valence"] for aid in env.agent_ids) / len(env.agent_ids)

    return {"mean_reward": mean_reward, "mean_valence": mean_valence}


def main():
    print("\n=== Running EA-MARL Multi-Agent Training ===\n")

    for name, agent_cls in AGENTS.items():
        all_metrics = []

        for seed in SEEDS:
            metrics = run_episode(agent_cls, seed)
            all_metrics.append(metrics)

        mean_reward = np.mean([m["mean_reward"] for m in all_metrics])
        mean_valence = np.mean([m["mean_valence"] for m in all_metrics])

        print(
            f"{name:15s} | "
            f"Reward: {mean_reward:.3f} | "
            f"Valence: {mean_valence:.3f}"
        )


if __name__ == "__main__":
    main()
