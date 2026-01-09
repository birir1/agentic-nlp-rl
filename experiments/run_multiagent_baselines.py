"""
Run multi-agent baseline agents on the MultiAgentTextEnv.
"""

import numpy as np
from utils.logger import ExperimentLogger


from envs.multi_agent_text_env import MultiAgentTextEnv
from agents.baselines import (
    RandomAgent,
    RuleBasedAffectiveAgent,
    FrozenLLMAgent,
)

TASK_DESCRIPTION = "Plan energy reduction strategies for a smart home."

AGENTS = {
    "random": RandomAgent,
    "rule_affective": RuleBasedAffectiveAgent,
    "frozen_llm": FrozenLLMAgent,
}

SEEDS = [0, 1, 2]
MAX_STEPS = 10


def run_episode(agent_cls, seed):
    env = MultiAgentTextEnv(
        task_description=TASK_DESCRIPTION,
        max_steps=MAX_STEPS,
        seed=seed,
    )

    agents = {aid: agent_cls(aid) for aid in env.agent_ids}

    obs = env.reset()
    done = False

    total_reward = 0.0
    final_valences = []

    while not done:
        actions = {
            aid: agents[aid].act(obs)
            for aid in env.agent_ids
        }

        # ✅ FIX: handle variable-length step() output
        step_out = env.step(actions)

        if len(step_out) == 3:
            obs, rewards, done = step_out
        elif len(step_out) == 4:
            obs, rewards, done, _ = step_out
        elif len(step_out) == 5:
            obs, rewards, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            raise RuntimeError(
                f"Unexpected number of returns from env.step(): {len(step_out)}"
            )

        total_reward += np.mean(list(rewards.values()))

    for aid in env.agent_ids:
        final_valences.append(
            obs["affective_states"][aid]["valence"]
        )

    return {
        "total_reward": total_reward,
        "final_valence": float(np.mean(final_valences)),
    }


def main():
    logger = ExperimentLogger()

    print("\n=== Running Multi-Agent Baselines ===\n")

    for name, agent_cls in AGENTS.items():
        metrics = [run_episode(agent_cls, s) for s in SEEDS]

        mean_reward = np.mean([m["total_reward"] for m in metrics])
        mean_valence = np.mean([m["final_valence"] for m in metrics])

        print(
            f"{name:15s} | "
            f"Reward: {mean_reward:.3f} | "
            f"Valence: {mean_valence:.3f}"
        )


if __name__ == "__main__":
    main()
