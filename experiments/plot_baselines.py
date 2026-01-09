"""
Run baseline agents and generate plots + saved results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from envs.text_task_env import TextTaskEnv
from agents.baselines import (
    RandomAgent,
    RuleBasedAffectiveAgent,
    FrozenLLMAgent,
)

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

OUTPUT_DIR = "outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
RES_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def run_agent(agent_cls, seed):
    env = TextTaskEnv(
        task_description=TASK_DESCRIPTION,
        seed=seed,
        max_steps=MAX_STEPS,
    )
    agent = agent_cls()

    obs = env.reset()
    done = False

    rewards = []
    valences = []

    while not done:
        action = agent.act(obs)
        result = env.step(action)

        obs = result.observation
        done = result.done

        rewards.append(float(result.reward))
        valences.append(float(obs["affective_state"]["valence"]))

    return {
        "rewards": rewards,
        "valences": valences,
        "total_reward": float(np.sum(rewards)),
        "final_valence": float(valences[-1]),
    }


def aggregate_runs(runs):
    max_len = max(len(r["rewards"]) for r in runs)

    def pad(seq):
        return seq + [seq[-1]] * (max_len - len(seq))

    rewards = np.array([pad(r["rewards"]) for r in runs], dtype=np.float32)
    valences = np.array([pad(r["valences"]) for r in runs], dtype=np.float32)

    return {
        # curves
        "mean_reward_curve": rewards.mean(axis=0).tolist(),
        "std_reward_curve": rewards.std(axis=0).tolist(),
        "mean_valence_curve": valences.mean(axis=0).tolist(),
        "std_valence_curve": valences.std(axis=0).tolist(),
        # scalars
        "mean_total_reward": float(rewards.sum(axis=1).mean()),
        "mean_final_valence": float(valences[:, -1].mean()),
    }


def plot_curve(x, mean, std, ylabel, title, filename):
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean)
    plt.fill_between(x, np.array(mean) - np.array(std),
                     np.array(mean) + np.array(std), alpha=0.2)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_comparison(curves, ylabel, title, filename):
    plt.figure(figsize=(6, 4))

    for name, curve in curves.items():
        plt.plot(curve, label=name)

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    summary = {}

    reward_curves = {}
    valence_curves = {}

    for name, agent_cls in AGENTS.items():
        runs = []

        for seed in SEEDS:
            runs.append(run_agent(agent_cls, seed))

        agg = aggregate_runs(runs)
        summary[name] = agg

        timesteps = np.arange(len(agg["mean_reward_curve"]))

        reward_curves[name] = agg["mean_reward_curve"]
        valence_curves[name] = agg["mean_valence_curve"]

        plot_curve(
            timesteps,
            agg["mean_reward_curve"],
            agg["std_reward_curve"],
            ylabel="Reward",
            title=f"{name} — Reward over Time",
            filename=os.path.join(FIG_DIR, f"{name}_reward_curve.png"),
        )

        plot_curve(
            timesteps,
            agg["mean_valence_curve"],
            agg["std_valence_curve"],
            ylabel="Valence",
            title=f"{name} — Valence over Time",
            filename=os.path.join(FIG_DIR, f"{name}_valence_curve.png"),
        )

    # 🔥 Combined comparison plots
    plot_comparison(
        reward_curves,
        ylabel="Reward",
        title="Baseline Comparison — Reward",
        filename=os.path.join(FIG_DIR, "baseline_reward_comparison.png"),
    )

    plot_comparison(
        valence_curves,
        ylabel="Valence",
        title="Baseline Comparison — Valence",
        filename=os.path.join(FIG_DIR, "baseline_valence_comparison.png"),
    )

    with open(os.path.join(RES_DIR, "baseline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Baseline plots saved to outputs/figures/")
    print("✓ Comparison plots generated")
    print("✓ Baseline summary saved to outputs/results/baseline_summary.json\n")


if __name__ == "__main__":
    main()
