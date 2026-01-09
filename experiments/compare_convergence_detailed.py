import os
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "convergence_with_vs_without_comm.png")


def run_trial(steps=60, seed=0):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=seed
    )

    agent_ids = env.agent_ids
    n_agents = len(agent_ids)
    distances = np.zeros((steps, n_agents))
    goal = np.array(env.goal_pos)

    for t in range(steps):
        obs = env.step()
        for i, aid in enumerate(agent_ids):
            pos = np.array(obs["positions"][aid])
            distances[t, i] = np.linalg.norm(pos - goal)

    return distances


def aggregate_trials(n_trials=5, steps=60):
    all_runs = []
    for seed in range(n_trials):
        dist = run_trial(steps=steps, seed=seed)
        all_runs.append(dist.mean(axis=1))

    all_runs = np.stack(all_runs)
    return all_runs.mean(axis=0), all_runs.std(axis=0)


def plot_convergence(mean_dist, std_dist):
    steps = len(mean_dist)
    x = np.arange(steps)

    plt.figure(figsize=(7, 4))
    plt.plot(x, mean_dist, label="Coordinated agents", linewidth=2)
    plt.fill_between(
        x,
        mean_dist - std_dist,
        mean_dist + std_dist,
        alpha=0.3
    )

    plt.xlabel("Timestep")
    plt.ylabel("Mean distance to goal")
    plt.title("Multi-Agent Convergence with Coordination")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved: {OUTPUT_FILE}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mean_dist, std_dist = aggregate_trials()
    plot_convergence(mean_dist, std_dist)


if __name__ == "__main__":
    main()
