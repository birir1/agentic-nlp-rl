# experiments/compare_communication_convergence.py

import os
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "communication_vs_no_communication.png"
)


def run_trial(steps=50, communication=True, seed=42):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=seed,
        communication_enabled=communication,
    )

    goal = np.array(env.goal_pos)
    distances = []

    for _ in range(steps):
        obs = env.step()
        step_dist = []

        for pos in obs["positions"].values():
            pos = np.array(pos)
            step_dist.append(np.linalg.norm(pos - goal))

        distances.append(np.mean(step_dist))

    return np.array(distances)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = 50

    with_comm = run_trial(steps, communication=True, seed=42)
    no_comm = run_trial(steps, communication=False, seed=42)

    plt.figure(figsize=(7, 4))
    plt.plot(with_comm, label="With Communication", linewidth=2)
    plt.plot(
        no_comm,
        linestyle="--",
        label="No Communication",
        linewidth=2,
    )

    plt.xlabel("Timestep")
    plt.ylabel("Mean Distance to Goal")
    plt.title("Causal Effect of Communication on Multi-Agent Convergence")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
