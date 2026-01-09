import os
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "joint_convergence_multiseed.png")


def run_episode(steps=50, seed=0):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=seed,
        communication_enabled=True,
    )

    goal = np.array(env.goal_pos)
    distances = []

    for _ in range(steps):
        obs = env.step()
        dists = [
            np.linalg.norm(np.array(pos) - goal)
            for pos in obs["positions"].values()
        ]
        distances.append(np.mean(dists))

    return np.array(distances)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = 50
    seeds = [0, 1, 2, 3, 4]

    all_runs = np.array([
        run_episode(steps=steps, seed=s) for s in seeds
    ])

    mean = all_runs.mean(axis=0)
    std = all_runs.std(axis=0)

    plt.figure(figsize=(7, 4))
    plt.plot(mean, linewidth=2, label="Mean Convergence")
    plt.fill_between(
        range(steps),
        mean - std,
        mean + std,
        alpha=0.3,
        label="±1 Std"
    )

    plt.xlabel("Timestep")
    plt.ylabel("Mean Team Distance to Goal")
    plt.title("Robust Joint Convergence Across Random Seeds")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved multi-seed convergence plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
