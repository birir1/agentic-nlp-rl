# experiments/compare_goal_positions.py

import os
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "goal_position_robustness.png")


# ---------------------------------------------------------------------
def run_trial(goal_pos, steps=50, seed=0):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=goal_pos,
        seed=seed,
        max_steps=steps
    )

    distances = []

    for _ in range(steps):
        obs = env.step()
        goal = np.array(goal_pos)

        step_dist = []
        for pos in obs["positions"].values():
            pos = np.array(pos)
            step_dist.append(np.linalg.norm(pos - goal))

        distances.append(np.mean(step_dist))

    return np.array(distances)


# ---------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    goal_positions = [
        (0.2, 0.2),
        (0.8, 0.2),
        (0.2, 0.8),
        (0.8, 0.8),
    ]

    steps = 50

    plt.figure(figsize=(6, 4))

    for i, goal in enumerate(goal_positions):
        dist = run_trial(goal, steps=steps, seed=i)
        plt.plot(dist, label=f"Goal {goal}")

    plt.title("Robustness Across Goal Positions")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Distance to Goal")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved robustness plot: {OUTPUT_FILE}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
