import os
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "communication_rate_sweep.png")


def run_trial(steps=50, comm_prob=0.0, seed=0):
    """
    Run a single simulation with a given communication probability.
    Communication is filtered post-step to simulate partial coordination.
    """
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=seed
    )

    goal = np.array(env.goal_pos)
    distances = []

    for _ in range(steps):
        obs = env.step()

        # ---- simulate reduced communication ----
        if comm_prob < 1.0:
            filtered_comm = [
                pair for pair in obs["communications"]
                if np.random.rand() < comm_prob
            ]
            obs["communications"] = filtered_comm
        # ----------------------------------------

        step_distances = []
        for pos in obs["positions"].values():
            pos = np.array(pos)
            step_distances.append(np.linalg.norm(pos - goal))

        distances.append(np.mean(step_distances))

    return np.array(distances)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = 50
    comm_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
    seed = 42

    plt.figure(figsize=(7, 4))

    for p in comm_rates:
        curve = run_trial(steps=steps, comm_prob=p, seed=seed)
        plt.plot(curve, label=f"Comm prob = {p}")

    plt.xlabel("Timestep")
    plt.ylabel("Mean Distance to Goal")
    plt.title("Communication Rate vs Multi-Agent Convergence")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved communication sweep plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
