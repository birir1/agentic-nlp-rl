import os
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "joint_agent_convergence.png")


def run_joint_convergence(steps=50, seed=42):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=seed,
        communication_enabled=True,
    )

    goal = np.array(env.goal_pos)
    joint_distances = []

    for _ in range(steps):
        obs = env.step()
        dists = []

        for pos in obs["positions"].values():
            pos = np.array(pos)
            dists.append(np.linalg.norm(pos - goal))

        joint_distances.append(np.mean(dists))

    return np.array(joint_distances)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = 50
    convergence = run_joint_convergence(steps)

    plt.figure(figsize=(7, 4))
    plt.plot(convergence, linewidth=2)

    plt.xlabel("Timestep")
    plt.ylabel("Mean Team Distance to Goal")
    plt.title("Joint Multi-Agent Convergence (Coordinated Behavior)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved joint convergence plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
