# experiments/ablation_no_affect.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ablation_no_affect.png")


# ---------------------------------------------------------------------
def run_trial(steps=50, disable_affect=False, seed=42):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=(0.9, 0.9),
        seed=seed,
        max_steps=steps
    )

    distances = []
    coop_scores = []

    for _ in range(steps):
        obs = env.step()

        # --- affect ablation ---
        if disable_affect:
            for aid in env.agent_ids:
                env.valence[aid] = 0.0
                env.arousal[aid] = 0.0
                env.weights[aid] = 0.5  # fixed policy weight

        goal = np.array(env.goal_pos)

        step_dist = []
        for pos in obs["positions"].values():
            pos = np.array(pos)
            step_dist.append(np.linalg.norm(pos - goal))

        distances.append(np.mean(step_dist))
        coop_scores.append(obs["coop_score"])

    return np.array(distances), np.array(coop_scores)


# ---------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = 50

    dist_affect, coop_affect = run_trial(
        steps=steps, disable_affect=False, seed=42
    )

    dist_no_affect, coop_no_affect = run_trial(
        steps=steps, disable_affect=True, seed=42
    )

    # ------------------ Plot ------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Distance to goal
    ax[0].plot(dist_affect, label="With Affect")
    ax[0].plot(dist_no_affect, "--", label="No Affect")
    ax[0].set_title("Convergence to Goal")
    ax[0].set_xlabel("Timestep")
    ax[0].set_ylabel("Mean Distance")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # Cooperation
    ax[1].plot(coop_affect, label="With Affect")
    ax[1].plot(coop_no_affect, "--", label="No Affect")
    ax[1].set_title("Cooperation Dynamics")
    ax[1].set_xlabel("Timestep")
    ax[1].set_ylabel("Mean Valence")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.suptitle("Ablation Study: Removing Affective Dynamics")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved ablation plot: {OUTPUT_FILE}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
