# experiments/plot_digital_twin_convergence.py

import json
import os
import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = "outputs/digital_twin/digital_twin_simulation.json"
SAVE_DIR = "outputs/digital_twin/plots"


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Simulation file not found: {DATA_PATH}")

    with open(DATA_PATH, "r") as f:
        return json.load(f)


def get_goal_position(data):
    # Preferred: explicitly stored goal
    if "goal_pos" in data:
        return np.array(data["goal_pos"])

    # Fallback: infer from final agent positions (mean convergence point)
    print("[!] goal_pos not found — inferring goal from final positions")
    final_positions = data["history"][-1]["positions"]
    inferred_goal = np.mean(
        [np.array(p) for p in final_positions.values()],
        axis=0
    )
    return inferred_goal


def plot_agent_convergence(data):
    os.makedirs(SAVE_DIR, exist_ok=True)

    history = data["history"]
    agent_ids = history[0]["positions"].keys()
    goal = get_goal_position(data)

    plt.figure(figsize=(8, 5))

    for aid in agent_ids:
        distances = []
        for step in history:
            pos = np.array(step["positions"][aid])
            distances.append(np.linalg.norm(goal - pos))

        plt.plot(distances, label=aid)

    plt.xlabel("Time step")
    plt.ylabel("Distance to goal")
    plt.title("Agent Convergence Toward Shared Goal")
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(SAVE_DIR, "agent_convergence.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[✓] Saved: {out_path}")


def plot_cooperation(data):
    history = data["history"]
    coop = [step["coop_score"] for step in history]

    plt.figure(figsize=(8, 5))
    plt.plot(coop, linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("Cooperation score")
    plt.title("Global Cooperation Convergence")
    plt.grid(True)

    out_path = os.path.join(SAVE_DIR, "cooperation_convergence.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[✓] Saved: {out_path}")


def main():
    data = load_data()
    plot_agent_convergence(data)
    plot_cooperation(data)
    print("[✓] All plots generated successfully")


if __name__ == "__main__":
    main()
