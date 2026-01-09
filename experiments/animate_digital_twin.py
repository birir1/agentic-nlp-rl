# experiments/animate_digital_twin.py

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


DATA_FILE = "outputs/digital_twin/digital_twin_simulation.json"
OUTPUT_DIR = "outputs/digital_twin/animations"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "digital_twin_coordination.gif")


def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def animate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data()

    history = data["history"]
    agent_ids = list(history[0]["positions"].keys())
    goal = np.array(data["goal_pos"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Digital Twin: Multi-Agent Coordination")

    # Plot goal
    ax.scatter(goal[0], goal[1], marker="*", s=300)
    ax.text(goal[0] + 0.02, goal[1] + 0.02, "GOAL", fontsize=10, weight="bold")

    # Agent dots and labels
    dots = {}
    labels = {}
    for aid in agent_ids:
        dot, = ax.plot([], [], marker="o", markersize=10)
        dots[aid] = dot
        labels[aid] = ax.text(0, 0, aid, fontsize=9)

    # Communication lines
    comm_lines = []

    def update(frame):
        nonlocal comm_lines

        # Clear old communication lines
        for line in comm_lines:
            line.remove()
        comm_lines = []

        step = history[frame]
        positions = step["positions"]
        valence = step["valence"]
        communications = step["communications"]

        for aid in agent_ids:
            x, y = positions[aid]
            dots[aid].set_data([x], [y])

            # Color by valence (clipped)
            v = max(0.0, min(1.0, valence[aid]))
            dots[aid].set_color((1 - v, v, 0.2))

            labels[aid].set_position((x + 0.02, y + 0.02))

        # Draw communication edges
        for src, dst in communications:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            line, = ax.plot([x1, x2], [y1, y2], linewidth=1.5, alpha=0.6)
            comm_lines.append(line)

        ax.set_xlabel(f"Timestep {frame}")
        return list(dots.values()) + comm_lines + list(labels.values())

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=600,
        blit=False
    )

    ani.save(OUTPUT_FILE, writer="pillow", fps=2)
    plt.close()

    print(f"[✓] Animation saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    animate()
