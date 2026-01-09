# experiments/simulate_digital_twin_convergent.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.digital_twin_env import DigitalTwinEnv

def run_digital_twin_sim(num_agents=4, steps=20):
    env = DigitalTwinEnv(num_agents=num_agents, max_steps=steps)
    traj_positions = []
    traj_valence = []
    traj_arousal = []
    traj_weights = []
    traj_comm = []
    traj_coop = []

    for t in range(steps):
        step_data = env.step()
        traj_positions.append(step_data["positions"])
        traj_valence.append(step_data["valence"])
        traj_arousal.append(step_data["arousal"])
        traj_weights.append(step_data["weights"])
        traj_comm.append(step_data["communications"])
        traj_coop.append(step_data["coop_score"])

        print(
            f"Timestep {t+1} positions: {step_data['positions']}\n"
            f"valence: {step_data['valence']} | coop_score: {step_data['coop_score']}\n"
            f"communications: {step_data['communications']}\n"
        )

    return traj_positions, traj_valence, traj_arousal, traj_weights, traj_comm, traj_coop


def animate_digital_twin(traj_positions, traj_valence, traj_weights, traj_comm, traj_coop,
                         goal=(0.9, 0.9), save_file="digital_twin_convergent.mp4"):

    num_agents = len(traj_positions[0])

    # Create figure with two subplots
    fig, (ax_env, ax_comm) = plt.subplots(1, 2, figsize=(14, 6))
    ax_env.set_xlim(0, 1)
    ax_env.set_ylim(0, 1)
    ax_env.set_title("Digital Twin Multi-Agent Coordination")
    ax_env.plot(goal[0], goal[1], "r*", markersize=15, label="Goal")

    # Communication bar plot
    ax_comm.set_xlim(0, len(traj_positions))
    ax_comm.set_ylim(0, num_agents*(num_agents-1))
    ax_comm.set_xlabel("Timestep")
    ax_comm.set_ylabel("Number of communications")
    ax_comm.set_title("Communication Intensity")
    comm_bar = ax_comm.bar(range(len(traj_positions)), [0]*len(traj_positions), color="blue", alpha=0.5)

    # Agent dots and labels
    agent_dots = {}
    agent_labels = {}
    for i in range(num_agents):
        aid = f"agent_{i}"
        agent_dots[aid], = ax_env.plot([], [], "o", markersize=12)
        agent_labels[aid] = ax_env.text(0, 0, "", fontsize=9, ha="left", va="bottom")

    # Communication arrows
    comm_arrows = []

    # Cooperation bar
    coop_bar = ax_env.barh(["Cooperation"], [0], color="green", alpha=0.5)

    def update(frame):
        # Clear old arrows
        for arr in comm_arrows:
            arr.remove()
        comm_arrows.clear()

        # Update agent positions, colors, and labels
        for aid, dot in agent_dots.items():
            pos = traj_positions[frame][aid]
            val = traj_valence[frame][aid]
            weight = traj_weights[frame][aid]

            # Color represents cooperation success (valence × weight)
            color_val = np.clip(val * weight, 0, 1)
            dot.set_data(pos[0], pos[1])
            dot.set_color((0.5 - 0.5*color_val, 0.5 + 0.5*color_val, 0.5))

            # Update label
            label_text = f"{aid}\nval:{val:.2f}\nw:{weight:.2f}"
            agent_labels[aid].set_position((pos[0]+0.01, pos[1]+0.01))
            agent_labels[aid].set_text(label_text)

        # Draw communications as arrows
        for sender, receiver in traj_comm[frame]:
            start = traj_positions[frame][sender]
            end = traj_positions[frame][receiver]
            arr = ax_env.arrow(
                start[0], start[1],
                (end[0]-start[0])*0.9,
                (end[1]-start[1])*0.9,
                head_width=0.02, head_length=0.02, fc="blue", ec="blue", alpha=0.5
            )
            comm_arrows.append(arr)

        # Update cooperation bar
        coop_bar[0].set_width(traj_coop[frame])

        # Update communication intensity
        for i, rect in enumerate(comm_bar):
            rect.set_height(len(traj_comm[i]))

        return list(agent_dots.values()) + list(agent_labels.values()) + comm_arrows + list(coop_bar) + list(comm_bar)

    ani = FuncAnimation(fig, update, frames=len(traj_positions), interval=800, blit=False)

    # Save animation
    try:
        writer = FFMpegWriter(fps=2, metadata=dict(artist='Agentic Simulation'))
        ani.save(save_file, writer=writer)
        print(f"Animation saved as {save_file}")
    except Exception as e:
        print(f"Could not save as MP4: {e}. Falling back to GIF.")
        ani.save("digital_twin_convergent.gif", writer="pillow", fps=2)

    plt.show()


if __name__ == "__main__":
    traj_positions, traj_valence, traj_arousal, traj_weights, traj_comm, traj_coop = run_digital_twin_sim(num_agents=4, steps=20)
    animate_digital_twin(traj_positions, traj_valence, traj_weights, traj_comm, traj_coop)
