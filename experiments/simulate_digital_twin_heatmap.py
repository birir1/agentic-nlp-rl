# experiments/simulate_digital_twin_heatmap.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import random
from scipy.stats import gaussian_kde

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

def animate_digital_twin_heatmap(traj_positions, traj_valence, traj_weights, traj_comm, traj_coop,
                                 goal=(0.9, 0.9), save_file="digital_twin_heatmap.mp4"):
    num_agents = len(traj_positions[0])
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title("Digital Twin Agents Cooperation Heatmap")
    ax.plot(goal[0], goal[1], "r*", markersize=15, label="Goal")
    
    # Agent dots, labels, and trails
    agent_dots = {}
    agent_labels = {}
    agent_trails = {f"agent_{i}": [] for i in range(num_agents)}

    for i in range(num_agents):
        aid = f"agent_{i}"
        agent_dots[aid], = ax.plot([], [], "o", markersize=12)
        agent_labels[aid] = ax.text(0,0,"",fontsize=9, ha="left", va="bottom")

    # Heatmap initialization
    heatmap = ax.imshow(np.zeros((100,100)), extent=[0,1,0,1], origin='lower', cmap="hot", alpha=0.5, vmin=0, vmax=1)

    comm_arrows = []

    def update(frame):
        # Clear arrows
        for arr in comm_arrows:
            arr.remove()
        comm_arrows.clear()

        # Collect cooperation-weighted positions for heatmap
        coop_x, coop_y, coop_intensity = [], [], []

        for aid, dot in agent_dots.items():
            pos = traj_positions[frame][aid]
            val = traj_valence[frame][aid]
            weight = traj_weights[frame][aid]
            coop_val = val * weight

            # Update dot
            color_val = np.clip(coop_val, 0, 1)
            dot.set_data(pos[0], pos[1])
            dot.set_color((0.5 - 0.5*color_val, 0.5 + 0.5*color_val, 0.5))

            # Update label
            label_text = f"{aid}\nval:{val:.2f}\nw:{weight:.2f}"
            agent_labels[aid].set_position((pos[0]+0.01, pos[1]+0.01))
            agent_labels[aid].set_text(label_text)

            # Update trail
            agent_trails[aid].append(pos.copy())
            trail_array = np.array(agent_trails[aid])
            ax.plot(trail_array[:,0], trail_array[:,1], color=(0.2, 0.8, 0.2, 0.3))  # semi-transparent trail

            # Add to heatmap data
            coop_x.append(pos[0])
            coop_y.append(pos[1])
            coop_intensity.append(coop_val)

        # Draw communication arrows
        for sender, receiver in traj_comm[frame]:
            start = traj_positions[frame][sender]
            end = traj_positions[frame][receiver]
            arr = ax.arrow(
                start[0], start[1],
                (end[0]-start[0])*0.9,
                (end[1]-start[1])*0.9,
                head_width=0.02, head_length=0.02, fc="blue", ec="blue", alpha=0.5
            )
            comm_arrows.append(arr)

        # Update heatmap using gaussian kernel density weighted by cooperation
        if coop_x:
            xy = np.vstack([coop_x, coop_y])
            kde = gaussian_kde(xy, weights=coop_intensity)
            xi, yi = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            heatmap.set_data(zi)
        return list(agent_dots.values()) + list(agent_labels.values()) + comm_arrows + [heatmap]

    ani = FuncAnimation(fig, update, frames=len(traj_positions), interval=800, blit=False)

    try:
        writer = FFMpegWriter(fps=2, metadata=dict(artist='Agentic Simulation'))
        ani.save(save_file, writer=writer)
        print(f"Animation saved as {save_file}")
    except Exception as e:
        print(f"Could not save as MP4: {e}. Falling back to GIF.")
        ani.save("digital_twin_heatmap.gif", writer="pillow", fps=2)

    plt.show()


def plot_convergence(traj_valence, traj_weights, save_file="digital_twin_convergence.png"):
    num_agents = len(traj_valence[0])
    steps = len(traj_valence)
    plt.figure(figsize=(10,6))
    for i in range(num_agents):
        aid = f"agent_{i}"
        coop_over_time = [traj_valence[t][aid]*traj_weights[t][aid] for t in range(steps)]
        plt.plot(range(1, steps+1), coop_over_time, label=f"{aid}")
    plt.xlabel("Timestep")
    plt.ylabel("Cooperation Level (valence x weight)")
    plt.title("Agents Convergence Towards Goal Cooperation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_file)
    print(f"Convergence plot saved as {save_file}")
    plt.show()


if __name__ == "__main__":
    traj_positions, traj_valence, traj_arousal, traj_weights, traj_comm, traj_coop = run_digital_twin_sim(num_agents=4, steps=20)
    animate_digital_twin_heatmap(traj_positions, traj_valence, traj_weights, traj_comm, traj_coop)
    plot_convergence(traj_valence, traj_weights)
