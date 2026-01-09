"""
animate_multiagent_dynamic_cooperation.py
Agents dynamically adjust positions based on weights, communicate, and cooperate toward goal
"""

import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np

# Simulation parameters
NUM_AGENTS = 4
NUM_STEPS = 20
GOAL_POS = np.array([0.9, 0.9])  # Goal coordinates

np.random.seed(42)
random.seed(42)

# Initialize positions
current_pos = {f"agent_{i}": np.random.rand(2) for i in range(NUM_AGENTS)}

# Initialize affective states
valence_traj, arousal_traj, weights_traj = [], [], []
positions_traj, communications_traj, coop_score_traj = [], [], []

# Initialize agent states
valence = {f"agent_{i}": random.uniform(0, 0.5) for i in range(NUM_AGENTS)}
arousal = {f"agent_{i}": random.uniform(0, 0.5) for i in range(NUM_AGENTS)}
weights = {f"agent_{i}": 0.5 for i in range(NUM_AGENTS)}

for t in range(NUM_STEPS):
    step_pos = {}
    val_step = {}
    arousal_step = {}
    weights_step = {}
    comm = []

    # Compute movements influenced by weights and communication
    for i in range(NUM_AGENTS):
        aid = f"agent_{i}"
        # Move toward goal with weight influence
        weight_factor = weights[aid]
        direction = GOAL_POS - current_pos[aid]
        noise = np.random.normal(0, 0.02, 2)
        step = 0.1 * direction * weight_factor + noise
        current_pos[aid] = np.clip(current_pos[aid] + step, 0, 1)
        step_pos[aid] = current_pos[aid].copy()

        # Simulate communication with others
        for j in range(NUM_AGENTS):
            if j != i and random.random() < 0.6:  # 60% chance communicate
                comm.append((aid, f"agent_{j}"))
                # Communication boosts valence slightly
                valence[aid] += 0.02
                valence[f"agent_{j}"] += 0.02

        # Update affective states based on proximity to goal
        dist_to_goal = np.linalg.norm(GOAL_POS - current_pos[aid])
        valence[aid] = np.clip(1 - dist_to_goal + random.uniform(-0.05, 0.05), 0, 1)
        arousal[aid] = np.clip(0.5 + 0.5*valence[aid], 0, 1)

        # Update weights dynamically (higher valence → more influence)
        weights[aid] = np.clip(0.5 + 0.5 * valence[aid], 0, 1)

        val_step[aid] = valence[aid]
        arousal_step[aid] = arousal[aid]
        weights_step[aid] = weights[aid]

    # Global cooperation score
    coop_score = np.mean(list(valence.values()))

    # Save timestep trajectories
    positions_traj.append(step_pos)
    valence_traj.append(val_step)
    arousal_traj.append(arousal_step)
    weights_traj.append(weights_step)
    communications_traj.append(comm)
    coop_score_traj.append(coop_score)

# Setup plot
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_title("Dynamic Cooperative Multi-Agent Simulation")
goal_dot, = ax.plot(GOAL_POS[0], GOAL_POS[1], 'g*', markersize=18, label="Goal")

agent_dots, agent_labels, weight_labels = {}, {}, {}
comm_lines = []
coop_text = ax.text(0.05, 1.02, "", transform=ax.transAxes)

for i in range(NUM_AGENTS):
    aid = f"agent_{i}"
    x, y = positions_traj[0][aid]
    dot, = ax.plot(x, y, 'o', color='gray', markersize=12)
    label = ax.text(x+0.02, y+0.02, f"{i}", fontsize=10, weight='bold')
    w_label = ax.text(x+0.02, y-0.05, f"{weights_traj[0][aid]:.2f}", fontsize=8, color='blue')
    agent_dots[aid] = dot
    agent_labels[aid] = label
    weight_labels[aid] = w_label

def update(frame):
    # Clear previous communication arrows
    for line in comm_lines:
        line.remove()
    comm_lines.clear()

    pos = positions_traj[frame]
    val = valence_traj[frame]
    arousal = arousal_traj[frame]
    weights = weights_traj[frame]
    comm = communications_traj[frame]
    coop_score = coop_score_traj[frame]

    for i in range(NUM_AGENTS):
        aid = f"agent_{i}"
        x, y = pos[aid]
        v = np.clip(val[aid], 0, 1)
        agent_dots[aid].set_data([x], [y])
        agent_dots[aid].set_color((1-v, v, 0.3))  # Red->Green by valence
        agent_labels[aid].set_position((x+0.02, y+0.02))
        weight_labels[aid].set_position((x+0.02, y-0.05))
        weight_labels[aid].set_text(f"{weights[aid]:.2f}")

    # Draw communication arrows
    for sender, receiver in comm:
        x1, y1 = pos[sender]
        x2, y2 = pos[receiver]
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>', color='blue', alpha=0.4,
                                mutation_scale=15)
        ax.add_patch(arrow)
        comm_lines.append(arrow)

    coop_text.set_text(f"Cooperation Score: {coop_score:.2f}")
    ax.set_xlabel(f"Timestep {frame+1}")
    return list(agent_dots.values()) + list(agent_labels.values()) + list(weight_labels.values()) + comm_lines + [coop_text]

ani = animation.FuncAnimation(
    fig, update, frames=NUM_STEPS, interval=1000, blit=False, repeat=False
)

# Save robustly
try:
    ani.save("multiagent_dynamic_cooperation.mp4", writer="ffmpeg", fps=2)
    print("Animation saved as multiagent_dynamic_cooperation.mp4")
except:
    ani.save("multiagent_dynamic_cooperation.gif", writer="pillow", fps=2)
    print("Animation saved as multiagent_dynamic_cooperation.gif")

plt.legend()
plt.show()
