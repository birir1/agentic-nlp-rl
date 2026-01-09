"""
Creative Animated Multi-Agent Simulation
"""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ensure we can import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.baselines import BaseAgent, RandomAgent
from envs.multi_agent_text_env import MultiAgentTextEnv

# --- Simulation Settings ---
NUM_AGENTS = 4
MAX_STEPS = 15
TASK_DESCRIPTION = "Optimize energy usage collaboratively."
SEED = 42

# Initialize environment
env = MultiAgentTextEnv(task_description=TASK_DESCRIPTION, agent_ids=[f"agent_{i}" for i in range(NUM_AGENTS)], max_steps=MAX_STEPS, seed=SEED)

# Initialize agents
agents = {aid: RandomAgent(aid) for aid in env.agent_ids}

# Goal position in 2D space (x,y)
goal_pos = np.array([5.0, 5.0])

# Agent positions (random start)
positions = {aid: np.random.rand(2) * 10 for aid in env.agent_ids}

# Interaction weights (initially uniform)
weights = {aid: {other: 1.0 for other in env.agent_ids if other != aid} for aid in env.agent_ids}

# Track trajectories
valence_traj = {aid: [] for aid in env.agent_ids}
arousal_traj = {aid: [] for aid in env.agent_ids}
rewards_traj = {aid: [] for aid in env.agent_ids}
positions_traj = {aid: [positions[aid].copy()] for aid in env.agent_ids}

# --- Simulation Loop ---
obs = env.reset()
for t in range(MAX_STEPS):
    actions = {}
    for aid, agent in agents.items():
        # measure "effectiveness" towards goal
        dist_to_goal = np.linalg.norm(goal_pos - positions[aid])
        effectiveness = max(0, 1 - dist_to_goal / 10)

        # action includes random message + affect delta based on weights
        affect_delta = {
            "valence": random.uniform(-0.1, 0.1) * np.mean(list(weights[aid].values())),
            "arousal": random.uniform(-0.1, 0.1),
            "dominance": 0.0,
        }

        actions[aid] = {
            "message": f"Agent {aid} communicates.",
            "affect_delta": affect_delta
        }

    # Step environment
    obs, rewards, done, info = env.step(actions)

    # Update agent positions (move toward goal + small randomness)
    for aid in env.agent_ids:
        move = (goal_pos - positions[aid]) * 0.1 + np.random.randn(2) * 0.2
        positions[aid] += move
        positions_traj[aid].append(positions[aid].copy())

        valence_traj[aid].append(env.affective_states[aid]["valence"])
        arousal_traj[aid].append(env.affective_states[aid]["arousal"])
        rewards_traj[aid].append(rewards[aid])

    # Dynamically adjust weights based on recent reward
    for aid in env.agent_ids:
        for other in env.agent_ids:
            if aid != other:
                weights[aid][other] += 0.05 * (rewards[other] - 0.5)
                weights[aid][other] = max(0.0, min(2.0, weights[aid][other]))

# --- Visualization ---
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Creative Multi-Agent Simulation")

agent_dots = {aid: ax.plot([], [], 'o', label=aid, markersize=10)[0] for aid in env.agent_ids}
goal_dot = ax.plot(goal_pos[0], goal_pos[1], 'r*', markersize=15, label="Goal")[0]

def update(frame):
    for aid in env.agent_ids:
        x, y = positions_traj[aid][frame]
        # color reflects valence, size reflects arousal
        val = valence_traj[aid][frame]
        ar = arousal_traj[aid][frame]
        agent_dots[aid].set_data(x, y)
        agent_dots[aid].set_color((0.5 - val, 0.5 + val, 0.5))
        agent_dots[aid].set_markersize(5 + ar * 10)
    return list(agent_dots.values())

ani = FuncAnimation(fig, update, frames=MAX_STEPS, interval=800, blit=True, repeat=False)
ax.legend()
plt.show()
