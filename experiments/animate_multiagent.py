"""
Animated Multi-Agent Simulation with Affectiveness
"""

import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.baselines import BaseAgent, RandomAgent
from envs.multi_agent_text_env import MultiAgentTextEnv

# -------------------------
# Constants
# -------------------------
TASK_DESCRIPTION = "Collaboratively optimize energy usage in a smart grid."
SEED = 42
MAX_STEPS = 10
NUM_AGENTS = 4

# -------------------------
# Simulation Function
# -------------------------
def run_simulation(agent_cls, num_agents=NUM_AGENTS):
    random.seed(SEED)
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    env = MultiAgentTextEnv(task_description=TASK_DESCRIPTION, agent_ids=agent_ids, max_steps=MAX_STEPS, seed=SEED)
    agents = {aid: agent_cls(aid) for aid in agent_ids}

    positions = {aid: [random.uniform(0,1), random.uniform(0,1)] for aid in agent_ids}
    weights = {aid: 1.0 for aid in agent_ids}
    affect_goals = {aid: {"valence": 1.0, "arousal": 0.5} for aid in agent_ids}

    valence_traj = []
    weights_traj = []
    pos_traj = []
    messages_traj = []

    obs = env.reset()
    for t in range(MAX_STEPS):
        actions = {}
        # Pre-action weight adjustment
        for aid in agent_ids:
            state = obs["affective_states"][aid]
            weights[aid] = 1.0 - 0.5 * min(1.0, abs(state["valence"] - affect_goals[aid]["valence"]))

        # Agents act
        for aid in agent_ids:
            act = agents[aid].act(obs)
            act["affect_delta"]["valence"] *= weights[aid]
            act["affect_delta"]["arousal"] *= weights[aid]
            actions[aid] = act

        # Step env
        obs, rewards, done, info = env.step(actions)

        # Move agents
        for aid in agent_ids:
            dx, dy = random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05)
            positions[aid][0] = min(max(0, positions[aid][0] + dx), 1)
            positions[aid][1] = min(max(0, positions[aid][1] + dy), 1)

        # Log
        valence_traj.append({aid: obs["affective_states"][aid]["valence"] for aid in agent_ids})
        weights_traj.append(weights.copy())
        pos_traj.append({aid: positions[aid][:] for aid in agent_ids})
        messages_traj.append([actions[aid]["message"] for aid in agent_ids])

    return valence_traj, weights_traj, pos_traj, messages_traj, agent_ids


# -------------------------
# Animation
# -------------------------
def animate_simulation(valence_traj, weights_traj, pos_traj, messages_traj, agent_ids):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    scatters = {aid: ax.scatter([], [], s=100, label=aid) for aid in agent_ids}
    texts = {aid: ax.text(0,0,'') for aid in agent_ids}
    ax.set_title("Multi-Agent Simulation")
    ax.legend()

    def update(frame):
        for aid in agent_ids:
            x, y = pos_traj[frame][aid]
            # Size = weight, Color = valence
            size = 100 + 100*weights_traj[frame][aid]
            color = max(0, min(1, valence_traj[frame][aid]))
            scatters[aid].set_offsets([x,y])
            scatters[aid].set_sizes([size])
            scatters[aid].set_color([[1-color, 1, 1-color]])  # Cyan-to-white colormap
            texts[aid].set_position((x+0.02, y+0.02))
            texts[aid].set_text(messages_traj[frame][agent_ids.index(aid)])
        ax.set_title(f"Timestep {frame+1}")
        return list(scatters.values()) + list(texts.values())

    ani = FuncAnimation(fig, update, frames=len(pos_traj), interval=1000, blit=False, repeat=False)
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    valence_traj, weights_traj, pos_traj, messages_traj, agent_ids = run_simulation(RandomAgent, NUM_AGENTS)
    animate_simulation(valence_traj, weights_traj, pos_traj, messages_traj, agent_ids)


if __name__ == "__main__":
    main()
