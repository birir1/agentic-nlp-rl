"""
2D Visualization of Multi-Agent Affective States in the Multi-Agent Text Environment
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from envs.multi_agent_text_env import MultiAgentTextEnv
from agents.baselines import RandomAgent, RuleBasedAffectiveAgent, FrozenLLMAgent
from agents.ea_marl_agent import EAMARLAgent

# Simulation parameters
TASK_DESCRIPTION = "Plan energy reduction strategies for a smart home."
SEED = 42
MAX_STEPS = 50

AGENT_CLASSES = [
    RandomAgent,
    RuleBasedAffectiveAgent,
    FrozenLLMAgent,
    EAMARLAgent,
]

def run_simulation(agent_cls):
    """
    Run a single episode and return valence/arousal trajectories for all agents.
    """
    env = MultiAgentTextEnv(task_description=TASK_DESCRIPTION, seed=SEED)
    agents = {aid: agent_cls(aid) for aid in env.agent_ids}

    obs = env.reset()
    done = False
    step = 0

    valence_traj = {aid: [] for aid in env.agent_ids}
    arousal_traj = {aid: [] for aid in env.agent_ids}

    while not done and step < MAX_STEPS:
        actions = {aid: agents[aid].act(obs) for aid in env.agent_ids}
        obs, rewards, done, info = env.step(actions)

        for aid in env.agent_ids:
            valence_traj[aid].append(obs["affective_states"][aid]["valence"])
            arousal_traj[aid].append(obs["affective_states"][aid]["arousal"])

        step += 1

    return valence_traj, arousal_traj

def plot_agent_movements(valence_traj, arousal_traj, agent_cls_name):
    """
    Visualize agent movement based on valence (x) and arousal (y) over time.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"2D Agent Movement: {agent_cls_name}")

    colors = ['r', 'b', 'g', 'm', 'c']  # support up to 5 agents
    scatters = []
    for i, aid in enumerate(valence_traj.keys()):
        scat, = ax.plot([], [], 'o', color=colors[i % len(colors)], label=f"Agent {aid}")
        scatters.append(scat)

    ax.legend()

    def update(frame):
        for i, aid in enumerate(valence_traj.keys()):
            x = valence_traj[aid][frame]
            y = arousal_traj[aid][frame]
            scatters[i].set_data(x, y)
        return scatters

    ani = animation.FuncAnimation(fig, update, frames=len(next(iter(valence_traj.values()))),
                                  interval=300, blit=True, repeat=False)
    plt.show()

def main():
    print("=== Multi-Agent Simulation ===\n")
    for agent_cls in AGENT_CLASSES:
        print(f"Simulating {agent_cls.__name__}...")
        valence_traj, arousal_traj = run_simulation(agent_cls)
        plot_agent_movements(valence_traj, arousal_traj, agent_cls.__name__)

if __name__ == "__main__":
    main()
