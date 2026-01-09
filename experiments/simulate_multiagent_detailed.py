"""
Detailed Multi-Agent Simulation with Weights, Goals, and Affectiveness
"""

import random
import matplotlib.pyplot as plt
from collections import defaultdict
from agents.baselines import BaseAgent, RandomAgent, EAMARLAgent, FrozenLLMAgent, RuleBasedAffectiveAgent
from envs.multi_agent_text_env import MultiAgentTextEnv

# -------------------------
# Constants
# -------------------------
TASK_DESCRIPTION = "Collaboratively optimize energy usage in a smart grid."
SEED = 42
MAX_STEPS = 10

# -------------------------
# Simulation Function
# -------------------------
def run_simulation(agent_cls, num_agents=4):
    random.seed(SEED)

    # Initialize environment
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    env = MultiAgentTextEnv(task_description=TASK_DESCRIPTION, agent_ids=agent_ids, max_steps=MAX_STEPS, seed=SEED)

    # Initialize agents
    agents = {aid: agent_cls(aid) for aid in agent_ids}

    # Initialize positions in 2D space
    positions = {aid: [random.uniform(0, 1), random.uniform(0, 1)] for aid in agent_ids}

    # Initialize dynamic weights (for affective influence)
    weights = {aid: 1.0 for aid in agent_ids}

    # Goals for affective states
    affect_goals = {aid: {"valence": 1.0, "arousal": 0.5} for aid in agent_ids}

    # Trajectories
    valence_traj = []
    arousal_traj = []
    rewards_traj = []
    messages_traj = []
    weights_traj = []

    obs = env.reset()
    for t in range(MAX_STEPS):
        actions = {}
        # Pre-action measurement: adjust weights based on deviation from goals
        for aid in agent_ids:
            state = obs["affective_states"][aid]
            # Simple weight: closer to goal -> smaller weight (less influence)
            weights[aid] = 1.0 - 0.5 * min(1.0, abs(state["valence"] - affect_goals[aid]["valence"]))

        # Agents act
        for aid in agent_ids:
            act = agents[aid].act(obs)
            # Apply weight to affect_delta
            act["affect_delta"]["valence"] *= weights[aid]
            act["affect_delta"]["arousal"] *= weights[aid]
            actions[aid] = act

        # Step environment
        obs, rewards, done, info = env.step(actions)

        # Update positions (simple random move to simulate movement)
        for aid in agent_ids:
            dx = random.uniform(-0.05, 0.05)
            dy = random.uniform(-0.05, 0.05)
            positions[aid][0] = min(max(0.0, positions[aid][0] + dx), 1.0)
            positions[aid][1] = min(max(0.0, positions[aid][1] + dy), 1.0)

        # Compute affectiveness: closeness to valence goal
        affectiveness = {
            aid: 1.0 - abs(obs["affective_states"][aid]["valence"] - affect_goals[aid]["valence"])
            for aid in agent_ids
        }

        # Log trajectories
        valence_traj.append({aid: obs["affective_states"][aid]["valence"] for aid in agent_ids})
        arousal_traj.append({aid: obs["affective_states"][aid]["arousal"] for aid in agent_ids})
        rewards_traj.append(rewards.copy())
        messages_traj.append([act["message"] for act in actions.values()])
        weights_traj.append(weights.copy())

        # Print timestep info
        print(f"Timestep {t+1} valence: {valence_traj[-1]} | rewards: {rewards} | affectiveness: {affectiveness}")

        if done:
            break

    return valence_traj, arousal_traj, rewards_traj, messages_traj, weights_traj, positions


# -------------------------
# Plot Function
# -------------------------
def plot_trajectories(valence_traj, arousal_traj, rewards_traj, messages_traj, weights_traj, positions):
    agent_ids = valence_traj[0].keys()
    steps = list(range(1, len(valence_traj)+1))

    # Plot Valence Trajectories
    plt.figure(figsize=(12,5))
    for aid in agent_ids:
        plt.plot(steps, [v[aid] for v in valence_traj], label=f"{aid} valence")
    plt.xlabel("Timestep")
    plt.ylabel("Valence")
    plt.title("Agent Valence over Time")
    plt.legend()
    plt.show()

    # Plot Weights Trajectories
    plt.figure(figsize=(12,5))
    for aid in agent_ids:
        plt.plot(steps, [w[aid] for w in weights_traj], label=f"{aid} weight")
    plt.xlabel("Timestep")
    plt.ylabel("Weight")
    plt.title("Agent Weights over Time")
    plt.legend()
    plt.show()

    # 2D final positions
    plt.figure(figsize=(6,6))
    for aid, pos in positions.items():
        plt.scatter(pos[0], pos[1], label=aid)
        plt.text(pos[0]+0.01, pos[1]+0.01, aid)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title("Final Agent Positions")
    plt.legend()
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    print("=== Detailed Multi-Agent Simulation with Weights & Affectiveness ===")
    agent_cls = RandomAgent
    valence_traj, arousal_traj, rewards_traj, messages_traj, weights_traj, positions = run_simulation(agent_cls, num_agents=4)
    plot_trajectories(valence_traj, arousal_traj, rewards_traj, messages_traj, weights_traj, positions)


if __name__ == "__main__":
    main()
