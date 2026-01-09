"""
Enhanced plot: Agent comparison over episodes
Shows Reward and Valence progression for each agent.
"""

import matplotlib.pyplot as plt
import numpy as np

# Example: Number of episodes
episodes = np.arange(1, 21)  # 20 episodes

# Simulated per-episode metrics (replace with your actual logged data if available)
agent_metrics = {
    "random": {
        "reward": np.random.normal(7, 0.5, len(episodes)),
        "valence": np.random.normal(0.35, 0.05, len(episodes)),
    },
    "rule_affective": {
        "reward": np.random.normal(9.95, 0.5, len(episodes)),
        "valence": np.random.normal(0.8, 0.05, len(episodes)),
    },
    "frozen_llm": {
        "reward": np.random.normal(9.15, 0.5, len(episodes)),
        "valence": np.random.normal(0.75, 0.05, len(episodes)),
    },
    "ea_marl": {
        "reward": np.random.normal(11.675, 0.5, len(episodes)),
        "valence": np.random.normal(1.22, 0.05, len(episodes)),
    },
}

# Set up figure
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Colors for each agent
colors = {
    "random": "skyblue",
    "rule_affective": "green",
    "frozen_llm": "orange",
    "ea_marl": "red",
}

# Plot Reward over episodes
for agent, metrics in agent_metrics.items():
    axs[0].plot(episodes, metrics["reward"], marker='o', color=colors[agent], label=agent)
axs[0].set_ylabel("Reward")
axs[0].set_title("Agent Reward over Episodes")
axs[0].legend()
axs[0].grid(True)

# Plot Valence over episodes
for agent, metrics in agent_metrics.items():
    axs[1].plot(episodes, metrics["valence"], marker='o', color=colors[agent], label=agent)
axs[1].set_ylabel("Valence")
axs[1].set_xlabel("Episode")
axs[1].set_title("Agent Valence over Episodes")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
