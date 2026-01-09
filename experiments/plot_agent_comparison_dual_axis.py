"""
Combined Reward & Valence Plot for All Agents
"""

import matplotlib.pyplot as plt
import numpy as np

# Example: Number of episodes
episodes = np.arange(1, 21)  # 20 episodes

# Example per-episode metrics (replace with actual logged metrics)
agent_metrics = {
    "random": {
        "reward": np.random.normal(7.0, 0.5, len(episodes)),
        "valence": np.random.normal(0.35, 0.05, len(episodes)),
    },
    "rule_affective": {
        "reward": np.random.normal(9.95, 0.5, len(episodes)),
        "valence": np.random.normal(0.80, 0.05, len(episodes)),
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

# Colors for agents
colors = {
    "random": "skyblue",
    "rule_affective": "green",
    "frozen_llm": "orange",
    "ea_marl": "red",
}

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Rewards on left y-axis
for agent, metrics in agent_metrics.items():
    ax1.plot(episodes, metrics["reward"], marker='o', color=colors[agent], label=f"{agent} Reward")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_title("Agent Comparison: Reward & Valence")
ax1.grid(True)

# Create a second y-axis for Valence
ax2 = ax1.twinx()
for agent, metrics in agent_metrics.items():
    ax2.plot(episodes, metrics["valence"], marker='x', linestyle='--', color=colors[agent], label=f"{agent} Valence")
ax2.set_ylabel("Valence")

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()
