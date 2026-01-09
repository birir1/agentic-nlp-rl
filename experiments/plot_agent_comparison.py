"""
Plot comparison of baseline agents and EA-MARL agent
"""

import matplotlib.pyplot as plt

# Agent names
agents = ["random", "rule_affective", "frozen_llm", "ea_marl"]

# Mean rewards from your latest runs
rewards = [7.018, 9.950, 9.150, 11.675]

# Mean valence from your latest runs
valences = [0.353, 0.804, 0.754, 1.227]

# Set up the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar positions
x = range(len(agents))

# Plot Rewards
bar_width = 0.4
ax1.bar([i - bar_width/2 for i in x], rewards, width=bar_width, color='skyblue', label='Reward')

# Plot Valence
ax1.bar([i + bar_width/2 for i in x], valences, width=bar_width, color='lightcoral', label='Valence')

# Labels
ax1.set_xticks(x)
ax1.set_xticklabels(agents, fontsize=12)
ax1.set_ylabel("Value", fontsize=12)
ax1.set_title("Comparison of Agents: Reward and Valence", fontsize=14)
ax1.legend()

# Add value labels on top of bars
for i, (r, v) in enumerate(zip(rewards, valences)):
    ax1.text(i - bar_width/2, r + 0.1, f"{r:.2f}", ha='center', fontsize=10)
    ax1.text(i + bar_width/2, v + 0.05, f"{v:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
