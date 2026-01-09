import matplotlib.pyplot as plt

agents = ["Random", "Rule-Based", "Frozen LLM", "Learning"]
rewards = [3.09, 5.50, 6.63, 4.55]

plt.figure(figsize=(6, 4))
plt.bar(agents, rewards)

plt.ylabel("Mean Episode Reward")
plt.title("Agent Performance Comparison")
plt.tight_layout()

plt.savefig("outputs/figures/baseline_vs_learning.png", dpi=200)
plt.show()
