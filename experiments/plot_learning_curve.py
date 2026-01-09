import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("outputs/results/learning_rewards.npy")

plt.figure(figsize=(6, 4))
plt.plot(rewards, alpha=0.4, label="Episode Reward")
plt.plot(
    np.convolve(rewards, np.ones(20)/20, mode="valid"),
    label="Moving Average (20)"
)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Affective Agent Performance")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/figures/learning_curve.png", dpi=200)
plt.show()
