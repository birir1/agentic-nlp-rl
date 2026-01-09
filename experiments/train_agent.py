import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import trange
from datasets import load_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("USING REAL DATASET — GoEmotions")

# =========================
# Dataset-backed Environment
# =========================
class GoEmotionsEnv:
    def __init__(self, split="train"):
        self.dataset = load_dataset("go_emotions", split=split)
        self.step_count = 0
        self.max_steps = len(self.dataset)

        # GoEmotions has 27 emotions + neutral
        self.act_dim = 28

    def reset(self):
        self.idx = 0
        self.step_count = 0
        return self._obs()

    def _obs(self):
        # Convert text to a simple embedding (placeholder: random)
        # Replace this with proper text embeddings like BERT for real training
        return torch.randn(32, device=DEVICE)

    def step(self, action):
        # Multi-label GoEmotions: reward = +1 if predicted label in true labels else -1
        true_labels = self.dataset[self.idx]["labels"]  # list of ints
        reward = 1.0 if action in true_labels else -1.0

        self.idx += 1
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._obs(), reward, done


# =========================
# Policy Network
# =========================
class Policy(nn.Module):
    def __init__(self, obs_dim=32, act_dim=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Training Loop
# =========================
def main():
    wandb.init(
        project="agentic-rl-baselines",
        name="ppo-goemotions-real-dataset",
        config={"dataset": "GoEmotions", "env": "offline-dialogue"},
    )

    env = GoEmotionsEnv()
    policy = Policy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    obs = env.reset()
    total_reward = 0.0

    for step in trange(env.max_steps, desc="RL on GoEmotions"):
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        next_obs, reward, done = env.step(action.item())

        loss = -dist.log_prob(action) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs if not done else env.reset()
        total_reward += reward

        wandb.log({"reward": reward, "step": step})

    wandb.log({"total_reward": total_reward})
    wandb.finish()


if __name__ == "__main__":
    main()
