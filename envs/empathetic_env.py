import torch
import random

class EmpatheticDialogueEnv:
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.states = data["states"]
        self.rewards = data["rewards"]
        self.size = len(self.states)

        self.idx = 0

    def reset(self):
        self.idx = random.randint(0, self.size - 1)
        return self.states[self.idx]

    def step(self, action: int):
        """
        Action = predicted empathy strategy (discrete)
        Reward = alignment with gold emotion
        """
        base_reward = self.rewards[self.idx].item()

        # Simple but meaningful shaping
        if action == 0:   # empathic response
            reward = base_reward
        elif action == 1: # neutral
            reward = base_reward * 0.3
        else:             # misaligned
            reward = -abs(base_reward)

        done = True  # one-step episode
        next_state = self.reset()

        return next_state, reward, done
