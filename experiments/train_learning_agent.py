import numpy as np
from tqdm import trange

from envs.text_task_env import TextTaskEnv
from agents.learning_agent import LearningAffectiveAgent

TASK_DESCRIPTION = (
    "Collaboratively plan a solution to reduce energy consumption "
    "in a smart home."
)

EPISODES = 300
MAX_STEPS = 10


def main():
    env = TextTaskEnv(
        task_description=TASK_DESCRIPTION,
        max_steps=MAX_STEPS,
        seed=0,
    )

    agent = LearningAffectiveAgent()
    episode_rewards = []

    for ep in trange(EPISODES, desc="Training"):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.act(obs)
            result = env.step(action)

            agent.observe(result.reward)

            obs = result.observation
            done = result.done
            total_reward += result.reward

        agent.update()
        episode_rewards.append(total_reward)

        if ep % 50 == 0:
            print(f"Episode {ep} | Reward: {total_reward:.3f}")

    np.save("outputs/results/learning_rewards.npy", episode_rewards)


if __name__ == "__main__":
    main()
