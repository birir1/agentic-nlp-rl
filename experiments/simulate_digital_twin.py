# experiments/simulate_digital_twin.py

import json
import os
import numpy as np
from envs.digital_twin_env import DigitalTwinEnv



OUTPUT_DIR = "outputs/digital_twin"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "digital_twin_simulation.json")


def to_list(x):
    """Safely convert numpy arrays or lists to plain Python lists"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def run_simulation(steps=40):
    env = DigitalTwinEnv(num_agents=4, goal_pos=(0.9, 0.9), max_steps=steps)

    history = []

    for t in range(steps):
        step_data = env.step()

        history.append({
            "timestep": t,
            "positions": {
                k: to_list(v) for k, v in step_data["positions"].items()
            },
            "valence": step_data["valence"],
            "arousal": step_data["arousal"],
            "weights": step_data["weights"],
            "communications": step_data["communications"],
            "coop_score": step_data["coop_score"]
        })

    return {
        "env": "DigitalTwinEnv",
        "num_agents": env.num_agents,
        "goal_pos": to_list(env.goal_pos),
        "history": history
    }


def save_simulation(data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Simulation saved to: {OUTPUT_FILE}")


def main():
    data = run_simulation()
    save_simulation(data)


if __name__ == "__main__":
    main()
