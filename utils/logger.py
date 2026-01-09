# utils/logger.py
import csv
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_dir="results", filename=None):
        os.makedirs(log_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multiagent_baselines_{timestamp}.csv"

        self.path = os.path.join(log_dir, filename)
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "agent",
                    "seed",
                    "total_reward",
                    "avg_valence"
                ])

    def log(self, agent_name, seed, reward, valence):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                agent_name,
                seed,
                round(float(reward), 4),
                round(float(valence), 4)
            ])
