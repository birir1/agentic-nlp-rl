import csv
from pathlib import Path


class MetricsLogger:
    def __init__(self, out_dir="results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def log_episode(self, agent_name, seed, trajectory):
        """
        trajectory: list of dicts with keys:
        timestep, reward, valence, arousal, dominance
        """
        path = self.out_dir / f"{agent_name}_seed{seed}.csv"

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestep", "reward", "valence", "arousal", "dominance"],
            )
            writer.writeheader()
            for row in trajectory:
                writer.writerow(row)
