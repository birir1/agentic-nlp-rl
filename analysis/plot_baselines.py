# analysis/plot_baselines.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

def load_latest_csv():
    files = sorted(RESULTS_DIR.glob("multiagent_baselines_*.csv"))
    if not files:
        raise FileNotFoundError("No baseline result CSV found.")
    return pd.read_csv(files[-1])

def aggregate(df):
    return (
        df.groupby("agent")
        .agg(
            reward_mean=("total_reward", "mean"),
            reward_std=("total_reward", "std"),
            valence_mean=("avg_valence", "mean"),
            valence_std=("avg_valence", "std"),
        )
        .reset_index()
    )

def plot_metric(df, mean_col, std_col, ylabel, filename):
    plt.figure()
    plt.bar(
        df["agent"],
        df[mean_col],
        yerr=df[std_col],
        capsize=6
    )
    plt.ylabel(ylabel)
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()

def main():
    df = load_latest_csv()
    agg = aggregate(df)

    plot_metric(
        agg,
        "reward_mean",
        "reward_std",
        "Average Episode Reward",
        "baseline_reward.png"
    )

    plot_metric(
        agg,
        "valence_mean",
        "valence_std",
        "Average Valence",
        "baseline_valence.png"
    )

    print("Saved figures:")
    print(" - figures/baseline_reward.png")
    print(" - figures/baseline_valence.png")

if __name__ == "__main__":
    main()
