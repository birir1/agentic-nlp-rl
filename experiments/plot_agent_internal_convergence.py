import os
import json
import matplotlib.pyplot as plt

DATA_FILE = "outputs/digital_twin/digital_twin_simulation.json"
OUTPUT_DIR = "outputs/digital_twin/plots"


def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def plot_agent_series(series, ylabel, title, filename):
    agent_ids = series[0].keys()

    plt.figure(figsize=(7, 4))
    for aid in agent_ids:
        values = [step[aid] for step in series]
        plt.plot(values, label=aid)

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[✓] Saved: {filename}")


def plot_global_series(values, ylabel, title, filename):
    plt.figure(figsize=(7, 4))
    plt.plot(values, linewidth=2)

    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[✓] Saved: {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data()

    # ---- Valence convergence (agent-level)
    if "valence" in data:
        plot_agent_series(
            series=data["valence"],
            ylabel="Valence",
            title="Affective Alignment Across Agents",
            filename=os.path.join(OUTPUT_DIR, "agent_valence_convergence.png"),
        )

    # ---- Arousal convergence (agent-level)
    if "arousal" in data:
        plot_agent_series(
            series=data["arousal"],
            ylabel="Arousal",
            title="Agent Arousal Stabilization",
            filename=os.path.join(OUTPUT_DIR, "agent_arousal_convergence.png"),
        )

    # ---- Cooperation score (global)
    if "coop_scores" in data:
        plot_global_series(
            values=data["coop_scores"],
            ylabel="Mean Valence",
            title="Global Cooperation Convergence",
            filename=os.path.join(OUTPUT_DIR, "global_cooperation_convergence.png"),
        )

    # ---- Implicit coordination proxy (mean valence gradient)
    if "valence" in data:
        mean_valence = [
            sum(step.values()) / len(step) for step in data["valence"]
        ]
        plot_global_series(
            values=mean_valence,
            ylabel="Mean Valence",
            title="Implicit Coordination Signal (Affect-Driven)",
            filename=os.path.join(OUTPUT_DIR, "implicit_coordination_signal.png"),
        )


if __name__ == "__main__":
    main()
