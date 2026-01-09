import json
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

DATA_FILE = "outputs/digital_twin/digital_twin_simulation.json"
OUTPUT_DIR = "outputs/digital_twin/plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "agent_communication_graph.png")


# ---------------------------------------------------------------------
# Robust loader
# ---------------------------------------------------------------------
def load_steps(data):
    """Extract step list from multiple possible formats"""
    if isinstance(data, list):
        return data

    if "steps" in data:
        return data["steps"]

    if "trajectory" in data:
        return data["trajectory"]

    if "history" in data:
        return data["history"]

    raise KeyError(
        "Could not find steps/trajectory/history in simulation file"
    )


def load_simulation():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing simulation file: {DATA_FILE}")

    with open(DATA_FILE, "r") as f:
        raw = json.load(f)

    steps = load_steps(raw)

    if len(steps) == 0:
        raise ValueError("Simulation file contains no steps")

    return steps


# ---------------------------------------------------------------------
# Communication / coordination inference
# ---------------------------------------------------------------------
def infer_communication_edges(steps):
    agent_ids = list(steps[0]["positions"].keys())

    G = nx.Graph()
    for aid in agent_ids:
        G.add_node(aid)

    # Build trajectories
    trajectories = {
        aid: np.array([s["positions"][aid] for s in steps])
        for aid in agent_ids
    }

    # Infer coordination via motion correlation
    for i, a in enumerate(agent_ids):
        for b in agent_ids[i + 1 :]:
            da = np.linalg.norm(np.diff(trajectories[a], axis=0), axis=1)
            db = np.linalg.norm(np.diff(trajectories[b], axis=0), axis=1)

            if len(da) < 2:
                continue

            if np.std(da) == 0 or np.std(db) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(da, db)[0, 1])

            if corr > 0.25:
                G.add_edge(a, b, weight=round(corr, 2))

    return G


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_graph(G):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))

    pos = nx.spring_layout(G, seed=42)

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    widths = [2 + 4 * (w / max_w) for w in weights]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=900,
        node_color="#4C72B0",
        alpha=0.9
    )

    nx.draw_networkx_edges(
        G, pos,
        width=widths,
        edge_color="#55A868",
        alpha=0.8
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color="white"
    )

    edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title("Emergent Agent Communication / Coordination")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150)
    plt.close()

    print(f"[✓] Saved: {OUTPUT_FILE}")


# ---------------------------------------------------------------------
def main():
    steps = load_simulation()
    G = infer_communication_edges(steps)
    plot_graph(G)


if __name__ == "__main__":
    main()
