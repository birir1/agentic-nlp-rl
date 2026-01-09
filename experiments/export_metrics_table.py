# experiments/export_metrics_table.py

import os
import json
import numpy as np
from envs.digital_twin_env import DigitalTwinEnv

OUTPUT_DIR = "outputs/digital_twin/tables"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "agentic_results_table.tex")


# ---------------------------------------------------------------------
def run_trial(goal_pos=(0.8, 0.8), steps=50, seed=0):
    env = DigitalTwinEnv(
        num_agents=4,
        goal_pos=goal_pos,
        seed=seed,
        max_steps=steps
    )

    distances = []

    for _ in range(steps):
        obs = env.step()
        goal = np.array(goal_pos)

        step_dist = []
        for pos in obs["positions"].values():
            pos = np.array(pos)
            step_dist.append(np.linalg.norm(pos - goal))

        distances.append(np.mean(step_dist))

    return np.array(distances)


# ---------------------------------------------------------------------
def convergence_step(curve, threshold=0.2):
    for i, v in enumerate(curve):
        if v < threshold:
            return i
    return len(curve)


# ---------------------------------------------------------------------
def summarize(curve):
    return {
        "final_dist": curve[-1],
        "convergence_step": convergence_step(curve),
        "auc": np.trapz(curve)
    }


# ---------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    experiments = {
        "Central Goal": (0.8, 0.8),
        "Corner Goal": (0.2, 0.2),
        "Edge Goal": (0.8, 0.2)
    }

    rows = []

    for name, goal in experiments.items():
        curve = run_trial(goal_pos=goal, seed=0)
        stats = summarize(curve)

        rows.append((
            name,
            f"{stats['final_dist']:.3f}",
            stats["convergence_step"],
            f"{stats['auc']:.2f}"
        ))

    # ---------------- LaTeX Table ----------------
    with open(OUTPUT_FILE, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Agentic Communication Performance Summary}\n")
        f.write("\\label{tab:agentic_results}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Scenario & Final Dist. $\\downarrow$ & Conv. Step $\\downarrow$ & AUC $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")

        for r in rows:
            f.write(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"[✓] Exported LaTeX table: {OUTPUT_FILE}")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
