# analysis/export_baseline_table.py

import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_DIR = Path("latex")
OUT_DIR.mkdir(exist_ok=True)

def load_latest_csv():
    files = sorted(RESULTS_DIR.glob("multiagent_baselines_*.csv"))
    if not files:
        raise FileNotFoundError("No baseline CSV found in results/")
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

def to_latex_table(df):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Agent & Reward ($\\uparrow$) & Valence ($\\uparrow$) \\\\")
    lines.append("\\midrule")

    for _, r in df.iterrows():
        reward = f"{r.reward_mean:.2f} $\\pm$ {r.reward_std:.2f}"
        valence = f"{r.valence_mean:.2f} $\\pm$ {r.valence_std:.2f}"
        lines.append(f"{r.agent} & {reward} & {valence} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Multi-agent baseline performance (mean $\\pm$ std over seeds).}")
    lines.append("\\label{tab:multiagent_baselines}")
    lines.append("\\end{table}")

    return "\n".join(lines)

def main():
    df = load_latest_csv()
    agg = aggregate(df)
    latex = to_latex_table(agg)

    out_file = OUT_DIR / "baseline_results.tex"
    with open(out_file, "w") as f:
        f.write(latex)

    print(f"LaTeX table saved to {out_file}")

if __name__ == "__main__":
    main()
