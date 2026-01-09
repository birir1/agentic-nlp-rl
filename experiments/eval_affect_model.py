"""
Evaluate affect model behavior and visualize emotion → valence dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

from models.affect_model import AffectModel
from utils.emotion_mapping import emotion_to_valence


def main():
    print("Loading MELD dataset...")
    meld = load_from_disk("data/processed/meld")

    model = AffectModel()

    texts = meld["train"]["text"][:200]
    labels = meld["train"]["label"][:200]

    predicted_valences = []

    for text in texts:
        out = model.predict(text)
        val = emotion_to_valence(out["probs"])
        predicted_valences.append(val)

    # ---- Plot ----
    plt.figure(figsize=(8, 4))
    plt.hist(predicted_valences, bins=30)
    plt.axvline(0.0, linestyle="--")
    plt.title("Predicted Valence Distribution (MELD)")
    plt.xlabel("Valence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figures/valence_distribution.png")
    plt.show()

    print("Saved: figures/valence_distribution.png")

    # ---- Table summary ----
    print("\nSummary Statistics")
    print(f"Mean valence: {np.mean(predicted_valences):.3f}")
    print(f"Std  valence: {np.std(predicted_valences):.3f}")
    print(f"Min  valence: {np.min(predicted_valences):.3f}")
    print(f"Max  valence: {np.max(predicted_valences):.3f}")


if __name__ == "__main__":
    main()
