import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

OUTPUT_RESULTS = "outputs/results"
OUTPUT_FIGS = "outputs/figures"

os.makedirs(OUTPUT_RESULTS, exist_ok=True)
os.makedirs(OUTPUT_FIGS, exist_ok=True)

stats = []

def text_len(x):
    return len(x.split()) if isinstance(x, str) else 0

# ===============================
# 1) MELD (LOCAL)
# ===============================
print("Loading MELD from disk...")
meld = load_from_disk("data/processed/meld")

emotion_counts = {}
lengths_by_emotion = {}

for split in ["train", "validation", "test"]:
    for row in meld[split]:
        emotion = row["label"]
        length = text_len(row["text"])

        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        lengths_by_emotion.setdefault(emotion, []).append(length)

stats.append({
    "dataset": "MELD",
    "num_samples": sum(emotion_counts.values()),
    "avg_text_len": np.mean([l for ls in lengths_by_emotion.values() for l in ls]),
    "max_text_len": max([max(ls) for ls in lengths_by_emotion.values()]),
    "num_labels": len(emotion_counts)
})

plt.figure(figsize=(8, 4))
plt.bar(emotion_counts.keys(), emotion_counts.values())
plt.title("MELD Emotion Distribution")
plt.xlabel("Emotion Label")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGS}/meld_emotion_distribution.png")
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot(lengths_by_emotion.values(), labels=lengths_by_emotion.keys())
plt.xticks(rotation=45)
plt.title("Utterance Length by Emotion (MELD)")
plt.ylabel("Word Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGS}/meld_length_by_emotion.png")
plt.close()

# ===============================
# 2) ULTRAFEEDBACK (SFT)
# ===============================
print("Loading UltraFeedback SFT from disk...")
uf_sft = load_from_disk("data/processed/ultrafeedback")

chosen_lengths = []
score_margins = []

for row in uf_sft["train"]:
    chosen_lengths.append(text_len(row["chosen"]))
    score_margins.append(row["score_chosen"] - row["score_rejected"])

stats.append({
    "dataset": "UltraFeedback-SFT",
    "num_samples": len(uf_sft["train"]),
    "avg_text_len": np.mean(chosen_lengths),
    "max_text_len": max(chosen_lengths),
    "num_labels": "preference"
})

plt.figure(figsize=(6, 4))
plt.hist(score_margins, bins=50)
plt.title("UltraFeedback Preference Score Margin")
plt.xlabel("Score(chosen) − Score(rejected)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FIGS}/ultrafeedback_score_margin.png")
plt.close()

# ===============================
# SAVE TABLE
# ===============================
df = pd.DataFrame(stats)
df.to_csv(f"{OUTPUT_RESULTS}/dataset_stats.csv", index=False)

print("\nDataset characterization complete:")
print(df)
