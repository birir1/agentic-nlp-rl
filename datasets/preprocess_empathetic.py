import torch
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

OUT_DIR = Path("datasets/empathetic_dialogues/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[+] Loading EmpatheticDialogues (parquet export)...")
dataset = load_dataset("empathetic_dialogues", split="train")

print("[+] Loading sentence encoder...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

states = []
rewards = []
emotions = []

emotion_to_reward = {
    "joy": 1.0,
    "proud": 0.8,
    "grateful": 0.8,
    "hopeful": 0.7,
    "content": 0.6,
    "sad": -0.5,
    "angry": -0.7,
    "afraid": -0.8,
    "disappointed": -0.6,
    "anxious": -0.7,
    "lonely": -0.6,
}

print("[+] Encoding dialogue contexts...")
for sample in tqdm(dataset):
    context = " ".join(sample["context"])
    emotion = sample["emotion"]

    emb = encoder.encode(context, convert_to_tensor=True)
    reward = emotion_to_reward.get(emotion, 0.0)

    states.append(emb)
    rewards.append(reward)
    emotions.append(emotion)

torch.save({
    "states": torch.stack(states),
    "rewards": torch.tensor(rewards),
    "emotions": emotions,
}, OUT_DIR / "train.pt")

print("[✓] Saved processed dataset to", OUT_DIR)
