# envs/affect_loader.py
import csv
import random
from pathlib import Path

class AffectLoader:
    """
    Loads real affective states from MELD dataset
    """

    EMOTION_MAP = {
        "joy":       (0.8, 0.7),
        "neutral":   (0.0, 0.3),
        "anger":     (-0.7, 0.8),
        "sadness":   (-0.6, 0.4),
        "fear":      (-0.8, 0.9),
        "surprise":  (0.4, 0.8),
        "disgust":   (-0.6, 0.6),
    }

    def __init__(self, meld_path="data/raw/meld/data/MELD/train_sent_emo.csv", seed=42):
        self.meld_path = Path(meld_path)
        self.seed = seed
        random.seed(seed)

        self.samples = []
        self._load()

    def _load(self):
        if not self.meld_path.exists():
            raise FileNotFoundError(f"MELD file not found: {self.meld_path}")

        with open(self.meld_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                emotion = row["Emotion"].lower()
                if emotion in self.EMOTION_MAP:
                    self.samples.append(emotion)

        if not self.samples:
            raise RuntimeError("No valid emotions loaded from MELD.")

    def sample_affect(self):
        """
        Sample a real human affect state
        """
        emotion = random.choice(self.samples)
        valence, arousal = self.EMOTION_MAP[emotion]

        return {
            "emotion": emotion,
            "valence": valence,
            "arousal": arousal
        }

    def batch_affect(self, n):
        """
        Sample a batch for multi-agent steps
        """
        return [self.sample_affect() for _ in range(n)]
