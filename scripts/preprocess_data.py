import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

# Directories
raw_dir = Path("/root/workspace/agentic_nlp_rl/data/raw")
processed_dir = Path("/root/workspace/agentic_nlp_rl/data/processed")
processed_dir.mkdir(exist_ok=True, parents=True)

def preprocess_meld():
    print("Loading MELD dataset...")
    splits = ["train", "validation", "test"]
    csv_map = {
        "train": "meld/data/MELD/train_sent_emo.csv",
        "validation": "meld/data/MELD/dev_sent_emo.csv",
        "test": "meld/data/MELD/test_sent_emo.csv"
    }

    dataset_dict = {}
    for split in splits:
        df = pd.read_csv(raw_dir/csv_map[split])
        df = df[['Utterance', 'Emotion']].rename(columns={'Utterance': 'text', 'Emotion': 'label'})
        dataset_dict[split] = Dataset.from_pandas(df)

    ds = DatasetDict(dataset_dict)
    ds.save_to_disk(processed_dir / "meld")
    print(f"MELD dataset saved. Sizes: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")

def preprocess_ultrafeedback():
    print("Loading UltraFeedback datasets...")
    sft_files = {
        "train": str(raw_dir / "ultrafeedback/data/train_sft-00000-of-00001.parquet"),
        "test": str(raw_dir / "ultrafeedback/data/test_sft-00000-of-00001.parquet")
    }
    prefs_files = {
        "train": str(raw_dir / "ultrafeedback/data/train_prefs-00000-of-00001.parquet"),
        "test": str(raw_dir / "ultrafeedback/data/test_prefs-00000-of-00001.parquet")
    }

    # SFT dataset
    ds_sft = load_dataset("parquet", data_files=sft_files)
    ds_sft.save_to_disk(processed_dir / "ultrafeedback")
    print(f"UltraFeedback SFT saved. Sizes: train={len(ds_sft['train'])}, test={len(ds_sft['test'])}")

    # Preferences dataset
    ds_prefs = load_dataset("parquet", data_files=prefs_files)
    ds_prefs.save_to_disk(processed_dir / "ultrafeedback_prefs")
    print(f"UltraFeedback preferences saved. Sizes: train={len(ds_prefs['train'])}, test={len(ds_prefs['test'])}")


def load_webgpt_optional():
    webgpt_path = processed_dir / "webgpt"
    if webgpt_path.exists():
        try:
            webgpt_ds = load_from_disk(webgpt_path)
            print("WebGPT dataset loaded successfully.")
        except Exception as e:
            print(f"Failed to load WebGPT: {e}")
    else:
        print("WebGPT dataset not found, skipping.")

if __name__ == "__main__":
    preprocess_meld()
    preprocess_ultrafeedback()
    load_webgpt_optional()
    print("All preprocessing complete!")
