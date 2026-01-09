# load_datasets.py
# Script to load all preprocessed datasets (MELD, UltraFeedback SFT, UltraFeedback prefs)

from datasets import load_from_disk

def load_all_datasets():
    """Load all preprocessed datasets and return as a dictionary."""
    
    datasets = {}

    # MELD
    try:
        datasets["meld"] = load_from_disk("data/processed/meld")
        print(f"MELD dataset loaded: {datasets['meld']}")
    except FileNotFoundError:
        print("MELD dataset not found at 'data/processed/meld'. Please preprocess first.")

    # UltraFeedback SFT
    try:
        datasets["ultrafeedback_sft"] = load_from_disk("data/processed/ultrafeedback")
        print(f"UltraFeedback SFT dataset loaded: {datasets['ultrafeedback_sft']}")
    except FileNotFoundError:
        print("UltraFeedback SFT dataset not found at 'data/processed/ultrafeedback'. Please preprocess first.")

    # UltraFeedback Preferences
    try:
        datasets["ultrafeedback_prefs"] = load_from_disk("data/processed/ultrafeedback_prefs")
        print(f"UltraFeedback preferences dataset loaded: {datasets['ultrafeedback_prefs']}")
    except FileNotFoundError:
        print("UltraFeedback preferences dataset not found at 'data/processed/ultrafeedback_prefs'. Please preprocess first.")

    return datasets


if __name__ == "__main__":
    # Example usage
    all_ds = load_all_datasets()
    for name, ds in all_ds.items():
        print(f"{name} splits: {list(ds.keys())}")
