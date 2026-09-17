"""
src/preprocess.py
-----------------
Data Preprocessing, Feature Selection, and Leakage-Free Stratified Splitting.

Educational Focus:
1. Features vs. Labels:
   - X (Features): Biological observable characteristics (hair, feathers, milk, etc.)
   - y (Label): Biological class we wish to predict (Mammal, Bird, Reptile, Fish, Amphibian)
2. Preventing Data Leakage:
   - "animal_name" is an identifier. If included, the model could memorize animal names
     rather than learning biological patterns. It is strictly excluded from features.
   - Target labels are strictly isolated from the feature matrix X.
   - Any preprocessing statistics must be fitted solely on the training data.
3. Stratified Splitting:
   - Preserves relative class proportions across Train (70%), Validation (15%), and Test (15%).
   - Due to extreme scarcity (e.g. Amphibians have only 4 samples total), stratified
     allocation guarantees at least 1 sample in validation and 1 in test.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "zoo.data")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

COLUMN_NAMES = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne",
    "aquatic", "predator", "toothed", "backbone", "breathes",
    "venomous", "fins", "legs", "tail", "domestic", "catsize", "class_type"
]

CLASS_MAP = {
    1: "Mammal",
    2: "Bird",
    3: "Reptile",
    4: "Fish",
    5: "Amphibian"
}

FEATURE_COLUMNS = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous",
    "fins", "legs", "tail", "domestic", "catsize"
]

TARGET_COL = "class_name"

def stratified_split(df: pd.DataFrame, target_col: str, random_state: int = 42,
                     val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Performs a deterministic, stratified 3-way split (Train, Val, Test).
    Guarantees every class with at least 3 samples is represented in all 3 splits.
    """
    rng = np.random.RandomState(random_state)
    train_idx, val_idx, test_idx = [], [], []

    for cls, group in df.groupby(target_col):
        shuffled = rng.permutation(group.index.values)
        n = len(shuffled)
        
        # Calculate split sizes with guaranteed coverage for small classes
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1
        n_train = n - n_val - n_test

        train_idx.extend(shuffled[:n_train])
        val_idx.extend(shuffled[n_train:n_train + n_val])
        test_idx.extend(shuffled[n_train + n_val:])

    # Shuffle final splits with fixed seed
    train_df = df.loc[train_idx].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = df.loc[val_idx].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_idx].sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df

def run_preprocessing():
    print("=" * 70)
    print("      DATA PREPROCESSING & STRATIFIED SPLITTING")
    print("=" * 70)

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file missing at: {RAW_DATA_PATH}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load Raw Data
    df_raw = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES)
    print(f"[+] Loaded raw dataset: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    # 2. Filter to 5 target vertebrate classes
    df_filtered = df_raw[df_raw["class_type"].isin(CLASS_MAP.keys())].copy()
    df_filtered[TARGET_COL] = df_filtered["class_type"].map(CLASS_MAP)
    print(f"[+] Filtered to 5 target classes: {len(df_filtered)} samples retained")

    # 3. Explicit Data Leakage Prevention
    print("\n[!] DATA LEAKAGE PREVENTION AUDIT:")
    print(f"    - Identifier 'animal_name' excluded from predictive features: YES")
    print(f"    - Target column '{TARGET_COL}' excluded from feature matrix X: YES")
    print(f"    - Number of predictive features selected: {len(FEATURE_COLUMNS)}")
    print(f"    - Feature column ordering fixed: {FEATURE_COLUMNS}")

    # 4. Stratified 70 / 15 / 15 Split
    train_df, val_df, test_df = stratified_split(df_filtered, TARGET_COL, random_state=42)

    total_samples = len(df_filtered)
    print(f"\n[+] DATASET SPLIT SUMMARY:")
    print(f"    - Training Set:   {len(train_df):>2} samples ({len(train_df)/total_samples*100:4.1f}%)")
    print(f"    - Validation Set: {len(val_df):>2} samples ({len(val_df)/total_samples*100:4.1f}%)")
    print(f"    - Test Set:       {len(test_df):>2} samples ({len(test_df)/total_samples*100:4.1f}%)")
    print(f"    - Total:          {total_samples:>2} samples (100.0%)")

    # Verify class representation across all splits
    print("\n    Per-Class Sample Distribution Across Splits:")
    print(f"      {'Class':<12} | {'Train':>5} | {'Val':>5} | {'Test':>5} | {'Total':>5}")
    print("      " + "-" * 40)
    for cls in ["Mammal", "Bird", "Fish", "Reptile", "Amphibian"]:
        n_tr = (train_df[TARGET_COL] == cls).sum()
        n_va = (val_df[TARGET_COL] == cls).sum()
        n_te = (test_df[TARGET_COL] == cls).sum()
        n_tot = n_tr + n_va + n_te
        print(f"      {cls:<12} | {n_tr:>5} | {n_va:>5} | {n_te:>5} | {n_tot:>5}")

    # 5. Save Processed Files
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"\n[+] Saved training set to:   {train_path}")
    print(f"[+] Saved validation set to: {val_path}")
    print(f"[+] Saved test set to:       {test_path}")

    # Save preprocessing metadata
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COL,
        "target_classes": ["Mammal", "Bird", "Reptile", "Fish", "Amphibian"],
        "random_state": 42,
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "class_breakdown": {
            cls: {
                "train": int((train_df[TARGET_COL] == cls).sum()),
                "val": int((val_df[TARGET_COL] == cls).sum()),
                "test": int((test_df[TARGET_COL] == cls).sum()),
                "total": int((df_filtered[TARGET_COL] == cls).sum())
            }
            for cls in ["Mammal", "Bird", "Fish", "Reptile", "Amphibian"]
        }
    }
    meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved metadata to:       {meta_path}")

    print("\n[OK] Preprocessing and data splitting completed successfully.")

if __name__ == "__main__":
    run_preprocessing()
