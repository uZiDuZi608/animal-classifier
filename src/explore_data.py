"""
src/explore_data.py
-------------------
Exploratory Data Analysis (EDA) and Data Health Inspection.

Educational Focus:
- Before training ANY machine learning model, we must thoroughly inspect the data.
- We check:
  1. Data shape (rows, columns)
  2. Data types
  3. Missing values (nulls)
  4. Duplicate records (both exact rows and duplicate feature signatures)
  5. Potential data leakage (identifiers like animal names)
  6. Class distribution and class imbalance
  7. Biological consistency (checking for conflicting labels on identical features)
- We also generate informative visualizations for understanding.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_DIR, "data", "raw", "zoo.data")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

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
    5: "Amphibian",
    6: "Insect",
    7: "Invertebrate"
}

TARGET_CLASSES = ["Mammal", "Bird", "Reptile", "Fish", "Amphibian"]

def inspect_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}. Please run src/download_data.py first.")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("=" * 70)
    print("      EXPLORATORY DATA ANALYSIS & DATA HEALTH INSPECTION")
    print("=" * 70)

    # 1. Load Data
    df_raw = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMN_NAMES)
    df_raw["class_name"] = df_raw["class_type"].map(CLASS_MAP)

    total_rows, total_cols = df_raw.shape
    print(f"\n[1] DATASET DIMENSIONS (Raw UCI Zoo Dataset):")
    print(f"    - Total Rows (Instances): {total_rows}")
    print(f"    - Total Columns: {total_cols} (including animal_name, 16 features, and class_type)")

    # 2. Inspect Column Types & Missing Values
    print(f"\n[2] MISSING VALUE & DATA TYPE AUDIT:")
    null_counts = df_raw.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"    - Total Missing (NaN) Values: {total_nulls}")
    if total_nulls > 0:
        print(null_counts[null_counts > 0])
    else:
        print("    - Clean: Zero missing values detected across all columns.")

    print("\n    Column Schemas:")
    for col in df_raw.columns:
        print(f"      * {col:<15}: dtype={str(df_raw[col].dtype):<8} unique_values={df_raw[col].nunique()}")

    # 3. Duplicate Audit
    print(f"\n[3] DUPLICATE AUDIT:")
    exact_duplicates = df_raw.duplicated().sum()
    print(f"    - Exact row duplicates (all columns including name): {exact_duplicates}")

    duplicate_names = df_raw[df_raw["animal_name"].duplicated(keep=False)]
    print(f"    - Duplicate animal names count: {len(duplicate_names)}")
    if len(duplicate_names) > 0:
        print("      Note on duplicate names:")
        for idx, row in duplicate_names.iterrows():
            print(f"        Row {idx}: {row['animal_name']} | venomous={row['venomous']} | class={row['class_name']}")
        print("      Finding: 'frog' appears twice, but represents two biological variants (one non-venomous, one venomous).")

    feature_cols = [c for c in COLUMN_NAMES if c not in ["animal_name", "class_type"]]
    dup_features = df_raw.duplicated(subset=feature_cols, keep=False).sum()
    print(f"    - Rows sharing identical feature vectors (excluding name & label): {dup_features}")
    print("      Biological Rationale: Multiple distinct animals share identical attributes in this 16-feature space")
    print("      (e.g., cheetah, leopard, lion, and wolf all share the same binary mammalian carnivore profile).")

    # Check for label contradictions on identical features
    grouped_conflicts = df_raw.groupby(feature_cols)["class_type"].nunique()
    conflicts = grouped_conflicts[grouped_conflicts > 1]
    print(f"    - Contradictory feature vectors (identical features mapped to different classes): {len(conflicts)}")
    if len(conflicts) == 0:
        print("      Clean: No contradictory labels exist for identical feature vectors.")

    # 4. Filter to Target 5 Classes
    df_5class = df_raw[df_raw["class_name"].isin(TARGET_CLASSES)].copy()
    print(f"\n[4] FILTERING TO 5 TARGET BIOLOGICAL CLASSES:")
    print(f"    - Target classes: {TARGET_CLASSES}")
    print(f"    - Filtered instances: {len(df_5class)} out of {len(df_raw)} (excluding Insects and Invertebrates)")

    class_dist = df_5class["class_name"].value_counts()[TARGET_CLASSES]
    class_pct = (class_dist / len(df_5class)) * 100

    print("\n    Class Distribution:")
    for cls in TARGET_CLASSES:
        cnt = class_dist[cls]
        pct = class_pct[cls]
        print(f"      * {cls:<12}: {cnt:>2} samples ({pct:>5.1f}%)")

    # 5. Honest Limitations and Imbalance Warning
    print(f"\n[!] CRITICAL DATASET LIMITATIONS (EDUCATIONAL HONESTY):")
    print("    1. Extremely small sample size (Total N = 83 across 5 classes).")
    print("    2. Severe class imbalance: Mammals (41) and Birds (20) account for 73.5% of the data.")
    print("    3. Reptiles (5) and Amphibians (4) have critically few observations.")
    print("    4. In a small test split (~13 samples), 1 sample represents ~7.7% accuracy shift.")
    print("    5. This dataset is suitable for teaching the mechanics of the ML pipeline,")
    print("       NOT as a production-grade biological classification engine.")

    # 6. Generate Visualizations
    # Figure A: Class Distribution Plot
    plt.figure(figsize=(9, 5))
    palette = ["#2b5c8f", "#3690c0", "#67a9cf", "#a6bddb", "#ece7f2"]
    ax = sns.barplot(x=class_dist.index, y=class_dist.values, hue=class_dist.index, palette="Blues_r", legend=False)
    plt.title("Class Distribution (5 Target Biological Classes)", fontsize=14, pad=12, fontweight="bold")
    plt.xlabel("Biological Class", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)} ({height/len(df_5class)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                    textcoords="offset points")
    plt.ylim(0, max(class_dist.values) + 6)
    plt.tight_layout()
    dist_fig_path = os.path.join(FIGURES_DIR, "class_distribution.png")
    plt.savefig(dist_fig_path, dpi=200)
    plt.close()
    print(f"\n[+] Saved class distribution figure to: {dist_fig_path}")

    # Figure B: Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    corr = df_5class[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="vlag", annot=True, fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
    plt.title("Feature Correlation Matrix (16 Biological Features)", fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    corr_fig_path = os.path.join(FIGURES_DIR, "correlation_matrix.png")
    plt.savefig(corr_fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved correlation heatmap to: {corr_fig_path}")

    # Save summary stats JSON for transparency
    eda_summary = {
        "raw_rows": total_rows,
        "raw_cols": total_cols,
        "target_class_rows": len(df_5class),
        "target_classes": TARGET_CLASSES,
        "class_counts": {k: int(v) for k, v in class_dist.items()},
        "duplicate_rows": int(exact_duplicates),
        "duplicate_animal_names": len(duplicate_names),
        "duplicate_feature_vectors": int(dup_features),
        "label_conflicts": int(len(conflicts)),
        "feature_names": feature_cols
    }
    summary_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=4)
    print(f"[+] Saved EDA metrics summary to: {summary_path}")
    print("\n[OK] Exploratory Data Analysis completed successfully.")

if __name__ == "__main__":
    inspect_dataset()
