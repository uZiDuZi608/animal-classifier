"""
src/train.py
------------
Model Training, Validation, Hyperparameter Tuning, and Tree Visualization.

Educational Focus:
1. Supervised Learning:
   - Model learns the mapping f(X) -> y from labeled training examples.
   - The tree does NOT contain handcoded rules (e.g. "if hair == 1: return Mammal").
     Instead, it calculates information gain / Gini impurity reduction to choose splits.
2. Parameters vs. Hyperparameters:
   - Hyperparameters: Set before training (e.g., max_depth, criterion, min_samples_split).
   - Parameters: Learned during training (split features, threshold values, leaf class counts).
3. Baseline Model:
   - A dummy classifier (majority class) to answer: "Does the model beat dumb guessing?"
4. Validation & Tuning:
   - Compare hyperparameter configs using the validation set to pick the best model.
   - Test set is NEVER touched during training or tuning.
5. Overfitting Check:
   - Compare training score vs. validation score.
6. Optional Comparative Model:
   - Random Forest (ensemble of trees) compared against single Decision Tree.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

FEATURE_COLUMNS = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous",
    "fins", "legs", "tail", "domestic", "catsize"
]
TARGET_COL = "class_name"
TARGET_CLASSES = ["Mammal", "Bird", "Reptile", "Fish", "Amphibian"]

def train_and_tune():
    print("=" * 70)
    print("      MODEL TRAINING, VALIDATION & HYPERPARAMETER TUNING")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load Processed Splits
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path = os.path.join(PROCESSED_DIR, "val.csv")

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Processed datasets missing. Run src/preprocess.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COL]

    print(f"[+] Loaded Training set:   {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"[+] Loaded Validation set: {X_val.shape[0]} samples")

    # 2. Baseline Model (Majority Class Dummy Classifier)
    print("\n[1] BASELINE MODEL (Majority Class):")
    print("    Answers: 'What score do we get if we simply guess the most frequent class?'")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    train_base_acc = accuracy_score(y_train, baseline.predict(X_train))
    val_base_acc = accuracy_score(y_val, baseline.predict(X_val))
    print(f"    - Baseline Training Accuracy:   {train_base_acc:.4f} ({train_base_acc*100:.1f}%)")
    print(f"    - Baseline Validation Accuracy: {val_base_acc:.4f} ({val_base_acc*100:.1f}%)")
    print(f"    - Baseline predicted class:     '{baseline.classes_[np.argmax(baseline.class_prior_)]}'")

    # 3. Hyperparameter Tuning on Validation Set
    print("\n[2] HYPERPARAMETER SEARCH (Decision Tree):")
    print("    Exploring configurations across criterion, max_depth, min_samples_split, min_samples_leaf.")
    print(f"    {'Criterion':<10} | {'Max Depth':<10} | {'Min Split':<10} | {'Min Leaf':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val F1 (macro)':<12}")
    print("    " + "-" * 85)

    search_grid = []
    for criterion in ["gini", "entropy"]:
        for max_depth in [2, 3, 4, 5, None]:
            for min_samples_split in [2, 4]:
                for min_samples_leaf in [1, 2]:
                    search_grid.append({
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf
                    })

    best_score = -1.0
    best_config = None
    best_dt = None
    tuning_results = []

    for cfg in search_grid:
        dt = DecisionTreeClassifier(
            criterion=cfg["criterion"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=42
        )
        dt.fit(X_train, y_train)

        tr_acc = accuracy_score(y_train, dt.predict(X_train))
        va_acc = accuracy_score(y_val, dt.predict(X_val))
        va_f1 = f1_score(y_val, dt.predict(X_val), average="macro", zero_division=0)

        depth_str = str(cfg["max_depth"]) if cfg["max_depth"] is not None else "None"
        tuning_results.append({
            **cfg,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "val_f1_macro": va_f1
        })

        # Selection criteria: highest validation accuracy, broken by macro F1, then simpler depth
        score = (va_acc, va_f1, -(cfg["max_depth"] or 99))
        if score > (best_score if best_config else (-1, -1, -999)):
            best_score = score
            best_config = cfg
            best_dt = dt

    # Print top candidate configurations
    sorted_results = sorted(tuning_results, key=lambda x: (x["val_acc"], x["val_f1_macro"]), reverse=True)
    for r in sorted_results[:8]:
        d_str = str(r["max_depth"]) if r["max_depth"] is not None else "None"
        print(f"    {r['criterion']:<10} | {d_str:<10} | {r['min_samples_split']:<10} | {r['min_samples_leaf']:<10} | {r['train_acc']:<10.3f} | {r['val_acc']:<10.3f} | {r['val_f1_macro']:<12.3f}")

    print("\n[+] SELECTED OPTIMAL HYPERPARAMETERS (Selected via Validation Set):")
    for k, v in best_config.items():
        print(f"    - {k}: {v}")

    # Retrain/confirm final Decision Tree
    final_dt = DecisionTreeClassifier(**best_config, random_state=42)
    final_dt.fit(X_train, y_train)

    train_dt_acc = accuracy_score(y_train, final_dt.predict(X_train))
    val_dt_acc = accuracy_score(y_val, final_dt.predict(X_val))
    val_dt_f1 = f1_score(y_val, final_dt.predict(X_val), average="macro", zero_division=0)

    print(f"\n    Selected Decision Tree Performance:")
    print(f"    - Training Accuracy:   {train_dt_acc:.4f} ({train_dt_acc*100:.1f}%)")
    print(f"    - Validation Accuracy: {val_dt_acc:.4f} ({val_dt_acc*100:.1f}%)")
    print(f"    - Validation Macro F1: {val_dt_f1:.4f}")
    print(f"    - Learned Tree Depth:  {final_dt.get_depth()}")
    print(f"    - Number of Leaves:    {final_dt.get_n_leaves()}")

    # 4. Comparative Model: Random Forest (Section 31)
    print("\n[3] COMPARATIVE MODEL (Random Forest Classifier):")
    rf = RandomForestClassifier(n_estimators=50, max_depth=best_config["max_depth"], random_state=42)
    rf.fit(X_train, y_train)

    train_rf_acc = accuracy_score(y_train, rf.predict(X_train))
    val_rf_acc = accuracy_score(y_val, rf.predict(X_val))
    val_rf_f1 = f1_score(y_val, rf.predict(X_val), average="macro", zero_division=0)

    print(f"    - RF Training Accuracy:   {train_rf_acc:.4f} ({train_rf_acc*100:.1f}%)")
    print(f"    - RF Validation Accuracy: {val_rf_acc:.4f} ({val_rf_acc*100:.1f}%)")
    print(f"    - RF Validation Macro F1: {val_rf_f1:.4f}")

    # Cross-validation insight on training data (Stratified 3-fold)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores_dt = cross_val_score(final_dt, X_train, y_train, cv=cv, scoring="accuracy")
    cv_scores_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")
    print("\n[+] 3-Fold Stratified Cross-Validation on Training Data:")
    print(f"    - Decision Tree CV Accuracy: {cv_scores_dt.mean():.4f} +/- {cv_scores_dt.std():.4f}")
    print(f"    - Random Forest CV Accuracy: {cv_scores_rf.mean():.4f} +/- {cv_scores_rf.std():.4f}")

    # 5. Overfitting Diagnostic
    print("\n[4] OVERFITTING DIAGNOSTIC:")
    if train_dt_acc - val_dt_acc > 0.15:
        print("    [!] WARNING: Noticeable gap between training and validation accuracy.")
        print("        Decision trees are prone to memorizing small datasets if left unconstrained.")
    else:
        print("    [OK] Training and validation accuracies are closely aligned, suggesting reasonable generalization.")

    # 6. Save Model Artifacts
    model_save_path = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
    joblib.dump(final_dt, model_save_path)
    print(f"\n[+] Saved trained model to: {model_save_path}")

    # Also save Random Forest for comparison if needed
    rf_save_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")
    joblib.dump(rf, rf_save_path)

    # Save metadata
    model_meta = {
        "model_type": "DecisionTreeClassifier",
        "hyperparameters": best_config,
        "feature_names": FEATURE_COLUMNS,
        "classes": list(final_dt.classes_),
        "tree_depth": int(final_dt.get_depth()),
        "leaf_count": int(final_dt.get_n_leaves()),
        "train_accuracy": float(train_dt_acc),
        "val_accuracy": float(val_dt_acc),
        "val_macro_f1": float(val_dt_f1),
        "baseline_val_accuracy": float(val_base_acc),
        "random_forest_val_accuracy": float(val_rf_acc)
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=4)
    print(f"[+] Saved model metadata to: {meta_path}")

    # 7. Decision Tree Visualization (Text & Graphical)
    print("\n[5] DECISION TREE VISUALIZATION:")
    text_tree = export_text(final_dt, feature_names=FEATURE_COLUMNS)
    print("    Textual Tree Structure:")
    for line in text_tree.strip().split("\n"):
        print(f"      {line}")

    text_tree_path = os.path.join(REPORTS_DIR, "decision_tree_structure.txt")
    with open(text_tree_path, "w", encoding="utf-8") as f:
        f.write(text_tree)
    print(f"\n[+] Saved textual tree structure to: {text_tree_path}")

    # Graphical Tree Plot
    plt.figure(figsize=(14, 8), dpi=300)
    plot_tree(
        final_dt,
        feature_names=FEATURE_COLUMNS,
        class_names=list(final_dt.classes_),
        filled=True,
        rounded=True,
        fontsize=9,
        precision=2
    )
    plt.title(f"Learned Decision Tree (Criterion: {best_config['criterion']}, Max Depth: {best_config['max_depth']})",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plot_path = os.path.join(FIGURES_DIR, "decision_tree.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Saved graphical tree visualization to: {plot_path}")

    print("\n[OK] Training, tuning, and visualization completed successfully.")

if __name__ == "__main__":
    train_and_tune()
