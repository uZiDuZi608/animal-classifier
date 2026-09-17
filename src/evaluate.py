"""
src/evaluate.py
---------------
Final Evaluation on the Untouched Test Set.

Educational Focus:
1. The Golden Rule of ML Testing:
   - The test set must remain completely untouched until model development is finished.
   - We evaluate only ONCE on the test set to estimate how the model will generalize
     to unseen data.
2. Metrics Reported:
   - Overall Accuracy: fraction of correct predictions.
   - Precision: of all instances predicted as Class C, how many were actually Class C?
   - Recall: of all true instances of Class C, how many did the model detect?
   - F1-Score: harmonic mean of Precision and Recall.
   - Confusion Matrix: reveals exactly which classes were confused with which.
3. Radical Transparency & Limitations:
   - With N_test = 13, a single sample represents 1/13 = 7.69% of the accuracy.
   - Rare classes (Reptile = 1, Amphibian = 1) cannot provide statistical confidence.
   - This experiment demonstrates ML mechanics, not production viability.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier

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

def run_evaluation():
    print("=" * 70)
    print("      FINAL MODEL EVALUATION (UNTOUCHED TEST SET)")
    print("=" * 70)

    # 1. Load Test Set
    test_path = os.path.join(PROCESSED_DIR, "test.csv")
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    # 2. Load Models
    dt_model_path = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
    rf_model_path = os.path.join(MODELS_DIR, "random_forest_model.joblib")

    if not os.path.exists(dt_model_path):
        raise FileNotFoundError(f"Trained model not found at: {dt_model_path}. Run src/train.py first.")

    dt_model = joblib.load(dt_model_path)
    rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None

    # Baseline Model
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    # 3. Compute Predictions
    y_pred_base = baseline.predict(X_test)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test) if rf_model is not None else None

    # 4. Metrics Computation
    base_acc = accuracy_score(y_test, y_pred_base)
    dt_acc = accuracy_score(y_test, y_pred_dt)
    dt_prec_macro = precision_score(y_test, y_pred_dt, average="macro", zero_division=0)
    dt_rec_macro = recall_score(y_test, y_pred_dt, average="macro", zero_division=0)
    dt_f1_macro = f1_score(y_test, y_pred_dt, average="macro", zero_division=0)
    dt_f1_weighted = f1_score(y_test, y_pred_dt, average="weighted", zero_division=0)

    rf_acc = accuracy_score(y_test, y_pred_rf) if y_pred_rf is not None else None

    print(f"\n[1] TEST SET OVERVIEW:")
    print(f"    - Total Test Samples: {len(test_df)}")
    print(f"    - Test Class Frequencies:")
    for cls in TARGET_CLASSES:
        cnt = (y_test == cls).sum()
        print(f"      * {cls:<10}: {cnt} ({cnt/len(y_test)*100:.1f}%)")

    print(f"\n[2] OVERALL MODEL COMPARISON (TEST SET):")
    print(f"    {'Model':<25} | {'Test Accuracy':<15} | {'Notes':<30}")
    print("    " + "-" * 75)
    print(f"    {'Baseline (Majority Class)':<25} | {base_acc:<15.4f} | Always predicts '{baseline.classes_[np.argmax(baseline.class_prior_)]}'")
    print(f"    {'Decision Tree (Tuned)':<25} | {dt_acc:<15.4f} | Primary Model")
    if rf_acc is not None:
        print(f"    {'Random Forest (Ensemble)':<25} | {rf_acc:<15.4f} | 50 Trees Comparison")

    print(f"\n[3] DECISION TREE DETAILED TEST METRICS:")
    print(f"    - Test Accuracy:          {dt_acc:.4f} ({dt_acc*100:.1f}%)")
    print(f"    - Macro Precision:        {dt_prec_macro:.4f}")
    print(f"    - Macro Recall:           {dt_rec_macro:.4f}")
    print(f"    - Macro F1-Score:         {dt_f1_macro:.4f}")
    print(f"    - Weighted F1-Score:      {dt_f1_weighted:.4f}")

    # Per-class report
    print("\n[4] PER-CLASS CLASSIFICATION REPORT:")
    report_dict = classification_report(y_test, y_pred_dt, labels=TARGET_CLASSES, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred_dt, labels=TARGET_CLASSES, digits=4, zero_division=0)
    print(report_text)

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_dt, labels=TARGET_CLASSES)
    print("\n[5] CONFUSION MATRIX (Rows: Actual, Columns: Predicted):")
    cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in TARGET_CLASSES],
                             columns=[f"Pred {c}" for c in TARGET_CLASSES])
    print(cm_df.to_string())

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES, cbar=False)
    plt.title("Confusion Matrix: Test Set (N=13)", fontsize=13, pad=12, fontweight="bold")
    plt.xlabel("Predicted Biological Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual Biological Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\n[+] Saved confusion matrix visualization to: {cm_path}")

    # 6. Detailed Test Sample Breakdown
    test_results_df = test_df.copy()
    test_results_df["predicted_class"] = y_pred_dt
    test_results_df["correct"] = test_results_df[TARGET_COL] == test_results_df["predicted_class"]
    print("\n[6] ROW-BY-ROW TEST SAMPLE AUDIT:")
    for _, row in test_results_df.iterrows():
        status = "[CORRECT]" if row["correct"] else "[ERROR]  "
        print(f"    {status} Actual: {row[TARGET_COL]:<10} | Predicted: {row['predicted_class']:<10} | Animal: {row['animal_name']}")

    # 7. Write Comprehensive Evaluation Markdown Report
    eval_report_path = os.path.join(REPORTS_DIR, "evaluation.md")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Report: Animal Biological Class Predictor\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("This report documents the final evaluation of the Decision Tree Classifier on the untouched test set.\n")
        f.write("In accordance with strict machine learning ethics and educational integrity, **all metrics reported below represent actual observed experimental values**.\n\n")
        
        f.write("## 2. Test Set Characteristics\n\n")
        f.write(f"- **Total Samples**: {len(test_df)} instances\n")
        f.write("- **Features**: 16 biological attributes (excluding identifiers and target labels)\n")
        f.write("- **Class Distribution**:\n")
        for cls in TARGET_CLASSES:
            cnt = int((y_test == cls).sum())
            f.write(f"  - `{cls}`: {cnt} sample(s) ({cnt/len(y_test)*100:.1f}%)\n")
        f.write("\n")

        f.write("## 3. Model Benchmark Comparison\n\n")
        f.write("| Model | Strategy | Test Accuracy | Macro F1 |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Baseline** | Majority Class ('Mammal') | {base_acc:.4f} ({base_acc*100:.1f}%) | {f1_score(y_test, y_pred_base, average='macro', zero_division=0):.4f} |\n")
        f.write(f"| **Decision Tree** | Tuned (`gini`, max_depth=4) | {dt_acc:.4f} ({dt_acc*100:.1f}%) | {dt_f1_macro:.4f} |\n")
        if rf_acc is not None:
            f.write(f"| **Random Forest** | 50 Trees Ensemble | {rf_acc:.4f} ({rf_acc*100:.1f}%) | {f1_score(y_test, y_pred_rf, average='macro', zero_division=0):.4f} |\n")
        f.write("\n")

        f.write("## 4. Detailed Decision Tree Metrics\n\n")
        f.write(f"- **Overall Accuracy**: `{dt_acc:.4f}` ({dt_acc*100:.1f}%)\n")
        f.write(f"- **Macro Precision**: `{dt_prec_macro:.4f}`\n")
        f.write(f"- **Macro Recall**: `{dt_rec_macro:.4f}`\n")
        f.write(f"- **Macro F1-Score**: `{dt_f1_macro:.4f}`\n")
        f.write(f"- **Weighted F1-Score**: `{dt_f1_weighted:.4f}`\n\n")

        f.write("### Per-Class Performance Breakdown\n\n")
        f.write("| Biological Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|---|---|---|---|---|\n")
        for cls in TARGET_CLASSES:
            m = report_dict[cls]
            f.write(f"| **{cls}** | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} | {int(m['support'])} |\n")
        f.write("\n")

        f.write("## 5. Confusion Matrix\n\n")
        f.write("```text\n")
        f.write(cm_df.to_string())
        f.write("\n```\n\n")

        f.write("## 6. Critical Analysis & Dataset Limitations\n\n")
        f.write("> [!WARNING]\n")
        f.write("> **Crucial Disclaimer**: This project is created for educational and pedagogic purposes. The results below must **NOT** be interpreted as evidence of a production-ready biological classification system.\n\n")
        f.write("### Key Limitations:\n")
        f.write("1. **Extremely Small Sample Size ($N=83$ total, $N_{test}=13$)**:\n")
        f.write("   - The entire test set contains only 13 animals.\n")
        f.write("   - A single misclassified animal changes the reported test accuracy by $\\frac{1}{13} \\approx 7.69\\%$.\n")
        f.write("2. **Severe Class Imbalance**:\n")
        f.write("   - Mammals make up 46.2% of the test set, while Reptile and Amphibian each have only **1 sample**.\n")
        f.write("   - Evaluating recall or precision on a single observation (e.g. 1 reptile) provides zero statistical confidence (the score is binary: either 1.0 or 0.0).\n")
        f.write("3. **High Accuracy Context**:\n")
        f.write("   - The high accuracy achieved by the Decision Tree is a natural consequence of the well-separated, unambiguous biological attributes present in the UCI Zoo dataset (e.g., `milk`, `feathers`, `toothed`, `breathes`, `tail`).\n")
        f.write("   - In real-world zoological classification, animals present biological edge cases (e.g., monotremes like platypuses that lay eggs and produce milk, legless lizards, lungfish) which require vastly richer multivariate datasets.\n")

    print(f"[+] Written full evaluation report to: {eval_report_path}")
    print("\n[OK] Final test evaluation completed successfully.")

if __name__ == "__main__":
    run_evaluation()
