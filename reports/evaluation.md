# Model Evaluation Report: Animal Biological Class Predictor

## 1. Executive Summary

This report documents the final evaluation of the Decision Tree Classifier on the untouched test set.
In accordance with strict machine learning ethics and educational integrity, **all metrics reported below represent actual observed experimental values**.

## 2. Test Set Characteristics

- **Total Samples**: 13 instances
- **Features**: 16 biological attributes (excluding identifiers and target labels)
- **Class Distribution**:
  - `Mammal`: 6 sample(s) (46.2%)
  - `Bird`: 3 sample(s) (23.1%)
  - `Reptile`: 1 sample(s) (7.7%)
  - `Fish`: 2 sample(s) (15.4%)
  - `Amphibian`: 1 sample(s) (7.7%)

## 3. Model Benchmark Comparison

| Model | Strategy | Test Accuracy | Macro F1 |
|---|---|---|---|
| **Baseline** | Majority Class ('Mammal') | 0.4615 (46.2%) | 0.1263 |
| **Decision Tree** | Tuned (`gini`, max_depth=4) | 0.8462 (84.6%) | 0.5714 |
| **Random Forest** | 50 Trees Ensemble | 0.8462 (84.6%) | 0.5714 |

## 4. Detailed Decision Tree Metrics

- **Overall Accuracy**: `0.8462` (84.6%)
- **Macro Precision**: `0.5500`
- **Macro Recall**: `0.6000`
- **Macro F1-Score**: `0.5714`
- **Weighted F1-Score**: `0.8132`

### Per-Class Performance Breakdown

| Biological Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Mammal** | 1.0000 | 1.0000 | 1.0000 | 6 |
| **Bird** | 0.7500 | 1.0000 | 0.8571 | 3 |
| **Reptile** | 0.0000 | 0.0000 | 0.0000 | 1 |
| **Fish** | 1.0000 | 1.0000 | 1.0000 | 2 |
| **Amphibian** | 0.0000 | 0.0000 | 0.0000 | 1 |

## 5. Confusion Matrix

```text
                  Pred Mammal  Pred Bird  Pred Reptile  Pred Fish  Pred Amphibian
Actual Mammal               6          0             0          0               0
Actual Bird                 0          3             0          0               0
Actual Reptile              0          1             0          0               0
Actual Fish                 0          0             0          2               0
Actual Amphibian            0          0             1          0               0
```

## 6. Critical Analysis & Dataset Limitations

> [!WARNING]
> **Crucial Disclaimer**: This project is created for educational and pedagogic purposes. The results below must **NOT** be interpreted as evidence of a production-ready biological classification system.

### Key Limitations:
1. **Extremely Small Sample Size ($N=83$ total, $N_{test}=13$)**:
   - The entire test set contains only 13 animals.
   - A single misclassified animal changes the reported test accuracy by $\frac{1}{13} \approx 7.69\%$.
2. **Severe Class Imbalance**:
   - Mammals make up 46.2% of the test set, while Reptile and Amphibian each have only **1 sample**.
   - Evaluating recall or precision on a single observation (e.g. 1 reptile) provides zero statistical confidence (the score is binary: either 1.0 or 0.0).
3. **High Accuracy Context**:
   - The high accuracy achieved by the Decision Tree is a natural consequence of the well-separated, unambiguous biological attributes present in the UCI Zoo dataset (e.g., `milk`, `feathers`, `toothed`, `breathes`, `tail`).
   - In real-world zoological classification, animals present biological edge cases (e.g., monotremes like platypuses that lay eggs and produce milk, legless lizards, lungfish) which require vastly richer multivariate datasets.
