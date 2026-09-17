# 🐾 Animal Biological Class Predictor
### A Complete, Transparent, Beginner-Friendly Machine Learning Project from Absolute Zero

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## 1. Project Overview

Welcome! This repository contains a **real, working, end-to-end machine learning project** designed to teach the fundamental principles of machine learning from scratch.

Rather than treating machine learning as a "black box" that magically makes guesses, this project makes every single stage of the machine learning pipeline visible and verifiable:

```text
Data Acquisition
      ↓
Data Inspection & Health Check
      ↓
Preprocessing & Feature Selection (Anti-Leakage)
      ↓
Stratified Train / Validation / Test Splitting
      ↓
Baseline Model Construction
      ↓
Decision Tree Model Training (Parameter Learning)
      ↓
Validation & Hyperparameter Tuning
      ↓
Final Model Selection & Tree Visualization
      ↓
Evaluation on Untouched Test Set
      ↓
Inference (CLI & Web Interface)
```

The system classifies animals into **5 target vertebrate biological classes**:
1. **Mammal**
2. **Bird**
3. **Reptile**
4. **Fish**
5. **Amphibian**

---

## 2. What is Machine Learning?

In traditional software engineering, a programmer manually writes explicit deterministic logic:

$$\text{Input Data} + \text{Handwritten Rules} \longrightarrow \text{Answers}$$

For example, a traditional program might do:
```python
if hair == 1 and milk == 1:
    return "Mammal"
elif feathers == 1:
    return "Bird"
```
**This project does NOT do that.** That would defeat the purpose of machine learning.

In **Machine Learning**, the paradigm is inverted:

$$\text{Input Data} + \text{Known Answers} \longrightarrow \text{Learned Rules (Model)}$$

We feed historical animal observations and their known biological classes into an algorithm. The algorithm automatically determines which combinations of features best distinguish one class from another.

---

## 3. Why is this Supervised Learning?

Machine learning is broadly divided into categories based on the supervision provided during training:

1. **Supervised Learning**: Every training example comes paired with the correct answer (called a **label** or **ground truth**). The model's objective is to learn a mapping function $f(X) \approx y$.
2. **Unsupervised Learning**: The algorithm receives only features $X$ without any labels and looks for inherent patterns, clusters, or dimensionality reductions.
3. **Reinforcement Learning**: An agent interacts with an environment and learns through rewards and penalties.

**Why our project is Supervised Learning:**
Every animal in our training set has both known attributes (hair, teeth, milk, legs) and a verified biological class label (Mammal, Bird, Reptile, Fish, Amphibian) provided by zoologists. The model is supervised by these true labels.

---

## 4. Why is this Classification?

Supervised learning problems typically fall into two categories:

- **Regression**: The target $y$ is a continuous numerical quantity (e.g., predicting the weight of an animal in kilograms or body temperature).
- **Classification**: The target $y$ is a discrete category or group.
  - *Binary Classification*: Exactly two categories (e.g., Is Mammal? Yes / No).
  - *Multiclass Classification*: Three or more distinct categories where each sample belongs to exactly one class.

This project is a **Multiclass Classification** task because each animal is assigned to exactly one of 5 distinct categories:
$$\mathcal{Y} = \{\text{Mammal}, \text{Bird}, \text{Reptile}, \text{Fish}, \text{Amphibian}\}$$

---

## 5. What are the Features?

The **features** (denoted collectively as the matrix $X$) represent the measurable, observable attributes of an animal given to the model as input.

In our dataset, there are **16 predictive features**:

| Feature Name | Type | Domain | Biological Interpretation |
|---|---|---|---|
| `hair` | Boolean | $\{0, 1\}$ | Presence of hair or fur |
| `feathers` | Boolean | $\{0, 1\}$ | Presence of plumage / feathers |
| `eggs` | Boolean | $\{0, 1\}$ | Reproduces by laying eggs |
| `milk` | Boolean | $\{0, 1\}$ | Produces milk / mammary nursing |
| `airborne` | Boolean | $\{0, 1\}$ | Capable of powered flight |
| `aquatic` | Boolean | $\{0, 1\}$ | Spends extensive time in aquatic environments |
| `predator` | Boolean | $\{0, 1\}$ | Carnivorous / predatory feeding behavior |
| `toothed` | Boolean | $\{0, 1\}$ | Presence of teeth |
| `backbone` | Boolean | $\{0, 1\}$ | Vertebrate (spinal column) |
| `breathes` | Boolean | $\{0, 1\}$ | Pulmonary air respiration |
| `venomous` | Boolean | $\{0, 1\}$ | Produces toxin / venom |
| `fins` | Boolean | $\{0, 1\}$ | Presence of fins |
| `legs` | Numeric | $\{0, 2, 4, 5, 6, 8\}$ | Count of limbs |
| `tail` | Boolean | $\{0, 1\}$ | Presence of tail |
| `domestic` | Boolean | $\{0, 1\}$ | Domesticated animal species |
| `catsize` | Boolean | $\{0, 1\}$ | Approximately cat-sized or larger |

---

## 6. What is the Label?

The **label** (denoted as vector $y$) is the target attribute we want the machine learning model to predict from features $X$:

$$y \in \{\text{Mammal}, \text{Bird}, \text{Reptile}, \text{Fish}, \text{Amphibian}\}$$

During training, $y$ is provided so the algorithm can evaluate its errors and update its parameters. During testing and prediction on new animals, $y$ is hidden or unknown, and the model must infer it.

---

## 7. What is the Model?

The **model** is the mathematical structure or computational object that embodies the relationship learned from the data.

In this project, our primary model is a **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`).

```text
                  Does the animal produce milk?
                         /              \
                   [YES]                 [NO]
                   /                        \
              Mammal                    Has teeth?
                                       /          \
                                  [NO]            [YES]
                                  /                  \
                               Bird             Breathes air?
                                                /           \
                                            [NO]             [YES]
                                            /                   \
                                         Fish                 Has tail?
                                                              /       \
                                                           [NO]       [YES]
                                                           /             \
                                                    Amphibian         Reptile
```

### Why a Decision Tree?
1. **White-Box Interpretability**: Unlike deep neural networks, a decision tree's entire logic can be directly inspected, visualized, and audited.
2. **Multiclass Support**: Handles multi-category classification naturally without requiring multiple binary wrappers.
3. **Educational Transparency**: It clearly demonstrates how data is partitioned into purer subsets using information theory.

---

## 8. What are Parameters?

**Parameters** are the internal variables that the model **learns automatically from the data during training**.
The human engineer does *not* set these.

In a Decision Tree, the parameters are:
- Which feature to split on at each node (e.g., `milk`, `toothed`, `breathes`, `tail`).
- The numerical threshold value for each split (e.g., $\le 0.50$ vs. $> 0.50$).
- The class distribution and leaf assignments at each terminal node.

In linear regression, parameters are the weights $w$ and bias $b$. In decision trees, parameters are the tree's split structure.

---

## 9. What are Hyperparameters?

**Hyperparameters** are configuration knobs that **we (the engineers) set before training begins**.
The model cannot learn them directly via standard gradient descent or split criteria; they control *how* the model learns.

Key hyperparameters tuned in this project:

| Hyperparameter | Values Tested | Role |
|---|---|---|
| `criterion` | `'gini'`, `'entropy'` | Mathematical formula used to measure node impurity. |
| `max_depth` | `2`, `3`, `4`, `5`, `None` | Maximum depth of the tree. Limits depth to prevent overfitting. |
| `min_samples_split` | `2`, `4` | Minimum samples required inside a node before it can be split. |
| `min_samples_leaf` | `1`, `2` | Minimum samples required to exist in a leaf node. |

---

## 10. How Training Works

Training is the execution of an optimization algorithm over the training data:

```text
Training Features (X_train) + Training Labels (y_train)
                    ↓
        Decision Tree Algorithm
                    ↓
  Evaluate all possible candidate splits
                    ↓
  Select split maximizing Impurity Reduction
                    ↓
  Recursively partition subsets until stopping criteria met
                    ↓
        Learned Decision Tree
```

For our Decision Tree using **Gini Impurity**:
$$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
where $p_i$ is the probability of class $i$ at that node. A node is completely pure ($I_G = 0$) when all samples belong to a single class.

---

## 11. What is Validation?

The **validation set** is a separate partition of data used to:
1. Compare different candidate hyperparameter configurations.
2. Detect overfitting early.
3. Select the best-performing model architecture.

We must **never** tune hyperparameters on the test set. If we repeatedly tweak settings to get a better test score, information from the test set leaks into our decisions, rendering the test set biased and invalid.

---

## 12. What is Testing?

The **test set** is an untouched, held-out partition of data that represents the "unseen future".
It is locked away during:
- Preprocessing
- Training
- Hyperparameter tuning

We evaluate on the test set **exactly once** at the very end to produce an unbiased estimate of how well the finalized model generalizes to new data.

---

## 13. Why We Don't Train on Test Data

If a model is trained on test data (or if test statistics leak into preprocessing), the model simply memorizes the answers.

Consider a teacher who gives students the exact test questions and answers the day before an exam. A student who scores 100% may simply possess good memory rather than an understanding of the subject. Training on test data destroys the ability to measure **generalization**.

---

## 14. What is Overfitting?

**Overfitting** occurs when a model learns the random noise, idiosyncrasies, and specific quirks of the training data rather than true underlying patterns.

- **Symptom**: Training accuracy is 100%, but validation/test accuracy drops significantly.
- **Decision Tree Cause**: An unconstrained decision tree will keep splitting until every single training example is in its own leaf node, creating an excessively complex tree that fails on unseen examples.
- **Solution**: Regularization / pruning by constraining `max_depth` and `min_samples_leaf`.

---

## 15. What is Generalization?

**Generalization** is a machine learning model's ability to make accurate predictions on **new, previously unseen data** drawn from the same distribution.

The true goal of machine learning is never high training accuracy; the true goal is **high generalization performance**.

---

## 16. Evaluation Metrics

For multiclass classification, relying solely on raw accuracy can be dangerously misleading when classes are imbalanced. We evaluate multiple complementary metrics:

1. **Accuracy**: Fraction of all predictions that were correct:
   $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$
2. **Precision**: Of all instances the model labeled as Class $C$, how many actually belonged to Class $C$?
   $$\text{Precision} = \frac{TP}{TP + FP}$$
3. **Recall (Sensitivity)**: Of all actual instances of Class $C$ in the dataset, how many did the model successfully find?
   $$\text{Recall} = \frac{TP}{TP + FN}$$
4. **F1-Score**: The harmonic mean of precision and recall:
   $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
5. **Macro Average**: Calculates the metric independently for each class and takes an unweighted average, treating rare classes (like Amphibians) with equal importance as dominant classes (like Mammals).

---

## 17. Confusion Matrix

A **Confusion Matrix** is a 2D contingency table where rows represent the true biological classes and columns represent the predicted classes:

```text
                  Predicted
              M   B   R   F   A
Actual M   [  6   0   0   0   0  ]
Actual B   [  0   3   0   0   0  ]
Actual R   [  0   1   0   0   0  ]  <- 1 Tortoise misclassified as Bird
Actual F   [  0   0   0   2   0  ]
Actual A   [  0   0   1   0   0  ]  <- 1 Newt misclassified as Reptile
```

Diagonal entries are correct predictions. Off-diagonal entries pinpoint exact biological confusions.

---

## 18. Dataset Details & Provenance

- **Dataset Name**: UCI Zoo Database
- **Creator / Donor**: Richard S. Forsyth (May 15, 1990)
- **Source URL**: [https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/](https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Total Raw Instances**: 101 animals across 7 classes.
- **5-Class Target Vertebrates Subset**: 83 animals.

### Verified Sample Breakdown (Programmatically Audited):
- `Mammal`: 41 animals (49.4%)
- `Bird`: 20 animals (24.1%)
- `Fish`: 13 animals (15.7%)
- `Reptile`: 5 animals (6.0%)
- `Amphibian`: 4 animals (4.8%)

### Verified Dataset Health Findings:
- **Missing Values**: 0 (Clean).
- **Exact Row Duplicates**: 0 across all columns.
- **Duplicate Animal Names**: Exactly 1 name appears twice (`frog` at Row 25 and Row 26). Inspection reveals row 25 is non-venomous (`venomous=0`) while row 26 is venomous (`venomous=1`), representing distinct biological frog variants.
- **Identical Feature Vectors**: In this 16-feature space, 61 animals share attribute profiles with at least one other animal (e.g., lion, cheetah, leopard, and wolf have identical binary mammalian carnivore signatures).
- **Contradictory Labels**: **0**. No two animals with identical feature vectors belong to conflicting classes.

---

## 19. Data Preprocessing & Anti-Leakage Approach

1. **Identifier Exclusion**: `animal_name` is strictly dropped from predictive features. If retained, the model could simply memorize names rather than learning biological characteristics.
2. **Target Isolation**: `class_name` and `class_type` are strictly separated from feature matrix $X$.
3. **Stratified Splitting**:
   Because Reptiles (5) and Amphibians (4) have so few samples, an unstratified split could easily leave validation or test sets with zero examples of these classes.
   We implemented a deterministic stratified 70 / 15 / 15 split:
   - **Training Set**: 57 samples (68.7%)
   - **Validation Set**: 13 samples (15.7%)
   - **Test Set**: 13 samples (15.7%)
4. **Leakage Guarantee**: All transformations and metadata ordering are fit solely on the training data.

---

## 20. How to Run the Project

### Prerequisites
Make sure Python 3.10+ is installed.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/download_data.py
```

### 3. Run Exploratory Data Analysis & Visualizations
```bash
python src/explore_data.py
```
*Generates figures in `reports/figures/class_distribution.png` and `reports/figures/correlation_matrix.png`.*

### 4. Preprocess Data & Perform Stratified Split
```bash
python src/preprocess.py
```
*Creates `data/processed/train.csv`, `val.csv`, and `test.csv`.*

### 5. Train Model & Tune Hyperparameters
```bash
python src/train.py
```
*Trains baseline, tunes Decision Tree on validation data, trains comparative Random Forest, serializes model to `models/decision_tree_model.joblib`, and outputs tree visualizations.*

### 6. Evaluate Final Model on Untouched Test Set
```bash
python src/evaluate.py
```
*Calculates all test metrics, generates `reports/figures/confusion_matrix.png`, and writes `reports/evaluation.md`.*

### 7. Run Automated Tests
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 21. How to Make Predictions

### Option A: Interactive CLI Mode
```bash
python src/predict.py
```
Step-by-step prompts ask for each biological feature.

### Option B: Predefined Animal Samples
```bash
python src/predict.py --sample
```
Runs prediction on iconic animals (Golden Retriever, Bald Eagle, Clownfish, Tree Frog, Rattlesnake, and Fruit Bat).

### Option C: Direct CLI Flags
```bash
python src/predict.py --hair 1 --feathers 0 --eggs 0 --milk 1 --airborne 0 --aquatic 0 --predator 1 --toothed 1 --backbone 1 --breathes 1 --venomous 0 --fins 0 --legs 4 --tail 1 --domestic 1 --catsize 1
```

### Option D: Local Web Interface
```bash
python src/app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your web browser. Toggle animal attributes and click **Predict Biological Class** to view real-time predictions and class probability bar charts.

---

## 22. Actual Experimental Results

> [!NOTE]
> All figures below reflect **genuine, verified execution numbers**. No synthetic or fabricated scores are reported.

### 1. Model Benchmark Comparison

| Model | Architecture | Training Accuracy | Validation Accuracy | Test Accuracy (N=13) | Test Macro F1 |
|---|---|---|---|---|---|
| **Baseline** | Majority Class (`Mammal`) | 50.88% | 46.15% | **46.15%** | 0.1263 |
| **Decision Tree** | `criterion='gini'`, `max_depth=4` | 100.00% | 92.31% | **84.62%** | 0.5714 |
| **Random Forest** | 50 Trees Ensemble | 100.00% | 92.31% | **84.62%** | 0.5714 |

### 2. Decision Tree Detailed Test Metrics ($N_{test} = 13$)
- **Overall Accuracy**: `84.62%` (11 of 13 correct)
- **Macro Precision**: `0.5500`
- **Macro Recall**: `0.6000`
- **Macro F1-Score**: `0.5714`
- **Weighted F1-Score**: `0.8132`

### 3. Per-Class Test Performance

| Class | True Samples | Correct | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| **Mammal** | 6 | 6 | 1.0000 | 1.0000 | 1.0000 |
| **Bird** | 3 | 3 | 0.7500 | 1.0000 | 0.8571 |
| **Fish** | 2 | 2 | 1.0000 | 1.0000 | 1.0000 |
| **Reptile** | 1 | 0 | 0.0000 | 0.0000 | 0.0000 |
| **Amphibian** | 1 | 0 | 0.0000 | 0.0000 | 0.0000 |

### 4. Anatomy of the Two Test Errors (A Fantastic Learning Lesson!):

The model achieved 11 / 13 correct. Let's analyze the 2 errors:
1. **Animal `newt` (Actual: Amphibian, Predicted: Reptile)**:
   - The learned decision tree learned that animals with `milk=0`, `toothed=1`, `breathes=1`, and `tail=0` are Amphibians (which fit all frogs/toads in training).
   - Because a `newt` is a tailed amphibian (`tail=1`), the decision tree classified it as a Reptile!
2. **Animal `tortoise` (Actual: Reptile, Predicted: Bird)**:
   - In the training set, every animal without teeth (`toothed=0`) and without milk was a Bird.
   - However, a tortoise has a beak rather than teeth (`toothed=0`), causing the tree to predict Bird!

This demonstrates precisely how a decision tree operates: it does not "know" biological taxonomy; it simply follows the statistical boundaries learned from its training data.

---

## 23. Critical Limitations & Real-World Zoology

> [!WARNING]
> **This project is strictly an educational instrument.** The high accuracy on this subset must NOT be interpreted as evidence of real-world biological classification capability.

### Real-World Limitations:
1. **Tiny Sample Size ($N = 83$ total, $N_{test} = 13$)**:
   - In a test set of 13 animals, a single error shifts accuracy by $\frac{1}{13} \approx 7.69\%$.
   - The test sample size is far too small for statistical significance.
2. **Severe Class Imbalance**:
   - Mammals and Birds make up 73.5% of the data.
   - Reptile and Amphibian each had only 1 sample in the test set. Because both were missed due to unique edge traits (tailed amphibian, toothless reptile), their test recall dropped to 0.0%.
3. **Feature Space Oversimplification**:
   - Real-world zoology contains complex evolutionary convergences (e.g. monotremes like platypuses that lay eggs and produce milk, legless skinks, lungfish, cetaceans like dolphins with no hair).
   - A 16-feature binary representation cannot capture the full spectrum of biology.

---

## 24. Project Directory Map

```text
animal-classifier/
├── README.md                          <- You are here! Complete educational guide
├── requirements.txt                   <- Core dependencies
├── .gitignore                         <- Ignored files
├── data/
│   ├── raw/
│   │   ├── zoo.data                   <- Raw UCI database
│   │   ├── zoo.names                  <- UCI documentation & schema
│   │   └── dataset_info.json          <- Provenance & checksums
│   └── processed/
│       ├── train.csv                  <- 57 training samples
│       ├── val.csv                    <- 13 validation samples
│       ├── test.csv                   <- 13 test samples
│       └── metadata.json              <- Feature and class schemas
├── models/
│   ├── decision_tree_model.joblib     <- Serialized Decision Tree
│   ├── random_forest_model.joblib     <- Serialized Random Forest comparison
│   └── metadata.json                  <- Model parameters & test scores
├── notebooks/
│   └── exploration.ipynb              <- Interactive step-by-step Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── download_data.py               <- UCI data downloader
│   ├── explore_data.py                <- Exploratory data analysis & figure generator
│   ├── preprocess.py                  <- Anti-leakage feature selector & stratified splitter
│   ├── train.py                       <- Model trainer, tuner & tree visualizer
│   ├── evaluate.py                    <- Unbiased test evaluator & confusion matrix plotter
│   ├── predict.py                     <- Interactive & flag-based CLI prediction tool
│   └── app.py                         <- Zero-dependency local web interface
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py               <- 9 automated unit & pipeline tests
└── reports/
    ├── evaluation.md                  <- Full evaluation report
    ├── decision_tree_structure.txt    <- Plain-text tree decision logic
    └── figures/
        ├── class_distribution.png     <- Sample counts per class
        ├── correlation_matrix.png     <- Feature correlation heatmap
        ├── decision_tree.png          <- High-res visualization of the trained tree
        └── confusion_matrix.png       <- 5x5 test confusion matrix
```
