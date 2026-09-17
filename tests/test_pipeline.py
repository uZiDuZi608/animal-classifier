"""
tests/test_pipeline.py
----------------------
Comprehensive automated test suite for the Animal Biological Class Predictor.

Covers the 9 verification points mandated by the project specification:
1. Dataset loads successfully.
2. Required target classes exist (Mammal, Bird, Reptile, Fish, Amphibian).
3. Features and labels have matching lengths.
4. Train/validation/test splits contain data and proper stratification.
5. Model can train without error.
6. Model can predict on feature inputs.
7. Predictions strictly belong to the expected five classes.
8. Saved serialized model can be loaded from disk.
9. Prediction pipeline works on sample input and produces valid probabilities.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib

# Add src to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from preprocess import FEATURE_COLUMNS, TARGET_COL, CLASS_MAP
from predict import predict_animal, load_inference_artifacts
from sklearn.tree import DecisionTreeClassifier

class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_dir = PROJECT_DIR
        cls.raw_data_path = os.path.join(cls.project_dir, "data", "raw", "zoo.data")
        cls.train_path = os.path.join(cls.project_dir, "data", "processed", "train.csv")
        cls.val_path = os.path.join(cls.project_dir, "data", "processed", "val.csv")
        cls.test_path = os.path.join(cls.project_dir, "data", "processed", "test.csv")
        cls.model_path = os.path.join(cls.project_dir, "models", "decision_tree_model.joblib")
        cls.expected_classes = {"Mammal", "Bird", "Reptile", "Fish", "Amphibian"}

    # Test 1: Dataset loads successfully
    def test_01_dataset_loads_successfully(self):
        self.assertTrue(os.path.exists(self.raw_data_path), f"Raw dataset missing at: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path, header=None)
        self.assertGreater(len(df), 0, "Raw dataset has 0 rows.")
        self.assertEqual(df.shape[1], 18, "Raw dataset should have 18 columns.")

    # Test 2: Required target classes exist
    def test_02_required_target_classes_exist(self):
        df = pd.read_csv(self.raw_data_path, header=None)
        class_col = df.iloc[:, -1]
        mapped_classes = class_col.map(CLASS_MAP).dropna().unique()
        for expected in self.expected_classes:
            self.assertIn(expected, mapped_classes, f"Expected class '{expected}' not found in dataset.")

    # Test 3: Features and labels have matching lengths
    def test_03_features_and_labels_have_matching_lengths(self):
        self.assertTrue(os.path.exists(self.train_path), "Train CSV does not exist.")
        train_df = pd.read_csv(self.train_path)
        X = train_df[FEATURE_COLUMNS]
        y = train_df[TARGET_COL]
        self.assertEqual(len(X), len(y), "Features X and labels y must have identical length.")
        self.assertEqual(X.shape[1], 16, "Expected exactly 16 feature columns.")
        self.assertNotIn("animal_name", FEATURE_COLUMNS, "Data Leakage: animal_name must NOT be a feature.")

    # Test 4: Train / Validation / Test splits contain data
    def test_04_train_val_test_splits_contain_data(self):
        self.assertTrue(os.path.exists(self.train_path), "Train split missing.")
        self.assertTrue(os.path.exists(self.val_path), "Val split missing.")
        self.assertTrue(os.path.exists(self.test_path), "Test split missing.")

        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)

        self.assertGreater(len(train_df), 0, "Train split is empty.")
        self.assertGreater(len(val_df), 0, "Validation split is empty.")
        self.assertGreater(len(test_df), 0, "Test split is empty.")

        # Check total sample integrity
        total_split_samples = len(train_df) + len(val_df) + len(test_df)
        self.assertEqual(total_split_samples, 83, f"Expected 83 total samples across splits, got {total_split_samples}")

        # Check that all 5 classes are present in train, val, and test
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            classes_present = set(split_df[TARGET_COL].unique())
            self.assertEqual(classes_present, self.expected_classes,
                             f"{split_name} split missing classes: {self.expected_classes - classes_present}")

    # Test 5: Model can train
    def test_05_model_can_train(self):
        train_df = pd.read_csv(self.train_path)
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COL]

        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        try:
            model.fit(X_train, y_train)
            trained = True
        except Exception as e:
            trained = False
            self.fail(f"Model fitting failed with exception: {e}")
        self.assertTrue(trained)
        self.assertGreater(model.get_depth(), 0, "Trained tree depth should be > 0.")

    # Test 6: Model can predict
    def test_06_model_can_predict(self):
        train_df = pd.read_csv(self.train_path)
        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COL]

        model = DecisionTreeClassifier(max_depth=4, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_train.iloc[:5])
        self.assertEqual(len(preds), 5, "Predictions output length mismatch.")

    # Test 7: Predictions belong to the expected five classes
    def test_07_predictions_belong_to_expected_classes(self):
        test_df = pd.read_csv(self.test_path)
        X_test = test_df[FEATURE_COLUMNS]

        model = joblib.load(self.model_path)
        preds = model.predict(X_test)
        for p in preds:
            self.assertIn(p, self.expected_classes, f"Prediction '{p}' is not one of {self.expected_classes}")

    # Test 8: Saved model can be loaded
    def test_08_saved_model_can_be_loaded(self):
        self.assertTrue(os.path.exists(self.model_path), f"Saved model missing at: {self.model_path}")
        loaded_model = joblib.load(self.model_path)
        self.assertIsInstance(loaded_model, DecisionTreeClassifier, "Loaded object is not a DecisionTreeClassifier.")
        self.assertTrue(hasattr(loaded_model, "tree_"), "Loaded model has not been fitted.")

    # Test 9: Prediction pipeline works on sample input & valid probabilities
    def test_09_prediction_pipeline_works_on_sample(self):
        sample_input = {
            "hair": 1, "feathers": 0, "eggs": 0, "milk": 1, "airborne": 0,
            "aquatic": 0, "predator": 1, "toothed": 1, "backbone": 1, "breathes": 1,
            "venomous": 0, "fins": 0, "legs": 4, "tail": 1, "domestic": 1, "catsize": 1
        }
        pred_class, probs = predict_animal(sample_input)
        self.assertEqual(pred_class, "Mammal", f"Expected Mammal for dog-like input, got {pred_class}")

        # Check probability axioms
        self.assertEqual(len(probs), 5, "Should have probabilities for 5 classes.")
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=4, msg="Probabilities must sum to 1.0")
        for cls, p in probs.items():
            self.assertGreaterEqual(p, 0.0, f"Negative probability found for {cls}: {p}")
            self.assertLessEqual(p, 1.0, f"Probability > 1.0 found for {cls}: {p}")

if __name__ == "__main__":
    unittest.main()
