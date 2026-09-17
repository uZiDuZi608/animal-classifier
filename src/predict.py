"""
src/predict.py
--------------
Inference CLI for the Animal Biological Class Predictor.

Educational Focus:
1. Model Serving / Inference:
   - Once trained and evaluated, the model is serialized (saved to disk).
   - In production, we load the saved model without retraining.
2. Input Alignment:
   - Inference inputs MUST match the exact 16 feature names and ordering
     used during model training.
3. Probabilistic Predictions:
   - In addition to the final predicted class, Decision Trees provide class
     probabilities based on the fraction of training samples of each class
     present in the terminal leaf node.
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "decision_tree_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "metadata.json")

FEATURE_COLUMNS = [
    "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous",
    "fins", "legs", "tail", "domestic", "catsize"
]

FEATURE_DESCRIPTIONS = {
    "hair": "Has hair / fur (1 = Yes, 0 = No)",
    "feathers": "Has feathers (1 = Yes, 0 = No)",
    "eggs": "Lays eggs (1 = Yes, 0 = No)",
    "milk": "Produces milk / nurses young (1 = Yes, 0 = No)",
    "airborne": "Can fly / airborne (1 = Yes, 0 = No)",
    "aquatic": "Lives in or spends extensive time in water (1 = Yes, 0 = No)",
    "predator": "Is a predator / carnivorous hunter (1 = Yes, 0 = No)",
    "toothed": "Has teeth (1 = Yes, 0 = No)",
    "backbone": "Has a backbone / vertebrae (1 = Yes, 0 = No)",
    "breathes": "Breathes air (1 = Yes, 0 = No)",
    "venomous": "Is venomous (1 = Yes, 0 = No)",
    "fins": "Has fins (1 = Yes, 0 = No)",
    "legs": "Number of legs (0, 2, 4, 5, 6, 8)",
    "tail": "Has a tail (1 = Yes, 0 = No)",
    "domestic": "Is domesticated (1 = Yes, 0 = No)",
    "catsize": "Is roughly cat-sized or larger (1 = Yes, 0 = No)"
}

SAMPLE_ANIMALS = {
    "Golden Retriever (Domestic Mammal)": {
        "hair": 1, "feathers": 0, "eggs": 0, "milk": 1, "airborne": 0,
        "aquatic": 0, "predator": 1, "toothed": 1, "backbone": 1, "breathes": 1,
        "venomous": 0, "fins": 0, "legs": 4, "tail": 1, "domestic": 1, "catsize": 1
    },
    "Bald Eagle (Raptor Bird)": {
        "hair": 0, "feathers": 1, "eggs": 1, "milk": 0, "airborne": 1,
        "aquatic": 0, "predator": 1, "toothed": 0, "backbone": 1, "breathes": 1,
        "venomous": 0, "fins": 0, "legs": 2, "tail": 1, "domestic": 0, "catsize": 1
    },
    "Clownfish (Aquatic Fish)": {
        "hair": 0, "feathers": 0, "eggs": 1, "milk": 0, "airborne": 0,
        "aquatic": 1, "predator": 0, "toothed": 1, "backbone": 1, "breathes": 0,
        "venomous": 0, "fins": 1, "legs": 0, "tail": 1, "domestic": 0, "catsize": 0
    },
    "Tree Frog (Tail-less Amphibian)": {
        "hair": 0, "feathers": 0, "eggs": 1, "milk": 0, "airborne": 0,
        "aquatic": 1, "predator": 1, "toothed": 1, "backbone": 1, "breathes": 1,
        "venomous": 0, "fins": 0, "legs": 4, "tail": 0, "domestic": 0, "catsize": 0
    },
    "Rattlesnake (Venomous Reptile)": {
        "hair": 0, "feathers": 0, "eggs": 1, "milk": 0, "airborne": 0,
        "aquatic": 0, "predator": 1, "toothed": 1, "backbone": 1, "breathes": 1,
        "venomous": 1, "fins": 0, "legs": 0, "tail": 1, "domestic": 0, "catsize": 0
    },
    "Fruit Bat (Flying Mammal Edge Case)": {
        "hair": 1, "feathers": 0, "eggs": 0, "milk": 1, "airborne": 1,
        "aquatic": 0, "predator": 0, "toothed": 1, "backbone": 1, "breathes": 1,
        "venomous": 0, "fins": 0, "legs": 2, "tail": 1, "domestic": 0, "catsize": 0
    }
}

def load_inference_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run src/train.py first.")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def predict_animal(features_dict: dict, model=None, meta=None):
    if model is None or meta is None:
        model, meta = load_inference_artifacts()

    # Ensure correct column ordering
    input_vector = [features_dict[col] for col in FEATURE_COLUMNS]
    input_df = pd.DataFrame([input_vector], columns=FEATURE_COLUMNS)

    # Inference
    predicted_class = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    class_probs = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}

    return predicted_class, class_probs

def display_prediction(title: str, features: dict, predicted_class: str, probabilities: dict):
    print("\n" + "=" * 60)
    print(f"  PREDICTION RESULT: {title}")
    print("=" * 60)
    print("  Input Characteristics:")
    for k, v in features.items():
        print(f"    - {k:<10}: {v}")

    print("\n  " + "-" * 56)
    print(f"  >>> PREDICTED CLASS:  [ {predicted_class.upper()} ] <<<")
    print("  " + "-" * 56)

    print("\n  Model Class Probabilities (Learned Leaf Distribution):")
    # Sort descending by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for cls, p in sorted_probs:
        bar = "#" * int(round(p * 30))
        print(f"    {cls:<12}: {p:>6.2%}  |{bar:<30}|")

def run_interactive_prompt():
    print("=" * 65)
    print("   ANIMAL CLASSIFIER: INTERACTIVE PREDICTION MODE")
    print("=" * 65)
    print("Enter the biological characteristics for your mystery animal.\n")

    user_inputs = {}
    for feat in FEATURE_COLUMNS:
        desc = FEATURE_DESCRIPTIONS[feat]
        while True:
            try:
                raw = input(f"  {desc} > ").strip()
                val = int(raw)
                if feat == "legs":
                    if val < 0 or val > 8:
                        print("    [!] Please enter a valid number of legs (typically 0, 2, 4, 6, or 8).")
                        continue
                else:
                    if val not in [0, 1]:
                        print("    [!] Please enter 1 (Yes) or 0 (No).")
                        continue
                user_inputs[feat] = val
                break
            except ValueError:
                print("    [!] Invalid input. Please enter an integer number.")

    model, meta = load_inference_artifacts()
    pred_class, probs = predict_animal(user_inputs, model, meta)
    display_prediction("Custom User-Defined Animal", user_inputs, pred_class, probs)

def main():
    parser = argparse.ArgumentParser(description="Animal Biological Class Predictor (Decision Tree CLI)")
    parser.add_argument("--interactive", action="store_true", help="Launch step-by-step interactive prompt")
    parser.add_argument("--sample", action="store_true", help="Run prediction on predefined sample animals")

    # Optional individual feature flags
    for feat in FEATURE_COLUMNS:
        parser.add_argument(f"--{feat}", type=int, default=None, help=FEATURE_DESCRIPTIONS[feat])

    args = parser.parse_args()

    # If --sample is passed
    if args.sample:
        print("=" * 65)
        print("   RUNNING PREDICTIONS ON PREDEFINED SAMPLE ANIMALS")
        print("=" * 65)
        model, meta = load_inference_artifacts()
        for name, feats in SAMPLE_ANIMALS.items():
            pred, probs = predict_animal(feats, model, meta)
            display_prediction(name, feats, pred, probs)
        return

    # Check if features were passed via CLI flags
    passed_flags = {feat: getattr(args, feat) for feat in FEATURE_COLUMNS if getattr(args, feat) is not None}

    if len(passed_flags) == len(FEATURE_COLUMNS):
        model, meta = load_inference_artifacts()
        pred, probs = predict_animal(passed_flags, model, meta)
        display_prediction("CLI Specified Animal", passed_flags, pred, probs)
        return
    elif len(passed_flags) > 0:
        print(f"[!] Error: You provided {len(passed_flags)} of {len(FEATURE_COLUMNS)} features.")
        print("    Either provide ALL 16 feature flags, or run without flags for interactive mode / --sample.")
        sys.exit(1)

    # Default to interactive mode if no arguments provided
    run_interactive_prompt()

if __name__ == "__main__":
    main()
