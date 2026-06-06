"""
train_model.py
==============
End-to-end training pipeline for the SocialPhishing URL classifier.

Steps
-----
1. Load the dataset (PhishTank phishing + synthetic safe URLs)
2. Extract 8 URL features for every row
3. Encode labels to integers: Safe=0, Phishing=1
4. Split 80 / 20 into train / test (stratified, random_state=42)
5. Train a Random Forest classifier (random_state=42)
6. Evaluate on the test set and print a full classification report
7. Save the trained model to  model/phishing_model.pkl  using joblib

Usage
-----
    cd url-classifier/model
    pip install -r requirements.txt
    python train_model.py

Output files
------------
    phishing_model.pkl  -- serialised RandomForestClassifier + metadata
"""

import io
import sys

# Force UTF-8 output so Unicode chars print correctly on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dataset_loader import load_dataset
from feature_extractor import extract_features, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_PATH = Path(__file__).parent / "phishing_model.pkl"

# SVM hyperparameters
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Apply extract_features to every URL in *df* and return a (n, 8) numpy array.

    Prints a progress update every 10,000 rows.
    """
    print(f"[train] Extracting features for {len(df):,} URLs ...")
    start = time.time()
    rows = []
    for i, url in enumerate(df["url"], start=1):
        feat = extract_features(url)
        rows.append([feat[col] for col in FEATURE_COLUMNS])
        if i % 10_000 == 0:
            elapsed = time.time() - start
            print(f"  ... {i:,} / {len(df):,} done  ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"[train] Feature extraction complete in {elapsed:.1f}s")
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" SocialPhishing — URL Classifier Training Pipeline")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Load dataset (capped to 5,000 phishing rows for fast SVM training)
    # ------------------------------------------------------------------
    df = load_dataset(max_phishing=5000)

    # ------------------------------------------------------------------
    # 2. Extract features
    # ------------------------------------------------------------------
    X = build_feature_matrix(df)

    # ------------------------------------------------------------------
    # 3. Encode labels
    #    Safe      -> 0
    #    Phishing  -> 1
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(df["label"])  # alphabetical order by default
    class_names = list(le.classes_)   # ['Phishing', 'Safe']

    print(f"\n[train] Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"[train] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    # ------------------------------------------------------------------
    # 4. Train / test split — stratified to preserve class ratio
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[train] Train size: {len(X_train):,}   Test size: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 5. Scale features and Train the SVM
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\n[train] Training SVC ...")
    print(f"        Hyperparameters: {SVM_PARAMS}")
    t0 = time.time()
    clf = SVC(**SVM_PARAMS)
    clf.fit(X_train, y_train)
    print(f"[train] Training complete in {time.time() - t0:.1f}s\n")

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{'=' * 60}")
    print(f" ACCURACY:  {acc * 100:.2f}%  (target: >=92%)")
    print(f"{'=' * 60}\n")

    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("CONFUSION MATRIX")
    print("-" * 60)
    cm = confusion_matrix(y_test, y_pred)
    # Pretty-print the confusion matrix with labels
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names],
    )
    print(cm_df.to_string())
    print()

    # Calculate false positive rate (safe URLs misclassified as phishing)
    # True label = Safe (index of "Safe" in class_names)
    safe_idx = class_names.index("Safe")
    phishing_idx = class_names.index("Phishing")

    # FP = Safe predicted as Phishing;  TN = Safe predicted as Safe
    fp = cm[safe_idx][phishing_idx]
    tn = cm[safe_idx][safe_idx]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"FALSE POSITIVE RATE: {fpr * 100:.2f}%  (target: <=5%)")
    print()

    # Warn if targets are not met
    if acc < 0.92:
        print("[WARNING] Accuracy is below the 92% target. Consider expanding "
              "the dataset or tuning hyperparameters.")
    if fpr > 0.05:
        print("[WARNING] False positive rate exceeds 5%. Consider adding more "
              "diverse safe URLs or tuning min_samples_leaf.")

    # Note: feature_importances_ is not available for SVC with rbf kernel.

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    model_artifact = {
        "classifier": clf,
        "scaler": scaler,
        "label_encoder": le,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": class_names,
        "accuracy": float(acc),
        "false_positive_rate": float(fpr),
    }
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"[train] Model saved to: {MODEL_PATH}")
    print("[train] Done.")


if __name__ == "__main__":
    main()
