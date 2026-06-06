"""
evaluate_model.py
=================
Standalone evaluation script for the saved phishing_model.pkl.

Loads the serialised model and re-evaluates it against a fresh test split
(identical split used in train_model.py -- same random_state and test_size)
to produce a detailed academic-quality performance report.

Outputs
-------
- Overall accuracy
- Per-class precision, recall, F1-score (classification report)
- Confusion matrix (raw counts + normalised percentages)
- False positive rate (safe URLs wrongly flagged as phishing)
- False negative rate (phishing URLs missed)
- Feature importance ranking (if supported by model)
- Optional: saves a confusion-matrix PNG if matplotlib is available

Usage
-----
    cd url-classifier/model
    python evaluate_model.py
"""

import io
import sys

# Force UTF-8 output so Unicode chars print correctly on Windows cp1252 terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from dataset_loader import load_dataset
from feature_extractor import extract_features, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Configuration — must match train_model.py
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_PATH = Path(__file__).parent / "phishing_model.pkl"
PLOT_PATH = Path(__file__).parent / "confusion_matrix.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract features for every URL in *df* and return an (n, 8) array."""
    print(f"[evaluate] Extracting features for {len(df):,} URLs …")
    rows = [
        [extract_features(url)[col] for col in FEATURE_COLUMNS]
        for url in df["url"]
    ]
    return np.array(rows, dtype=float)


def print_separator(title: str = "", width: int = 62):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'=' * pad} {title} {'=' * (width - pad - len(title) - 2)}")
    else:
        print("=" * width)


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    """Pretty-print a confusion matrix with row/column labels."""
    col_width = 14
    header = " " * 22 + "".join(f"{'Pred: ' + c:>{col_width}}" for c in class_names)
    print(header)
    print(" " * 22 + "-" * (col_width * len(class_names)))
    for i, row_label in enumerate(class_names):
        row = f"  Actual: {row_label:<12}" + "".join(
            f"{cm[i][j]:>{col_width},}" for j in range(len(class_names))
        )
        print(row)
    print()


def save_confusion_matrix_plot(cm: np.ndarray, class_names: list):
    """Save a colour-coded confusion matrix figure (requires matplotlib + seaborn)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Confusion Matrix — SocialPhishing URL Classifier", fontsize=13)

        # Raw counts
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0]
        )
        axes[0].set_title("Raw Counts")
        axes[0].set_ylabel("Actual Label")
        axes[0].set_xlabel("Predicted Label")

        # Normalised (row-wise percentages)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2%", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names, ax=axes[1]
        )
        axes[1].set_title("Normalised (row %)")
        axes[1].set_ylabel("Actual Label")
        axes[1].set_xlabel("Predicted Label")

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150)
        plt.close()
        print(f"[evaluate] Confusion matrix plot saved → {PLOT_PATH}")
    except ImportError:
        print("[evaluate] matplotlib/seaborn not available — skipping plot.")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  SocialPhishing — Model Evaluation Report")
    print("=" * 62)

    # ------------------------------------------------------------------
    # 1. Load saved model artifact
    # ------------------------------------------------------------------
    if not MODEL_PATH.exists():
      raise FileNotFoundError(
          f"Model file not found: {MODEL_PATH}\n"
          "Run train_model.py first to generate phishing_model.pkl"
      )

    print(f"\n[evaluate] Loading model from: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    clf = artifact["classifier"]
    le = artifact["label_encoder"]
    class_names = artifact["class_names"]   # e.g. ['Phishing', 'Safe']
    stored_acc = artifact.get("accuracy", None)
    stored_fpr = artifact.get("false_positive_rate", None)

    print(f"[evaluate] Model type    : {type(clf).__name__}")
    print(f"[evaluate] Classes       : {class_names}")
    if stored_acc:
        print(f"[evaluate] Stored accuracy (train run): {stored_acc * 100:.2f}%")

    # ------------------------------------------------------------------
    # 2. Rebuild the identical test set
    # ------------------------------------------------------------------
    print("\n[evaluate] Re-loading dataset to reproduce the test split …")
    df = load_dataset()
    X = build_feature_matrix(df)
    y = le.transform(df["label"])

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale test features if scaler is saved in model
    scaler = artifact.get("scaler")
    if scaler is not None:
        print("[evaluate] Applying standard scaling using saved scaler ...")
        X_test = scaler.transform(X_test)

    print(f"[evaluate] Test set size: {len(X_test):,} samples\n")

    # ------------------------------------------------------------------
    # 3. Predict
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)   # shape (n, n_classes)

    # ------------------------------------------------------------------
    # 4. Accuracy
    # ------------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    print_separator("ACCURACY")
    status = "[PASS]" if acc >= 0.92 else "[FAIL] (target: >=92%)"
    print(f"\n  Overall accuracy : {acc * 100:.4f}%   {status}\n")

    # ------------------------------------------------------------------
    # 5. Classification report
    # ------------------------------------------------------------------
    print_separator("CLASSIFICATION REPORT")
    print()
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    # ------------------------------------------------------------------
    # 6. Confusion matrix
    # ------------------------------------------------------------------
    print_separator("CONFUSION MATRIX")
    print()
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm, class_names)

    # ------------------------------------------------------------------
    # 7. Key error rates for academic report
    # ------------------------------------------------------------------
    print_separator("KEY PERFORMANCE METRICS")

    safe_idx = class_names.index("Safe")
    phishing_idx = class_names.index("Phishing")

    # False Positive: Safe URL predicted as Phishing (annoys legitimate users)
    fp = cm[safe_idx][phishing_idx]
    tn = cm[safe_idx][safe_idx]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpr_status = "[PASS]" if fpr <= 0.05 else "[FAIL] (target: <=5%)"

    # False Negative: Phishing URL predicted as Safe (dangerous miss)
    fn = cm[phishing_idx][safe_idx]
    tp = cm[phishing_idx][phishing_idx]
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ROC-AUC (binary)
    safe_class_prob = y_prob[:, safe_idx]   # P(Safe)
    try:
        auc = roc_auc_score(y_test, safe_class_prob)
    except Exception:
        auc = float("nan")

    print(f"\n  False Positive Rate : {fpr * 100:.4f}%   {fpr_status}")
    print(f"    -> {fp:,} safe URLs incorrectly flagged as phishing out of {fp + tn:,}")
    print()
    print(f"  False Negative Rate : {fnr * 100:.4f}%")
    print(f"    -> {fn:,} phishing URLs missed (classified as safe) out of {fn + tp:,}")
    print()
    print(f"  ROC-AUC Score       : {auc:.4f}")
    print()

    # Macro averages
    macro_prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  Macro Precision     : {macro_prec:.4f}")
    print(f"  Macro Recall        : {macro_rec:.4f}")
    print(f"  Macro F1-Score      : {macro_f1:.4f}")

    # ------------------------------------------------------------------
    # 8. Feature importances
    # ------------------------------------------------------------------
    print_separator("FEATURE IMPORTANCES (RANKED)")
    print()
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
        for rank, (feat, imp) in enumerate(feat_imp, 1):
            bar = "#" * int(imp * 50)
            print(f"  {rank}. {feat:<25} {imp:.4f}  {bar}")
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        importances = importances / np.sum(importances) # normalise to sum to 1
        feat_imp = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True)
        for rank, (feat, imp) in enumerate(feat_imp, 1):
            bar = "#" * int(imp * 50)
            print(f"  {rank}. {feat:<25} {imp:.4f}  {bar}")
    else:
        print("  [INFO] Feature importances are not natively available for this model type.")
    print()

    # ------------------------------------------------------------------
    # 9. Classifier internal stats
    # ------------------------------------------------------------------
    print_separator(f"{type(clf).__name__.upper()} DETAILS")
    if hasattr(clf, "n_estimators"):
        print(f"\n  Number of trees    : {clf.n_estimators}")
        print(f"  Max features/split : {clf.max_features}")
        print(f"  Max depth          : {clf.max_depth or 'None (full)'}")
        print(f"  OOB score enabled  : {clf.oob_score}")
        depths = [est.get_depth() for est in clf.estimators_]
        print(f"  Avg tree depth     : {np.mean(depths):.1f}")
        print(f"  Min tree depth     : {np.min(depths)}")
        print(f"  Max tree depth     : {np.max(depths)}")
    else:
        print(f"\n  Kernel type        : {getattr(clf, 'kernel', 'N/A')}")
        print(f"  Regularisation C   : {getattr(clf, 'C', 'N/A')}")
        print(f"  Gamma              : {getattr(clf, 'gamma', 'N/A')}")
        print(f"  Support vectors    : {len(getattr(clf, 'support_vectors_', []))}")
    print()

    # ------------------------------------------------------------------
    # 10. Save confusion matrix plot
    # ------------------------------------------------------------------
    save_confusion_matrix_plot(cm, class_names)

    # ------------------------------------------------------------------
    # Summary banner
    # ------------------------------------------------------------------
    print_separator()
    print("\n  EVALUATION SUMMARY")
    print(f"  Accuracy      : {acc * 100:.2f}%    {'[PASS] Target met' if acc >= 0.92 else '[FAIL] Below target'}")
    print(f"  FP Rate       : {fpr * 100:.2f}%    {'[PASS] Target met' if fpr <= 0.05 else '[FAIL] Above target'}")
    print(f"  FN Rate       : {fnr * 100:.2f}%")
    print(f"  Macro F1      : {macro_f1:.4f}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print()


if __name__ == "__main__":
    main()
