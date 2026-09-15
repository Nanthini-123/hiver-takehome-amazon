"""
Automated Confusion Matrix & Per-Intent Performance Breakdown.
Computes class-level precision, recall, F1-score, support, and generates
the multi-class confusion matrix for @AmazonHelp intent classification.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREDICTIONS_PATH, RESULTS_DIR, INTENTS


def generate_detailed_breakdown():
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    y_true = [p["gold_intent"] for p in preds]
    y_pred = [p["predicted_intent"] for p in preds]

    # Use canonical intent labels order
    labels = [intent for intent in INTENTS if intent in set(y_true) | set(y_pred)]
    if not labels:
        labels = sorted(list(set(y_true) | set(y_pred)))

    # Generate classification report dict
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{lbl}" for lbl in labels], columns=[f"Pred_{lbl}" for lbl in labels])

    # Convert classification report to DataFrame for clean presentation
    df_report = pd.DataFrame(report_dict).transpose()

    # Save per_intent_breakdown.json
    out_path = RESULTS_DIR / "per_intent_breakdown.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Save confusion matrix CSV
    cm_out_path = RESULTS_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_out_path)

    print("\n" + "=" * 65)
    print("      PER-INTENT CLASSIFICATION METRICS (@AmazonHelp)      ")
    print("=" * 65)
    print(df_report.to_string())

    print("\n" + "=" * 65)
    print("              MULTI-CLASS CONFUSION MATRIX                 ")
    print("=" * 65)
    print(cm_df.to_string())
    print("\n[confusion_matrix] Breakdown saved to:")
    print(f"  • JSON: {out_path}")
    print(f"  • CSV:  {cm_out_path}\n")


if __name__ == "__main__":
    generate_detailed_breakdown()
