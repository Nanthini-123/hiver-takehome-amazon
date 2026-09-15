"""
Baseline models for @AmazonHelp customer support evaluation.
1. Trivial Majority Baseline: Always predicts dataset majority intent (delivery_issue) & auto_handle.
2. Simple Rule-Based Baseline: Keyword-driven intent classification & escalation matching.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GOLDEN_SET_PATH, BASELINE_PREDICTIONS_PATH, HIGH_RISK_KEYWORDS


def run_baselines(golden_set_path: Path = GOLDEN_SET_PATH, output_path: Path = BASELINE_PREDICTIONS_PATH):
    print(f"[baselines] Loading golden evaluation set from {golden_set_path}...")
    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_rows = [json.loads(line) for line in f if line.strip()]

    baseline_results = []

    for row in golden_rows:
        text_lower = row["customer_text"].lower()

        # 1. Trivial Baseline (Majority Intent: delivery_issue, always auto_handle)
        trivial_pred = {
            "intent": "delivery_issue",
            "decision": "auto_handle",
            "reply": "Please DM us your order details so we can investigate."
        }

        # 2. Simple Rule-Based Baseline
        if any(k in text_lower for k in ["refund", "return", "exchange"]):
            simple_intent = "refund_and_return"
        elif any(k in text_lower for k in ["delivery", "late", "package", "driver", "courier", "delivered"]):
            simple_intent = "delivery_issue"
        elif any(k in text_lower for k in ["charge", "billing", "card", "charged", "fee"]):
            simple_intent = "billing_and_charges"
        elif any(k in text_lower for k in ["track", "where", "status", "shipment", "eta"]):
            simple_intent = "order_status"
        elif any(k in text_lower for k in ["hack", "login", "password", "account", "otp", "2fa"]):
            simple_intent = "account_and_security"
        else:
            simple_intent = "other"

        simple_decision = "escalate" if any(k in text_lower for k in HIGH_RISK_KEYWORDS) else "auto_handle"

        simple_pred = {
            "intent": simple_intent,
            "decision": simple_decision,
            "reply": "Please reach out to our team via DM for assistance."
        }

        baseline_results.append({
            "thread_id": row["thread_id"],
            "trivial": trivial_pred,
            "simple": simple_pred
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for res in baseline_results:
            f.write(json.dumps(res) + "\n")

    print(f"[baselines] Successfully evaluated {len(baseline_results)} examples and saved to {output_path}")


if __name__ == "__main__":
    run_baselines()
