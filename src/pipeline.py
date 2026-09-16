"""
Main pipeline execution script for @AmazonHelp customer support agent.
Iterates over golden evaluation set, generates structured predictions,
evaluates composite risk scores, and enforces conservative safety guardrails.
Supports CLI arguments for brand selection and sample mode.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    GOLDEN_SET_PATH, PREDICTIONS_PATH, AGENT_PROMPT_PATH, 
    CONFIDENCE_THRESHOLD, RISK_THRESHOLD, HIGH_RISK_KEYWORDS,
    TARGET_BRAND
)
from llm_client import GeminiClient


def run_pipeline(golden_set_path: Path = GOLDEN_SET_PATH, output_path: Path = PREDICTIONS_PATH, brand: str = TARGET_BRAND, sample_mode: bool = False):
    client = GeminiClient()

    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    with open(golden_set_path, "r", encoding="utf-8") as f:
        golden_rows = [json.loads(line) for line in f if line.strip()]

    if sample_mode:
        print(f"[pipeline] Sample mode enabled: Evaluating first 20 examples for brand @{brand}...")
        golden_rows = golden_rows[:20]

    predictions = []
    print(f"[pipeline] Running agent pipeline over {len(golden_rows)} examples for @{brand}...")

    for row in tqdm(golden_rows, desc=f"Evaluating @{brand} Agent"):
        # Safe string replacement preventing curly brace format crashes
        prompt = prompt_template.replace(
            "{customer_text}", row["customer_text"]
        ).replace(
            "{context_text}", row.get("context_text", "")
        )

        pred = client.generate_json(prompt)

        # Risk score calculation & fallback guardrails
        text_lower = row["customer_text"].lower()
        keyword_risk = 0.5 if any(k in text_lower for k in HIGH_RISK_KEYWORDS) else 0.0
        calculated_risk = max(
            pred.get("risk_score", 0.0),
            round(1.0 - pred.get("intent_confidence", 1.0), 2),
            keyword_risk
        )
        pred["risk_score"] = round(calculated_risk, 2)

        # Force fallback & escalation rules
        if pred.get("intent_confidence", 0.0) < CONFIDENCE_THRESHOLD or pred.get("risk_score", 0.0) >= RISK_THRESHOLD:
            pred["decision"] = "escalate"
            pred["escalation_reason"] = f"Triggered safety guardrail: Confidence={pred.get('intent_confidence')}, Risk={pred.get('risk_score')}"
            pred["reply"] = "We want to help with your concern safely. Please send us a direct message so a specialized representative can assist."

        predictions.append({
            "thread_id": row["thread_id"],
            "brand": brand,
            "customer_text": row["customer_text"],
            "gold_intent": row["gold_intent"],
            "gold_decision": row["gold_decision"],
            "should_escalate": row.get("should_escalate", row["gold_decision"] == "escalate"),
            "predicted_intent": pred.get("intent", "other"),
            "intent_confidence": pred.get("intent_confidence", 0.0),
            "intent_reasoning": pred.get("intent_reasoning", ""),
            "predicted_reply": pred.get("reply", ""),
            "predicted_decision": pred.get("decision", "auto_handle"),
            "escalation_reason": pred.get("escalation_reason", ""),
            "risk_score": pred.get("risk_score", 0.0)
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"[pipeline] Pipeline predictions complete. Saved {len(predictions)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run @AmazonHelp AI Support Agent Pipeline")
    parser.add_argument("--brand", type=str, default=TARGET_BRAND, help="Target support brand handle (default: AmazonHelp)")
    parser.add_argument("--sample", action="store_true", help="Run on a small sample subset (20 items) for rapid verification")
    parser.add_argument("--golden_set", type=str, default=str(GOLDEN_SET_PATH), help="Path to golden set jsonl")
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_PATH), help="Path to output predictions jsonl")
    args = parser.parse_args()

    run_pipeline(
        golden_set_path=Path(args.golden_set),
        output_path=Path(args.output),
        brand=args.brand,
        sample_mode=args.sample
    )
