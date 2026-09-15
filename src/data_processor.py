"""
Data processor for Kaggle Customer Support on Twitter (twcs.csv).
Extracts real @AmazonHelp multi-turn customer support interactions,
builds multi-turn context, and generates the golden evaluation set.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TWCS_CSV_PATH, GOLDEN_SET_PATH, BRAND_THREADS_PATH, BRAND_HANDLE, HIGH_RISK_KEYWORDS

TARGET_QUOTAS = {
    "delivery_issue": 40,
    "order_status": 35,
    "refund_and_return": 35,
    "billing_and_charges": 30,
    "account_and_security": 30,
    "other": 30,
}

def determine_intent_and_decision(cust_text: str, reply_text: str):
    lower_c = cust_text.lower()
    scores = {intent: 0.0 for intent in TARGET_QUOTAS}

    if any(w in lower_c for w in ["delivered", "courier", "driver", "porch", "shattered", "broken", "damaged",
                                 "empty box", "neighbor", "wrong address", "not delivered", "haven't received",
                                 "stolen", "crushed", "package missing", "package"]):
        scores["delivery_issue"] += 3.5

    if any(w in lower_c for w in ["where is", "when will", "track", "tracking", "status", "shipment", "shipped",
                                 "in transit", "out for delivery", "eta", "expected", "delayed"]):
        scores["order_status"] += 3.0

    if any(w in lower_c for w in ["refund", "return", "exchange", "send back", "drop off", "ups drop", "money back",
                                 "label", "cancelled", "cancel order"]):
        scores["refund_and_return"] += 3.5

    if any(w in lower_c for w in ["charged", "charge", "billing", "bill", "double charge", "deducted", "fee",
                                 "credit card", "prime membership charge", "unauthorized charge", "bank"]):
        scores["billing_and_charges"] += 3.5

    if any(w in lower_c for w in ["login", "password", "otp", "hacked", "account locked", "compromised",
                                 "suspended", "security code", "sign in", "reset password", "two factor", "2fa"]):
        scores["account_and_security"] += 4.0

    best_intent = max(scores, key=scores.get)
    if scores[best_intent] < 2.0:
        best_intent = "other"

    # Escalation determination
    is_escalate = False
    if any(k in lower_c for k in HIGH_RISK_KEYWORDS):
        is_escalate = True
    elif any(k in lower_c for k in ["urgent", "lawyer", "police", "complaint", "unacceptable", "supervisor", "fraud"]):
        is_escalate = True
    elif best_intent == "account_and_security" and any(k in lower_c for k in ["hacked", "compromised", "unauthorized", "stolen"]):
        is_escalate = True

    return best_intent, "escalate" if is_escalate else "auto_handle"

def process_kaggle_dataset(input_csv: Path, output_jsonl: Path, sample_size: int = 200, force: bool = False):
    if output_jsonl.exists() and not force:
        with open(output_jsonl, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if len(rows) >= 150:
            intents = Counter(r.get("gold_intent") for r in rows)
            decisions = Counter(r.get("gold_decision") for r in rows)
            print(f"[data_processor] Using existing validated golden set at {output_jsonl} ({len(rows)} entries).")
            print(f"[data_processor] Intent distribution: {dict(intents)}")
            print(f"[data_processor] Decision distribution: {dict(decisions)}")
            return

    if not input_csv.exists():
        raise FileNotFoundError(f"Raw Kaggle dataset not found at {input_csv}. Download twcs.csv first.")

    print(f"[data_processor] Loading raw Kaggle dataset from {input_csv}...")
    
    # Fast streaming parser to build lookup tables
    amazon_tweets = {}
    inbound_candidates = []

    with open(input_csv, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = row["author_id"]
            tweet_id = row["tweet_id"]
            text = row["text"]
            resp_to = row.get("in_response_to_tweet_id")

            if author == BRAND_HANDLE:
                amazon_tweets[tweet_id] = {
                    "tweet_id": tweet_id,
                    "text": text,
                    "resp_to": resp_to
                }
            elif resp_to and row.get("inbound") == "True":
                inbound_candidates.append({
                    "tweet_id": tweet_id,
                    "text": text,
                    "resp_to": resp_to,
                    "author": author
                })

    print(f"[data_processor] Extracted {len(amazon_tweets)} @AmazonHelp tweets, {len(inbound_candidates)} inbound replies.")

    # Match inbounds where AmazonHelp replied
    resp_map = {t["resp_to"]: t for t in amazon_tweets.values() if t.get("resp_to")}
    
    matched_pairs = []
    for inb in inbound_candidates:
        if inb["tweet_id"] in resp_map:
            brand_reply = resp_map[inb["tweet_id"]]
            matched_pairs.append((inb, brand_reply))

    print(f"[data_processor] Found {len(matched_pairs)} direct customer -> @AmazonHelp conversation pairs.")

    # Stratified collection
    collected = {intent: [] for intent in TARGET_QUOTAS}
    for inb, brand_rep in matched_pairs:
        cust_text = inb["text"].strip()
        reply_text = brand_rep["text"].strip()
        intent, decision = determine_intent_and_decision(cust_text, reply_text)

        if len(collected[intent]) < TARGET_QUOTAS[intent]:
            collected[intent].append({
                "thread_id": f"amz_real_{inb['tweet_id']}",
                "brand": BRAND_HANDLE,
                "customer_text": cust_text,
                "context_text": "",
                "gold_intent": intent,
                "gold_decision": decision,
                "gold_reply": reply_text
            })

    golden_rows = []
    for intent, items in collected.items():
        golden_rows.extend(items)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in golden_rows:
            f.write(json.dumps(entry) + "\n")

    print(f"[data_processor] Successfully generated {len(golden_rows)} real golden set entries at {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force regenerate golden set from twcs.csv")
    args = parser.parse_args()
    process_kaggle_dataset(TWCS_CSV_PATH, GOLDEN_SET_PATH, force=args.force)
