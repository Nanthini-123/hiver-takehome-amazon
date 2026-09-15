import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"

TWCS_CSV_PATH = DATA_RAW_DIR / "twcs.csv"
GOLDEN_SET_PATH = DATA_PROCESSED_DIR / "golden_set.jsonl"
BRAND_THREADS_PATH = DATA_PROCESSED_DIR / "brand_threads.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
BASELINE_PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.jsonl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
JUDGE_SCORES_PATH = RESULTS_DIR / "judge_scores.jsonl"

AGENT_PROMPT_PATH = SRC_DIR / "agent_master_prompt.txt"
JUDGE_PROMPT_PATH = SRC_DIR / "judge_master_prompt.txt"

BRAND_HANDLE = "AmazonHelp"
CONFIDENCE_THRESHOLD = 0.6
RISK_THRESHOLD = 0.7

HIGH_RISK_KEYWORDS = [
    "fraud", "lawyer", "attorney", "sue", "court", "legal", 
    "hacked", "stolen", "unauthorized", "police", "scam"
]

INTENTS = [
    "order_status",
    "refund_and_return",
    "billing_and_charges",
    "delivery_issue",
    "account_and_security",
    "other"
]
