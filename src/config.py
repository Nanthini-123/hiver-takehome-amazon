"""
Central configuration module for @AmazonHelp customer support pipeline.
Defines relative paths, model thresholds, high-risk keywords, intent labels,
and explicit sampling parameters conforming to the Hiver Take-Home Checklist.
"""

import os
from pathlib import Path

# Project root directory anchored via relative path resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories and paths
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"
LABELS_DIR = BASE_DIR / "labels"

TWCS_CSV_PATH = DATA_RAW_DIR / "twcs.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.csv"
GOLDEN_SET_PATH = DATA_PROCESSED_DIR / "golden_set.jsonl"
BRAND_THREADS_PATH = DATA_PROCESSED_DIR / "brand_threads.jsonl"
CODEBOOK_PATH = LABELS_DIR / "codebook.json"

# Results paths
PREDICTIONS_PATH = RESULTS_DIR / "predictions.jsonl"
BASELINE_PREDICTIONS_PATH = RESULTS_DIR / "baseline_predictions.jsonl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
JUDGE_SCORES_PATH = RESULTS_DIR / "judge_scores.jsonl"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.csv"
PER_INTENT_BREAKDOWN_PATH = RESULTS_DIR / "per_intent_breakdown.json"
COST_ANALYSIS_PATH = RESULTS_DIR / "cost_analysis.json"
HUMAN_SCORES_PATH = RESULTS_DIR / "human_scores.json"
HUMAN_VS_JUDGE_PATH = RESULTS_DIR / "human_vs_judge.json"

# Prompts
AGENT_PROMPT_PATH = SRC_DIR / "agent_master_prompt.txt"
JUDGE_PROMPT_PATH = SRC_DIR / "judge_master_prompt.txt"

# Target Brand & Calibration Parameters
TARGET_BRAND = "AmazonHelp"
BRAND_HANDLE = "AmazonHelp"
CONFIDENCE_THRESHOLD = 0.60
RISK_THRESHOLD = 0.70

# Explicit Sampling Notes (Checklist Compliance)
SAMPLING_RANDOM_SEED = 42
TOTAL_GOLDEN_EXAMPLES = 200
DATASET_TIME_RANGE = "2017-10 to 2017-12 (Authentic TWCS range)"
STRATIFICATION_QUOTAS = {
    "delivery_issue": 40,
    "order_status": 35,
    "refund_and_return": 35,
    "billing_and_charges": 30,
    "account_and_security": 30,
    "other": 30
}

# High-Risk Safety & Escalation Triggers
HIGH_RISK_KEYWORDS = [
    "fraud", "lawyer", "attorney", "sue", "court", "legal", 
    "hacked", "stolen", "unauthorized", "police", "scam"
]

# Canonical 6 Intent Classes
INTENTS = [
    "order_status",
    "refund_and_return",
    "billing_and_charges",
    "delivery_issue",
    "account_and_security",
    "other"
]
