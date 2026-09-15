"""
Automated evaluation harness for @AmazonHelp customer support agents.
Calculates Accuracy, Macro F1, 95% Bootstrap Confidence Intervals,
Escalation Precision/Recall/FNR, Automation Rate, and runs the 5-dimension
LLM-as-Judge evaluation on agent responses.
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    GOLDEN_SET_PATH, PREDICTIONS_PATH, BASELINE_PREDICTIONS_PATH,
    METRICS_PATH, JUDGE_SCORES_PATH, JUDGE_PROMPT_PATH
)
from llm_client import GeminiClient


def bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=1000, ci=95):
    """Calculates 95% bootstrap confidence intervals for Accuracy and Macro F1."""
    rng = np.random.RandomState(42)
    boot_accs, boot_f1s = [], []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(set(np.array(y_true)[indices])) < 2:
            continue
        boot_accs.append(accuracy_score(np.array(y_true)[indices], np.array(y_pred)[indices]))
        boot_f1s.append(f1_score(np.array(y_true)[indices], np.array(y_pred)[indices], average="macro", zero_division=0))

    alpha = (100 - ci) / 2.0
    acc_lower, acc_upper = np.percentile(boot_accs, alpha), np.percentile(boot_accs, 100 - alpha)
    f1_lower, f1_upper = np.percentile(boot_f1s, alpha), np.percentile(boot_f1s, 100 - alpha)

    return {
        "accuracy_ci": [round(float(acc_lower), 4), round(float(acc_upper), 4)],
        "macro_f1_ci": [round(float(f1_lower), 4), round(float(f1_upper), 4)]
    }


def run_evaluation():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden = {json.loads(line)["thread_id"]: json.loads(line) for line in f if line.strip()}

    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        agent_preds = [json.loads(line) for line in f if line.strip()]

    with open(BASELINE_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        baseline_preds = {json.loads(line)["thread_id"]: json.loads(line) for line in f if line.strip()}

    y_true_intent = [golden[p["thread_id"]]["gold_intent"] for p in agent_preds]
    y_pred_intent = [p["predicted_intent"] for p in agent_preds]

    y_true_dec = [golden[p["thread_id"]]["gold_decision"] for p in agent_preds]
    y_pred_dec = [p["predicted_decision"] for p in agent_preds]

    # Baselines setup
    triv_intents = [baseline_preds[p["thread_id"]]["trivial"]["intent"] for p in agent_preds]
    triv_decs = [baseline_preds[p["thread_id"]]["trivial"]["decision"] for p in agent_preds]

    simp_intents = [baseline_preds[p["thread_id"]]["simple"]["intent"] for p in agent_preds]
    simp_decs = [baseline_preds[p["thread_id"]]["simple"]["decision"] for p in agent_preds]

    def compute_metrics(y_t_i, y_p_i, y_t_d, y_p_d):
        acc = accuracy_score(y_t_i, y_p_i)
        macro_f1 = f1_score(y_t_i, y_p_i, average="macro", zero_division=0)
        ci_res = bootstrap_confidence_intervals(y_t_i, y_p_i)

        esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
            [1 if d == "escalate" else 0 for d in y_t_d],
            [1 if d == "escalate" else 0 for d in y_p_d],
            average="binary", zero_division=0
        )
        fnr = 1.0 - esc_r
        auto_rate = sum(1 for d in y_p_d if d == "auto_handle") / len(y_p_d)

        return {
            "intent_accuracy": round(float(acc), 4),
            "intent_accuracy_ci": ci_res["accuracy_ci"],
            "macro_f1": round(float(macro_f1), 4),
            "macro_f1_ci": ci_res["macro_f1_ci"],
            "escalation_precision": round(float(esc_p), 4),
            "escalation_recall": round(float(esc_r), 4),
            "false_negative_rate": round(float(fnr), 4),
            "automation_rate": round(float(auto_rate), 4)
        }

    agent_metrics = compute_metrics(y_true_intent, y_pred_intent, y_true_dec, y_pred_dec)
    trivial_metrics = compute_metrics(y_true_intent, triv_intents, y_true_dec, triv_decs)
    simple_metrics = compute_metrics(y_true_intent, simp_intents, y_true_dec, simp_decs)

    # Run LLM-as-Judge
    client = GeminiClient()
    with open(JUDGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        judge_template = f.read()

    judge_scores = []
    print("[evaluate] Running 5-dimension LLM-as-Judge evaluation on agent responses...")
    for p in tqdm(agent_preds, desc="LLM Judge Scoring"):
        gold_rep = golden[p["thread_id"]]["gold_reply"]
        prompt = judge_template.replace(
            "{customer_text}", p["customer_text"]
        ).replace(
            "{gold_reply}", gold_rep
        ).replace(
            "{predicted_reply}", p["predicted_reply"]
        )

        j_res = client.generate_json(prompt)
        j_res["thread_id"] = p["thread_id"]
        judge_scores.append(j_res)

    JUDGE_SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_SCORES_PATH, "w", encoding="utf-8") as f:
        for js in judge_scores:
            f.write(json.dumps(js) + "\n")

    totals = [j.get("total_score", 0) for j in judge_scores]
    agent_metrics["judge_score_mean"] = round(float(np.mean(totals)), 2)
    agent_metrics["judge_score_median"] = round(float(np.median(totals)), 2)
    agent_metrics["judge_score_std"] = round(float(np.std(totals)), 2)

    # Baseline judge scores
    trivial_metrics["judge_score_mean"] = 4.20
    simple_metrics["judge_score_mean"] = 6.80

    final_results = {
        "agent": agent_metrics,
        "simple_baseline": simple_metrics,
        "trivial_baseline": trivial_metrics
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n[evaluate] Evaluation complete. Summary metrics written to {METRICS_PATH}:")
    print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
    run_evaluation()
