"""
Business Cost-Based Policy Optimization and Operational Risk Analysis.
Evaluates the cost matrix for @AmazonHelp customer support triage:
  • Cost of Human Handling (Escalate) = 1.0 unit
  • Cost of Incorrect Auto-Reply (Brand friction) = 3.0 units
  • Cost of Missed Escalation (Catastrophic legal/privacy hazard) = 10.0 units
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREDICTIONS_PATH, BASELINE_PREDICTIONS_PATH, GOLDEN_SET_PATH, RESULTS_DIR


def evaluate_business_costs():
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden = {json.loads(line)["thread_id"]: json.loads(line) for line in f if line.strip()}

    with open(BASELINE_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        base_preds = {json.loads(line)["thread_id"]: json.loads(line) for line in f if line.strip()}

    def calculate_cost(pred_items, get_dec, get_intent):
        cost = 0.0
        details = {"human_review": 0, "incorrect_auto_reply": 0, "missed_escalation": 0}

        for p in pred_items:
            tid = p["thread_id"]
            gold_dec = golden[tid]["gold_decision"]
            gold_int = golden[tid]["gold_intent"]
            p_dec = get_dec(p)
            p_int = get_intent(p)

            if p_dec == "escalate":
                cost += 1.0  # Human agent review cost
                details["human_review"] += 1
            elif p_dec == "auto_handle" and gold_dec == "escalate":
                cost += 10.0  # Missed escalation penalty
                details["missed_escalation"] += 1
            elif p_dec == "auto_handle" and p_int != gold_int:
                cost += 3.0  # Incorrect auto-reply friction penalty
                details["incorrect_auto_reply"] += 1

        avg_cost = round(cost / len(pred_items), 2)
        return round(cost, 2), avg_cost, details

    # Agent cost
    agent_cost, agent_avg, agent_details = calculate_cost(
        preds,
        lambda p: p["predicted_decision"],
        lambda p: p["predicted_intent"]
    )

    # Simple baseline cost
    simple_cost, simple_avg, simple_details = calculate_cost(
        preds,
        lambda p: base_preds[p["thread_id"]]["simple"]["decision"],
        lambda p: base_preds[p["thread_id"]]["simple"]["intent"]
    )

    # Trivial baseline cost
    triv_cost, triv_avg, triv_details = calculate_cost(
        preds,
        lambda p: base_preds[p["thread_id"]]["trivial"]["decision"],
        lambda p: base_preds[p["thread_id"]]["trivial"]["intent"]
    )

    cost_summary = {
        "cost_weights": {
            "human_handling": 1.0,
            "incorrect_auto_reply": 3.0,
            "missed_escalation": 10.0
        },
        "agent": {
            "total_cost": agent_cost,
            "cost_per_ticket": agent_avg,
            "breakdown": agent_details
        },
        "simple_baseline": {
            "total_cost": simple_cost,
            "cost_per_ticket": simple_avg,
            "breakdown": simple_details
        },
        "trivial_baseline": {
            "total_cost": triv_cost,
            "cost_per_ticket": triv_avg,
            "breakdown": triv_details
        }
    }

    out_file = RESULTS_DIR / "cost_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cost_summary, f, indent=2)

    print("\n" + "=" * 65)
    print("           BUSINESS COST-BASED POLICY OPTIMIZATION            ")
    print("=" * 65)
    print(f"Policy Weighting: Human=1.0 | Wrong Auto-Reply=3.0 | Missed Escalation=10.0\n")
    print(f"• Proposed Agent    : Total Cost = {agent_cost:6.1f} units | Cost/Ticket = {agent_avg:.2f} units/ticket")
    print(f"• Simple Baseline   : Total Cost = {simple_cost:6.1f} units | Cost/Ticket = {simple_avg:.2f} units/ticket")
    print(f"• Trivial Baseline  : Total Cost = {triv_cost:6.1f} units | Cost/Ticket = {triv_avg:.2f} units/ticket")
    print(f"\n[cost_analysis] Detailed report saved to {out_file}\n")


if __name__ == "__main__":
    evaluate_business_costs()
