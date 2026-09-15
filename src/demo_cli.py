"""
Interactive CLI Demo Script for @AmazonHelp AI Support Agent.
Allows engineering reviewers and users to test arbitrary customer tweets
in real-time and observe intent classification, risk assessment, decision routing,
and drafted brand responses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import GeminiClient
from config import AGENT_PROMPT_PATH, HIGH_RISK_KEYWORDS, CONFIDENCE_THRESHOLD, RISK_THRESHOLD


def run_live_demo():
    print("==================================================")
    print("  @AmazonHelp AI Support Agent — Interactive CLI  ")
    print("==================================================")
    print("Type a customer tweet to test agent triage in real-time.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    client = GeminiClient()
    with open(AGENT_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    while True:
        try:
            user_tweet = input("\nEnter customer tweet (or 'exit' to quit): ").strip()
            if user_tweet.lower() in ["exit", "quit"]:
                print("\nExiting interactive demo. Thank you!")
                break
            if not user_tweet:
                continue

            # Safe template replacement (prevents curly brace KeyError on raw JSON in prompt)
            prompt = prompt_template.replace(
                "{customer_text}", user_tweet
            ).replace(
                "{context_text}", ""
            )

            result = client.generate_json(prompt)

            # Apply production risk scoring & safety guardrails
            text_lower = user_tweet.lower()
            keyword_risk = 0.5 if any(k in text_lower for k in HIGH_RISK_KEYWORDS) else 0.0
            calculated_risk = max(
                result.get("risk_score", 0.0),
                round(1.0 - result.get("intent_confidence", 1.0), 2),
                keyword_risk
            )
            result["risk_score"] = round(calculated_risk, 2)

            if result.get("intent_confidence", 0.0) < CONFIDENCE_THRESHOLD or result.get("risk_score", 0.0) >= RISK_THRESHOLD:
                result["decision"] = "escalate"
                result["escalation_reason"] = f"Triggered safety guardrail: Confidence={result.get('intent_confidence')}, Risk={result.get('risk_score')}"
                result["reply"] = "We want to help with your concern safely. Please send us a direct message so a specialized representative can assist."

            print("\n--- AGENT RESPONSE ---")
            print(f"Predicted Intent : {result.get('intent')}")
            print(f"Confidence       : {result.get('intent_confidence')}")
            print(f"Risk Score       : {result.get('risk_score')}")
            print(f"Decision         : {result.get('decision', 'auto_handle').upper()} ({result.get('escalation_reason')})")
            print(f"Drafted Reply    : \"{result.get('reply')}\"")
            print("----------------------")

        except KeyboardInterrupt:
            print("\nExiting interactive demo. Goodbye!")
            break


if __name__ == "__main__":
    run_live_demo()
