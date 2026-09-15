# Engineering Report: Production AI Support Agent for @AmazonHelp

**Candidate:** Hiver SDE Intern Candidate  
**Target Brand:** `@AmazonHelp` (Official Customer Support on Twitter)  
**Dataset Source:** Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`, `twcs.csv`, 516 MB, ~2.8M rows)  
**Evaluation Benchmark:** 200 Hand-Labelled Authentic Customer Tweets from @AmazonHelp  
**Automated Metrics:** Multi-Class Intent Accuracy, Macro F1, 95% Bootstrap Confidence Intervals, Escalation Precision/Recall/FNR, Automation Rate, 5-Dimension LLM-as-a-Judge, Business Cost Modeling, and Human-LLM Calibration  

---

## 1. Problem Framing

Operating customer care on Twitter for a global e-commerce enterprise like **@AmazonHelp** involves unique challenges distinct from closed, authenticated in-app chat:
- **Public Brand Exposure & Legal Liability:** Every tweet interaction is public. Fabricating refund policies, promising unrealistic delivery times, or failing to address safety and fraud risks can lead to immediate public relations damage or regulatory exposure.
- **Strict Privacy & Anti-Doxxing Guardrails:** Customer PII (Personally Identifiable Information)—such as full order IDs, physical addresses, email addresses, and credit card numbers—must never be requested or exposed in public tweets. Inbound interactions requiring account lookup must be guided to secure Direct Messages (`https://amzn.to/help`).
- **Asymmetric Operational Risk in Escalation:** Falsely automating a critical security breach (e.g., account takeover, stolen card, driver assault) is exponentially more dangerous than escalating a routine tracking query. Consequently, **Escalation Recall** and minimizing the **Critical False Negative Rate (FNR)** are our primary safety priorities.

### What "Good" Customer Support Means for @AmazonHelp
1. **Accurate Intent Classification:** Reliably sorting incoming inquiries into one of 6 orthogonal categories: tracking inquiries (`order_status`), return/replacement requests (`refund_and_return`), charge discrepancies (`billing_and_charges`), courier delivery problems (`delivery_issue`), compromised logins (`account_and_security`), or general inquiries (`other`).
2. **Authentic Amazon Voice & Empathy:** Maintaining a professional, concise (< 280 characters), empathetic, and solution-driven brand tone aligned with standard `@AmazonHelp` guidelines.
3. **Actionable Resolution Paths:** Leaving the customer with an unambiguous next step (e.g., prompting for specific order details via secure DM) without making unverified policy commitments.
4. **Conservative Safety Guardrails:** Forcing escalation to human specialists whenever classification confidence is low ($< 0.60$) or calculated risk is elevated ($\ge 0.70$).

### What Was Deliberately Not Built (Out of Scope)
- **Autonomous Financial Ledger Execution:** The agent does not execute financial transactions (such as issuing refunds, debiting cards) or modifying active shipments in backend databases. In Twitter triage, agents coordinate intake and verification; back-office ledger mutations require authenticated 2SV sessions.
- **Multi-Brand Orchestration:** Focused exclusively on `@AmazonHelp` to ensure realistic domain modeling and deep prompt specialization rather than a diluted generic classifier spanning airlines, ride-sharing, and retail.
- **Complex External Microservices:** Avoided multi-container vector database setups (Milvus, Pinecone) or streaming brokers (Kafka) that would introduce setup friction and violate Hiver's strict requirement for an end-to-end reproducible pipeline running in **under 15 minutes**.

---

## 2. Data & Golden Set Sampling Methodology

### Real Thread Reconstruction from Kaggle `twcs.csv`
All evaluation data was extracted strictly from Kaggle's *Customer Support on Twitter* (`twcs.csv`, 516 MB). Synthetic or external datasets (such as Banking77) were prohibited.
1. **Filtering Brand Interactions:** Filtered all rows where `author_id == "AmazonHelp"` (over 160,000 tweets) and indexed their outbound replies.
2. **Thread Graph Reconstruction:** Followed `in_response_to_tweet_id` references to match customer inbound tweets with official brand responses, assembling multi-turn conversational context.
3. **Stratified Sampling:** Sampled exactly **200 real customer tweets** distributed evenly across the 6 intent classes to prevent class domination:
   - `delivery_issue`: 40 tweets (20.0%)
   - `order_status`: 35 tweets (17.5%)
   - `refund_and_return`: 35 tweets (17.5%)
   - `billing_and_charges`: 30 tweets (15.0%)
   - `account_and_security`: 30 tweets (15.0%)
   - `other`: 30 tweets (15.0%)
   - *Decisions:* 141 `auto_handle` (70.5%) vs. 59 `escalate` (29.5%).
4. **Gold Labeling:** Verified ground-truth labels based on authentic tweet contents, user distress signals, and canonical resolution strategies executed by `@AmazonHelp` staff.

---

## 3. Baseline Comparison & Evaluation Results

We benchmarked three distinct systems over the 200 real Kaggle tweets:
1. **Trivial Baseline (Majority Class):** Always predicts dataset majority intent (`delivery_issue`), always chooses `auto_handle`, and outputs a static generic canned reply.
2. **Simple Rule-Based Baseline:** Keyword-matching heuristics across the 6 intent classes, triggering escalation strictly on high-risk keywords.
3. **Our Proposed Agent (Gemini Core):** Few-shot in-context learning with 6 intent definitions, Amazon brand voice guardrails, hybrid risk scoring, and conservative fallback overrides.

### 3.1 Comparative Benchmark Summary

| Metric | Trivial Baseline | Simple Rule Baseline | **Our Proposed Agent** |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 20.00% [14.50%, 26.00%] | 72.50% [66.50%, 78.50%] | **78.00% [72.49%, 83.50%]** |
| **Macro F1 Score** | 0.0556 [0.0422, 0.0688] | 0.7170 [0.6539, 0.7746] | **0.7738 [0.7104, 0.8252]** |
| **Escalation Recall (Safety Critical)** | 0.00% | 40.68% | **59.32%** |
| **Escalation Precision** | 0.00% | 92.31% | **37.63%** |
| **Critical False Negative Rate (FNR)** | 100.00% | 59.32% | **40.68%** |
| **Automation Rate** | 100.00% | 87.00% | **53.50%** |
| **Mean Judge Score (0–10 Scale)** | 4.20 / 10 | 6.80 / 10 | **8.54 / 10** |
| **Operational Risk Cost / Ticket** | 4.64 units | 2.46 units | **1.77 units** |

*Note: All values above are generated dynamically by `src/evaluate.py` and `src/cost_analysis.py` and saved to `results/metrics.json` and `results/cost_analysis.json`.*

### 3.2 Per-Intent Performance Breakdown

| Intent Class | Precision | Recall | F1-Score | Support | Key Findings on Real @AmazonHelp Tweets |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `order_status` | 0.8621 | 0.7143 | 0.7812 | 35 | Strong precision (86.2%); 8 ambiguous tracking tweets were safely buffered into `other`. |
| `refund_and_return` | 0.9167 | 0.9429 | 0.9296 | 35 | Outstanding recall (94.3%) and F1 (0.930) on returns, cancellations, and refund claims. |
| `billing_and_charges` | 0.9286 | 0.4333 | 0.5909 | 30 | Exceptional precision (92.9%); lower recall due to informal phrasing without explicit fee terms. |
| `delivery_issue` | 0.9048 | 0.9500 | 0.9268 | 40 | Superb recall (95.0%); successfully captures 38 of 40 damaged/late courier parcels. |
| `account_and_security` | 1.0000 | 0.7000 | 0.8235 | 30 | **Perfect precision (1.0000)**; zero false security alarms; captures 21 of 30 logins/hacks. |
| `other` | 0.4483 | 0.8667 | 0.5909 | 30 | Acts as a safe catch-all buffer absorbing unstructured customer venting. |
| **Overall Accuracy** | — | — | **0.7800** | 200 | Evaluated on 200 authentic Kaggle customer tweets |
| **Macro Average** | **0.8434** | **0.7679** | **0.7738** | 200 | Unweighted balance across all 6 classes |
| **Weighted Average** | **0.8488** | **0.7800** | **0.7856** | 200 | Support-weighted multi-class aggregate |

### 3.3 Multi-Class Confusion Matrix

| Actual \ Predicted | `order_status` | `refund_and_return` | `billing_and_charges` | `delivery_issue` | `account_and_security` | `other` | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True `order_status`** | **25** | 0 | 0 | 2 | 0 | 8 | 35 |
| **True `refund_and_return`** | 0 | **33** | 0 | 0 | 0 | 2 | 35 |
| **True `billing_and_charges`** | 1 | 1 | **13** | 1 | 0 | 14 | 30 |
| **True `delivery_issue`** | 0 | 2 | 0 | **38** | 0 | 0 | 40 |
| **True `account_and_security`** | 0 | 0 | 1 | 0 | **21** | 8 | 30 |
| **True `other`** | 3 | 0 | 0 | 1 | 0 | **26** | 30 |

*Generated dynamically via `src/confusion_matrix.py` and saved to `results/confusion_matrix.csv`.*

### 3.4 Business Cost-Based Policy Optimization
Evaluating raw accuracy treats all classification mistakes equally. In an enterprise contact center, however, costs are asymmetric:
- **Cost of Human Review ($C_{\text{human}}$):** $1.0\text{ unit}$ (Triage cost for an agent to review a ticket).
- **Cost of Incorrect Auto-Reply ($C_{\text{wrong}}$):** $3.0\text{ units}$ (Customer friction, repeated inbounds).
- **Cost of Missed Escalation ($C_{\text{missed}}$):** $10.0\text{ units}$ (Catastrophic legal liability, fraud, or account takeover).

$$\text{Total Cost} = 1.0 \times N_{\text{escalate}} + 3.0 \times N_{\text{wrong auto-reply}} + 10.0 \times N_{\text{missed escalation}}$$

- **Trivial Baseline:** Always auto-handling results in 59 missed escalations ($59 \times 10 = 590$) and 113 wrong auto-replies ($113 \times 3 = 339$), yielding **4.64 units/ticket**.
- **Simple Rule Baseline:** Yields **2.46 units/ticket**.
- **Our Proposed Agent:** Total operational risk cost of 354.0 units, yielding **1.77 units/ticket** (**61.9% cost reduction vs. Trivial Baseline**).

---

## 4. Failure Analysis (Top 5 Real Failure Modes)

Analyzing false predictions from `results/predictions.jsonl` against gold labels reveals five distinct, reproducible failure modes on authentic Kaggle customer tweets:

### 1. Sarcasm & Passive-Aggressive Phrasing Misclassification
- **Tweet ID:** `amz_real_104231` (Representative: `amz_real_64042`)
- **Customer Text:** `"@AmazonHelp Oh brilliant, another day without my package arriving. Fantastic job Amazon! I've gone days without a bed because of this."`
- **Gold vs. Predicted:** Gold: `delivery_issue` (`escalate`) | Predicted: `other` (`auto_handle`)
- **Hypothesis:** Sarcastic praise keywords (`"brilliant"`, `"fantastic job"`) masked the underlying delivery failure. The model's sentiment analyzer perceived positive language, diluting the delivery delay tokens.
- **Actionable Fix:** Implement sentiment contrast scoring: flag co-occurrences of high-positive valence words with delivery delay keywords (`"without my package"`, `"gone days"`) to detect irony and force escalation.

### 2. Keyword Over-Escalation on Benign Exploratory Questions
- **Tweet ID:** `amz_real_204112`
- **Customer Text:** `"@AmazonHelp Is it a scam if a third party seller asks for extra shipping fees outside of Amazon?"`
- **Gold vs. Predicted:** Gold: `billing_and_charges` (`auto_handle`) | Predicted: `billing_and_charges` (`escalate`)
- **Hypothesis:** The keyword rule detected `"scam"` and immediately boosted the risk score to $\ge 0.70$, forcing an escalation. The customer, however, was asking an informational policy question rather than reporting active fraud on their personal account.
- **Actionable Fix:** Replace static keyword matching with syntactic dependency parsing: check whether the risk keyword is an object of a hypothetical condition (`"if a seller asks"`) versus an active incident report (`"I have been scammed"`).

### 3. Ambiguous Refund vs. Delivery Boundaries
- **Tweet ID:** `amz_real_309114` (Representative: `amz_real_64214`)
- **Customer Text:** `"@AmazonHelp Been waiting for my order for over a week now, the first time it said delivered, now the replacement said attempted delivery... customer service are useless."`
- **Gold vs. Predicted:** Gold: `delivery_issue` | Predicted: `refund_and_return`
- **Hypothesis:** The customer mentioned `"replacement"`, which heavily weighted the return/refund head, even though the root cause was chronic courier delivery failure (`"attempted delivery"`, `"waiting for over a week"`).
- **Actionable Fix:** Introduce hierarchical intent taxonomy: resolve courier delivery logistics as a primary operational head before parsing fulfillment preferences (replacement vs. refund).

### 4. Account Access vs. Order Status Hierarchy Friction
- **Tweet ID:** `amz_real_401221`
- **Customer Text:** `"@AmazonHelp I can't log into my account to check where my package is! It's urgent."`
- **Gold vs. Predicted:** Gold: `account_and_security` (`escalate`) | Predicted: `order_status` (`auto_handle`)
- **Hypothesis:** The model focused on the end objective (`"check where my package is"`) rather than the blocker (`"can't log into my account"`), routing the ticket to an automated tracking DM flow that the customer cannot use while locked out.
- **Actionable Fix:** Implement a dependency precedence rule: authentication and security blockers must always take priority over downstream informational requests.

### 5. Multi-Turn Context Fragmentation Loss
- **Tweet ID:** `amz_real_502991` (Representative: `amz_real_75839`)
- **Customer Text:** `"@AmazonHelp Called up agent dipika supervisor ajay and manager Sohail is unable to do nothing they said wait for 3 more days you will receive ur product !! Why 3 more days if delivery date is for today ?"`
- **Gold vs. Predicted:** Gold: `order_status` | Predicted: `other`
- **Hypothesis:** Customer named multiple prior support agents, causing the intent classifier to view the tweet as conversational feedback or staff harassment (`other`) rather than an overdue package inquiry.
- **Actionable Fix:** Normalize tweets by stripping agent handles and employee names during pre-processing, allowing the semantic model to focus on the core query (`"Why 3 more days if delivery date is for today?"`).

---

## 5. What is Misleading About My Headline Number?

Our headline **78.00% Intent Accuracy** and **8.54 / 10 Mean Judge Score** look strong, but in an enterprise engineering review, honesty about limitations is essential:

1. **Stratified Test Set vs. Real-World Class Imbalance:** In production Twitter traffic, `delivery_issue` accounts for nearly 50% of inbounds, while `account_and_security` is rare (< 5%). Our evaluation set was stratified (20% delivery, 15% account) to evaluate rare classes fairly. In production, unweighted accuracy would drift higher, while rare-class failures would be hidden.
2. **LLM-as-a-Judge Politeness Bias:** The automated LLM judge awards high marks (mean 8.54/10) to polite, well-structured replies that direct customers to DM. However, human customers who have already tweeted three times are often infuriated by another generic request to *"please send us a direct message."*
3. **Conservative Escalation Precision (37.63%):** To achieve a high safety recall (59.32%) on critical risks, the agent escalates aggressively. A precision of 37.63% indicates that ~62% of escalated tickets could have been handled automatically. In a high-volume call center, this creates human queue congestion.
4. **Single-Turn vs. Multi-Turn Truncation:** Many tweets in `twcs.csv` are follow-ups. When evaluating isolated tweets without their 5 preceding conversation turns, both the agent and human evaluators must infer intent from partial context.

---

## 6. Next Steps with One More Week & Decision Log Summary

### 1-Week Engineering Expansion Plan
1. **Hierarchical Intent Taxonomy:** Build a two-stage classifier that first detects the operational domain (`logistics`, `finance`, `security`), then isolates customer intent (`tracking`, `refund`, `complaint`).
2. **Full Multi-Turn Context Graph:** Leverage the complete `in_response_to_tweet_id` tree from `twcs.csv` to inject up to 5 prior conversational turns into the prompt context.
3. **Active Learning & Review Queue:** Automatically route tickets with confidence between $0.40$ and $0.60$ to a human-in-the-loop review interface to continuously grow the golden evaluation set.
4. **Few-Shot RAG with Historical Brand Tweets:** Dynamically retrieve the top-3 most similar authentic `@AmazonHelp` historical replies using vector similarity to match canonical Amazon phrasing.

### Summary of Architectural Decisions
A detailed breakdown of all **15 non-obvious engineering decisions**—spanning hybrid risk scoring, prompt template formatting, dual-mode client design, and cost matrix optimization—is documented in [`DECISIONS.md`](../DECISIONS.md).
