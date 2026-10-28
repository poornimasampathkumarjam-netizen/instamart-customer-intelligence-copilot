# Instamart Customer Intelligence & Support Copilot

## Overview

This is a reproducible, offline capstone pipeline over fictional synthetic Instamart-style data. It cleans customer, product and order exports; writes trusted CSV tables and a SQLite database; performs SQL and exploratory analysis; builds RFM segments and a lapse model; triages support tickets with privacy and cost guardrails; retrieves policy passages with TF-IDF; and runs a bounded, resumable refund agent. All monetary values are in INR. The supplied `generate_data.py` and `mock_llm.py` are included unchanged; there are no API keys, hosted model calls, or network requests.

## Reproduce

Use Python 3.11 or newer from the repository root. The pinned dependencies in `requirements.txt` are the versions used for the recorded run.

1. Create an isolated environment: `python -m venv .venv`
2. Activate it on Windows: `.venv\Scripts\activate`; on macOS/Linux: `source .venv/bin/activate`
3. Install dependencies: `python -m pip install --upgrade pip` and `python -m pip install -r requirements.txt`
4. Regenerate all synthetic raw inputs: `python generate_data.py`
5. Run Parts A–D: `python run_all.py`
6. Validate the Part A cleaning, database and SQL checkpoints: `python validate_data.py`
7. Validate the Part B–D outputs and the agent resume demonstration: `python validate_outputs.py`

The validation commands print exactly `ALL CHECKPOINTS PASSED` and `ALL OUTPUT CHECKS PASSED`, respectively. The generator uses fixed seeds, as do the clustering, split, and model experiments. `run_all.py` recreates the clean data, database, model and output artifacts, triage results, RAG evaluation, refund ledger, and checkpoints. A normal full run completes in roughly half a minute on the tested environment, depending on hardware. `.gitignore` excludes virtual environments, `.env`, and Python bytecode; generated data and required output artifacts are intentionally committed.

## Part A

### Cleaning and storage

**Customers.** The raw export has 1,200 customers. City trimming/title-casing changed 476 city values and leaves exactly six cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191 and Mumbai 184. Gender normalization changed 495 values; the final values are female 610, male 560 and unknown 30. There were 61 missing or non-numeric ages and 6 numeric ages outside the allowed 10–100 range (67 invalid/missing ages in total); those were set to missing, then filled with the city median using `groupby('city').transform('median')`. The clean table has zero missing ages, with ages from 18 to 60. `signup_date` is parsed as a date; no signup dates were unparseable.

**Orders.** The raw export contains 13,084 rows. Cleaning follows the required order: parse `order_ts`; remove 119 exact duplicate rows; sort by timestamp and keep the latest record per `order_id`, dropping a further 656 older duplicate-order records; convert `order_amount` to numeric; fill 112 unparseable/missing amounts with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`; set 105 negative delivery durations to missing; and fill the 669 remaining missing delivery durations on completed orders with the completed-order median for that product category using `groupby(...).transform('median')`. Cancelled orders are not assigned invented delivery times. The final order table has 12,309 unique orders: 11,110 completed and 1,199 cancelled, with no pending rows. All order amounts are present; completed delivery durations have no missing values, while all 1,199 cancelled rows have missing delivery duration. Completed-order revenue is **INR 2,998,668.80**.

The three clean tables are saved in `data/clean/` and the SQLite database `data/instamart.db` has `customers`, `products`, and `orders` tables. `validate_data.py` asserts the core cleaning counts, delivery-time rules, revenue, table existence, and SQL query results. It includes more than eight independent checkpoints.

### SQL analysis

All seven statements in `sql/queries.sql` are executed through `pandas.read_sql_query` against `data/instamart.db`.

| Query | Method and result |
|---|---|
| Q1 | `WHERE` completed orders with amount above INR 500, then `HAVING` customer totals above INR 8,000: **30 customers**. |
| Q2 | Three-table join for city/category revenue: **30 rows**, total **INR 2,998,668.80**. Largest cell: Bengaluru / Electronics, **INR 233,909.45**. |
| Q3 | Scalar subquery for products above average catalogue price: **11 products**. |
| Q4 | `LEFT JOIN` customers without a completed order: **25 customers**. |
| Q5 | Two CTEs for customer spend and average spend: **550 customers** above the average; average completed spend is **INR 2,552.06**. |
| Q6 | `RANK`, `DENSE_RANK`, and `ROW_NUMBER` for **1,175 customers** with completed spend; `RANK` differs from `ROW_NUMBER` on **8 rows**. |
| Q7 | Daily completed revenue and cumulative window sum: **183 daily rows**; final cumulative revenue **INR 2,998,668.80**. |

The highest-spend customer in each city is Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55) and Pune C0426 (INR 10,377.10). `RANK` assigns tied rows the same rank and leaves a gap after a tie, whereas `ROW_NUMBER` assigns every row a unique sequential number.

### pandas cross-check and exploratory analysis

A pandas pivot table with margins independently totals completed-order revenue at **INR 2,998,668.80**, matching Q2. The city-level sum/mean/count are: Bengaluru INR 523,643.40 / INR 276.47 / 1,894; Chennai INR 463,629.15 / INR 272.72 / 1,700; Delhi INR 502,829.80 / INR 243.62 / 2,064; Hyderabad INR 528,589.90 / INR 284.95 / 1,855; Mumbai INR 479,058.50 / INR 297.00 / 1,613; Pune INR 500,918.05 / INR 252.48 / 1,984.

Completed order amounts describe basket value and are strongly right-skewed: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80 and Q3 INR 297.00. The 1.5×IQR fences are INR −283.50 and INR 645.30; there are no observations below the lower fence and 897 above the upper fence. The high values may represent large baskets and should be investigated rather than automatically deleted. The largest absolute pairwise correlation is order amount versus discount percentage at about −0.095 (order amount versus quantity is about +0.087); a correlation of this size does not establish causation. The exploratory summary is also written to `outputs/eda_summary.json`.

## Part B

### RFM and K-means segments

RFM uses completed orders at snapshot `2025-10-01 00:00:00`. The 1,175 customers with at least one completed order have median recency 19 days (minimum 0, maximum 180), mean frequency 9.46 (maximum 33), and mean monetary value INR 2,552.06 (maximum INR 14,766.55). StandardScaler is fit to the three RFM columns. K-means is fit for every K from 2 through 8 with `n_init=10` and `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.761 | 0.465 |
| 3 | 1268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K=4 is selected because it has the best silhouette among the tested values and a useful four-group business interpretation. Inertia normally decreases as K increases because more centroids can fit the data more closely; K=8 therefore has lower inertia than K=2, but inertia alone rewards adding clusters and does not establish that the extra groups are meaningful or actionable.

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Suggested action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Steady Value | 436 | 28.86 | 3.77 | 1,057.62 | Use low-cost cross-sell and replenishment reminders. |
| 1 | Premium High-Value | 138 | 24.12 | 6.77 | 7,129.70 | Offer tailored service and carefully measured loyalty benefits. |
| 2 | Frequent Essentials | 419 | 8.77 | 19.40 | 3,410.02 | Encourage repeat baskets with relevant replenishment offers. |
| 3 | At-Risk / Cooling | 182 | 109.04 | 2.20 | 685.99 | Test a modest win-back offer and monitor response. |

Segment 1 has the highest mean monetary value and is marked Premium. Cluster IDs are arbitrary labels, not an ordered score. PCA on scaled RFM explains **0.634** and **0.218** of variance in the first two components, **0.852** combined. `outputs/customer_segments.csv` contains the required customer-level RFM, segment, and Premium fields; K-means metrics are in `outputs/kmeans_scores.csv`.

### Lapse model

The cohort consists of customers with a completed order before `2025-09-01`; a customer is labelled lapsed if they have no completed order during September 2025. There are **1,156** customers and **443 lapsed (38.32%)**, split into **924 training** and **232 test** rows with `test_size=0.2`, `stratify=y`, and `random_state=42`. The always-not-lapsed baseline has **61.6% accuracy** and lapsed-class F1 **0.000**.

The required features are built only from orders before the cutoff and clean customer attributes: recency, frequency, monetary value, average discount, cancellation rate among final-status orders, average delivery duration, favourite category, city, age, gender and tenure. The task-3 segment labels are not features. The category encoder and numeric scaler are inside the modelling pipeline and are fitted only on training folds/rows; no encoder, scaler or resampler saw test rows. All feature columns are checked/fill-handled for missing values before fitting. The fitted scaler's training-only frequency mean is **8.116**, versus **8.029** for the full cohort; they differ because the training split is only a subset of the cohort. No order dated on or after `2025-09-01` enters the model features.

The five-fold F1 grid searches use LogisticRegression (`max_iter=1000`, C in 0.01, 0.1, 1, 10, 100) and RandomForestClassifier (`n_estimators=100`, `random_state=42`, all nine combinations of `max_depth` 3/5/8 and `min_samples_leaf` 5/10/20).

| Model | Best parameters | Best CV F1 | Test confusion matrix `[[TN, FP], [FN, TP]]` | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C = 0.1 | 0.6392 | `[[105, 38], [17, 72]]` | 0.655 | 0.809 | 0.724 |
| Random forest | depth = 8, leaf = 10 | 0.6718 | `[[112, 31], [21, 68]]` | 0.687 | 0.764 | 0.723 |

The selected persisted model is logistic regression: its test F1 is marginally higher (0.7236 versus 0.7234) and its recall is higher, which means it catches more customers who lapse. Missing a lapsing customer loses a retention opportunity; targeting a loyal customer unnecessarily spends an incentive and risks annoyance. The logistic model makes 38 false-positive offers versus 31 for the forest, so the choice depends on whether the business prioritizes catching more lapses or reducing unnecessary offers.

The class-balance experiment oversamples only the lapsed training rows with replacement until the training classes are equal, then evaluates on the unchanged test set. Balanced logistic regression gives precision **0.587**, recall **0.910** and F1 **0.714**, versus the original logistic model's 0.655 / 0.809 / 0.724. Recall moves most: more actual lapsers are caught, at the cost of more loyal customers being targeted and lower precision.

The five largest random-forest importances are frequency (0.320), monetary (0.220), recency (0.087), average discount (0.070) and cancellation rate (0.070). In the logistic model, frequency has a negative coefficient (−1.281; more prior orders are associated with lower lapse odds), favourite category Electronics has a negative coefficient (−0.378 relative to the dropped reference category), and city Pune has a positive coefficient (+0.113 relative to the dropped reference city). These are model associations, not causal effects. The selected pipeline is saved to `outputs/lapse_model.joblib`, reloaded, and its test predictions are asserted to match exactly.

The top-10 Premium customers from the test set, ranked by predicted lapse probability, are:

| Customer | Lapse probability | Actually lapsed |
|---|---:|---:|
| C0577 | 0.575 | 0 |
| C0102 | 0.570 | 1 |
| C0697 | 0.547 | 0 |
| C0942 | 0.510 | 0 |
| C0831 | 0.496 | 0 |
| C0324 | 0.477 | 0 |
| C0162 | 0.476 | 0 |
| C0509 | 0.476 | 0 |
| C0977 | 0.445 | 1 |
| C0299 | 0.445 | 0 |

Two of these ten customers actually lapsed. The Premium flag is based on whole-window RFM that includes the September label window, so the retention list is illustrative and not a leakage-free evaluation. The full list is in `outputs/retention_list.csv`; model diagnostics are in `outputs/model_diagnostics.json`.

## Part C

### Guardrails, redaction and validation

Tickets are processed in input order. Empty/whitespace text is skipped (**4 tickets**) and text over `MAX_INPUT_TOKENS = 400` is flagged before any model call (**4 tickets**, estimated at 734 tokens each). Email, phone and order identifiers are redacted in that order using the required regular expressions and placeholders. Across all 164 tickets the substitutions are **55 emails**, **91 phone numbers**, and **132 order IDs**, **278 total**. The code re-applies every pattern to every redacted text and asserts that none remain; the model's user message is exactly the redacted ticket text.

Both system prompts contain `CLASSIFY`, the four allowed categories and a JSON-only output contract. The zero-shot prompt does not contain `EXAMPLES:`; the few-shot prompt contains labelled examples including fraud. Every transient error is retried with exponential backoff and jitter (BASE_DELAY 0.05 seconds, at most three retries). The two full comparisons each recover all **19** simulated transient errors. JSON output is accepted only for an allowed category; otherwise the fallback word extraction uses `re.findall(r"[a-z]+", text.lower())`, and any invalid or ambiguous output goes to human review rather than being guessed.

| Full-run status | Tickets |
|---|---:|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |

### Zero-shot versus few-shot

| Prompt | Classified | Accuracy over classified tickets | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 6,436 | 43.78 |
| Few-shot | 147 | 100.0% (147/147) | 10,336 | 70.31 |

The zero-shot confusion table below uses true category as rows and predicted category as columns.

| True \ Predicted | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

Few-shot prompting costs an extra **3,900 tokens** overall (about **26.53 extra tokens per classified ticket**, or 60.6% more than zero-shot) and improves measured accuracy by **14.3 percentage points**. That cost is worthwhile where misrouting a fraud or technical issue has high downstream cost; in a low-risk, high-volume queue, zero-shot plus human review of invalid outputs may be preferable. The confusion table makes the zero-shot failure mode visible: **11 fraud tickets and 10 technical tickets are predicted as billing**. Outputs include `outputs/triage_zero_shot.csv` and `outputs/triage_summary.json`.

### Budget and routing

The zero-shot budget demo uses **6,000 tokens** and stops before making a call once the running total has reached the budget; **10 tickets** are marked `budget_exhausted`. The final few-shot output uses a 100,000-token budget and has 164 rows. Routing follows the specified precedence: empty, long, invalid-output and budget-exhausted tickets go to `human_review` with NORMAL priority; fraud goes to `security_team` with HIGH priority; otherwise Premium customers go to `premium_desk` with HIGH priority; all remaining valid tickets go to `support_queue` with NORMAL priority. Final route counts are support queue **102**, security team **34**, human review **17**, and Premium desk **11**. See `outputs/triage_budget_demo.csv` and `outputs/triage_results.csv`.

## Part D

### Policy RAG

The five fictional policy documents are split into 40-word chunks with 10-word overlap, yielding **15 chunks**: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer()` is fit to the chunk texts; a query is transformed with that vectorizer and ranked using cosine similarity. TF measures within-chunk term frequency; IDF downweights terms common across the corpus. TF-IDF is lexical, so semantically equivalent wording with few shared terms can retrieve poorly.

On the 14-question evaluation, 12 questions are answerable and 2 are not covered. With overlap, expected document recall is **12/12**, expected-fact substring matches are **12/12**, and fallback answers are **2/2**. With zero overlap, expected document recall remains **12/12**, expected-fact matches fall to **10/12**, and fallback answers remain **2/2**. Two zero-overlap failures illustrate boundary effects: “What delivery time makes a completed order eligible for a full refund?” loses the exact “more than 45 minutes” fact, and “How soon is money returned when a payment fails but is deducted?” loses the “48 hours” fact. The chunks with the answer's words are split across boundaries, so retrieval can find the right policy document without returning the complete answer fact.

### Refund tools and ReAct agent

The tool map contains `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`, each with a JSON-style parameter specification in `TOOLS_SPEC`. Only completed orders are eligible; delivery in more than 45 minutes qualifies for a full refund, delivery in 45 minutes or less gives INR 0, and amounts above INR 1,000 are escalated rather than automatically issued. Simulated requests are assumed to be within the policy's seven-day request window. Issued refunds and escalations are appended to `outputs/refund_ledger.jsonl` with timestamps and tool names.

| Scenario | Status | Refund / escalation amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | `REFUND` | 128.25 | 5 | 882 |
| S2 | `REFUND` | 135.00 | 5 | 882 |
| S3 | `ESCALATE` | 1,999.00 | 5 | 869 |
| S4 | `DENIED` | 0.00 | 4 | 593 |
| S5 | `DENIED` | 0.00 | 3 | 328 |
| S6 | `NOT_FOUND` | 0.00 | 2 | 66 |
| S7 | `FALLBACK_MAX_ITERATIONS` | 0.00 | 6 | 2,982 |

The ReAct loop enforces the six-call iteration limit, token limit and monotonic-clock time limit. The S1 demonstration with `token_limit=200` ends with `FALLBACK_TOKEN_LIMIT`; the S1 demonstration with `time_limit_seconds=0` ends with `FALLBACK_TIMEOUT`. S7 completes all six model calls under the default 6,000-token limit, then safely hands off with `FALLBACK_MAX_ITERATIONS`.

After every tool observation, the checkpoint in `state/` saves the run ID, scenario ID, full message transcript, tokens used, step number, tools executed and steps. The crash/resume demonstration intentionally raises a simulated crash after S1's second tool observation, then calls `resume_agent(run_id)`. The exact tools executed across the crash and resume are **get_order → search_policy → calculate_refund → issue_refund**; no tool is repeated, and the resumed final result is `REFUND: INR 128.25 issued.` The details are in `outputs/resume_demo.json` and `state/S1_resume_demo.json`.

### ReAct versus plan-and-execute

ReAct used 5 model calls for S1, 3 for S5, and 2 for S6; the seven normal scenarios used 2–6 calls. A plan-and-execute design would request the entire sequence once and could save tokens and calls when every assumption is correct, replanning only on error. But a missing order (`NOT_FOUND`) or cancelled order (`DENIED`) changes which actions are safe, so a static plan risks continuing with stale assumptions. For this refund workflow I prefer bounded ReAct: the additional calls are modest in this offline test, while each action depends on the latest database and policy observation. Iteration, token and time limits, durable checkpoints, and human handoff bound its failure modes.

## Limitations and Next Steps

All customers, orders, tickets, policies and outcomes are fictional synthetic data. The deterministic mock intentionally injects transient failures, malformed answers, invalid categories and a loop case; measured accuracy is not evidence of production LLM quality. The RFM snapshot and lapse window are short, and the lapse model has not been calibrated on real customer behaviour. The Premium flag uses whole-window RFM that includes September, so the Premium retention list is illustrative and not a leakage-free evaluation even though the model features use only pre-cutoff orders. TF-IDF is lexical rather than semantic, and the refund ledger is a local simulation, not a payment integration. Before any real deployment, add policy versioning, access controls, monitoring, calibrated evaluation, human review and audited financial-action safeguards.
