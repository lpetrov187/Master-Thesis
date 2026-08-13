# Evaluation Report

24 tasks, agent pipeline vs. no-tool baseline.

## Overall

| Metric | Value |
|---|---|
| Tool-selection accuracy | 66.7% |
| Agent hallucination rate | 16.7% |
| Baseline hallucination rate | 16.7% |
| Agent task success rate | 83.3% |
| Baseline task success rate | 75.0% |
| Groundedness score (mean, n=16) | 0.800 (min 0.500, max 1.000) |

## Agent hallucination breakdown

`synthesis` = an ungrounded claim made at answer-drafting time. `evidence_corruption` = the claim was faithfully grounded in tool evidence, but that evidence was itself wrong due to an upstream bug (see PLAN.md) - a different failure mode than the Controlled-Access/Claim-Verifier mechanisms are designed to catch.

| Category | Count |
|---|---|
| evidence_corruption | 2 |
| none | 20 |
| synthesis | 2 |

## By task category

| Category | n | Tool acc. | Agent halluc. | Baseline halluc. | Agent success | Baseline success |
|---|---|---|---|---|---|---|
| code_analysis | 8 | 87.5% | 25.0% | 12.5% | 75.0% | 62.5% |
| doc_search | 8 | 37.5% | 12.5% | 37.5% | 87.5% | 62.5% |
| programming_problem | 8 | 75.0% | 12.5% | 0.0% | 87.5% | 100.0% |
