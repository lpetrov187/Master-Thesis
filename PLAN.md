# Master's Thesis: Agent System with Tool Selection & Hallucination Reduction

## Context

The thesis proposal calls for an LLM-based agent system that (1) analyzes a user request and selects the right tool for it, and (2) reduces hallucinations via **verification mechanisms, structured decision-making, and controlled access to information sources**. Evaluation must compare the agent system against a no-tools baseline on tool-selection accuracy, response quality, and hallucination rate, using software-engineering scenarios (code analysis, documentation search, programming problems).

Hard constraint: **2 weeks full-time** for implementation (possibly a bit more). The design below is chosen specifically to be optimal under that constraint — it de-risks the parts most likely to blow the schedule (structured output from a small local model) while still giving each of the three proposal-named mechanisms its own distinct, describable subsystem, because that 1:1 mapping is what makes the design chapter easy to write and defend.

**Why this is still "enough" for ~40 pages**: page count in a thesis mostly comes from *written analysis*, not feature count. Roughly: Intro (3-4p) + Theory/Related work on LLMs, agents, tool-use, hallucination mitigation (10-12p, independent of implementation size) + System Design (8-10p, three mechanisms + diagrams + verifier pseudocode) + Implementation (5-6p) + Evaluation incl. ablation (6-8p) + Conclusion (2-3p) ≈ 40 pages. A tightly-scoped but well-documented system supports this better than a sprawling one you don't have time to write up properly.

## Chosen Constraints

- **Model**: `Qwen2.5-7B-Instruct` via Ollama (primary). Keep `Qwen2.5-Coder-7B-Instruct` on hand as a same-family ablation/comparison model for the evaluation chapter — cheap to add, gives you a same-architecture-different-training comparison to write about.
- **Framework**: agent loop built from scratch in Python (no LangChain/LangGraph) — the orchestration logic is your contribution and gives you more to describe/defend.
- **Structured output**: use Ollama's JSON-schema-constrained decoding (`format: <schema>`) for every LLM call that must return structured data (tool selection, claim extraction). This removes most of the "small model produces broken JSON" risk that would otherwise eat days of debugging, and doubles as your "structured decision-making" mechanism.
- **Tools (3, deliberately not 4)**:
  1. **Documentation RAG** — `sentence-transformers` embeddings + `Chroma`, retrieval over a curated doc corpus.
  2. **Code analysis** — wraps `ast` + `ruff`, structured findings.
  3. **Code execution / test runner** — sandboxed `subprocess`, timeout, captures stdout/stderr/exceptions.
- **Hallucination reduction — three explicit layers**:
  1. **Structured decision-making**: schema-constrained tool selection (above).
  2. **Controlled access to information sources**: a policy layer — the Answer Synthesizer may only assert facts sourced from a tool's output or a retrieved doc chunk; anything else must be hedged ("I don't know") or omitted. This wraps the RAG tool and the tool-executor outputs.
  3. **Verification mechanism**: a claim-level verifier — extract atomic claims from the draft answer (constrained-JSON LLM call), match each against the available evidence (tool outputs + retrieved chunks), compute a groundedness score, and either hedge unsupported claims or trigger one retrieval/tool retry. This is your most substantial algorithm and the centerpiece of the design chapter.

## Architecture

```
User Query
   │
   ▼
[1] Request Analyzer   ──▶ classifies intent, decides whether/which tool is needed
   │
   ▼
[2] Tool Selector       ──▶ schema-constrained JSON output: {"tool": "...", "args": {...}}
   │                         (Structured Decision-Making layer)
   ▼
[3] Tool Executor       ──▶ runs the tool, returns structured evidence
   │
   ▼
[4] Answer Synthesizer  ──▶ drafts answer; Controlled-Access policy restricts claims
   │                         to sourced evidence only, else hedge
   ▼
[5] Claim Verifier      ──▶ extract claims → match vs. evidence → groundedness score
   │                         unsupported claim ⇒ hedge or one retry of [2]-[4]
   ▼
Final Response (+ full trace log: query, tool calls, evidence, draft, verifier verdicts)
```

Log every step — the trace log is both your debugging tool and your evaluation raw data.

## Day-by-Day Plan (10-12 working days + buffer)

| Day | Work |
|---|---|
| 1 | Env: install Ollama, pull `qwen2.5:7b-instruct` (+ `qwen2.5-coder:7b-instruct`), Python project setup (venv, `chromadb`, `sentence-transformers`, `pydantic`, `ruff`, `pytest`), git init, skeleton. |
| 2 | Tool registry (name/description/JSON-schema args) + schema-constrained Tool Selector, tested against all 3 tool schemas. |
| 3 | Tool 1: Documentation RAG — ingest pipeline (chunk/embed/store), retrieval function, manual sanity check on sample queries. |
| 4 | Tool 2 (code analysis) + Tool 3 (sandboxed code execution/test runner), unit tests for both. |
| 5 | Wire orchestration loop: analyzer → selector → executor → synthesizer, no verifier yet. End-to-end smoke test. |
| 6 | Controlled-Access policy layer on the Answer Synthesizer (claims must cite evidence or hedge). |
| 7 | Claim Verifier: claim extraction (constrained JSON) + evidence matching + groundedness score + retry/hedge policy. Biggest single item — protect this day. |
| 8 | No-tool baseline pipeline (same model, direct prompting). Harden error handling across the pipeline. **Checkpoint**: if behind, this is where you'd cut, not later. |
| 9 | Curate eval set: ~20-25 tasks across the 3 categories (code analysis, doc search, programming problems), with expected tool + reference answer/ground truth per task. |
| 10 | Run both conditions (baseline vs. agent) + Qwen2.5-Coder ablation if time allows; save full traces. |
| 11 | Hand-annotate tool-selection correctness, hallucination presence, task success; compute metrics; build comparison tables/charts. |
| 12 (+buffer) | Polish, write up architecture notes/README, export logs/metrics/screenshots for the thesis writeup. |

### Stretch (only if extra time materializes)
- Ablation: RAG-only vs. verifier-only vs. both, to isolate each mechanism's individual contribution — strong, easy-to-write experiments content.
- Swap in Qwen2.5-Coder as the primary model for a full second run, not just spot checks.
- Replace/augment manual hallucination annotation with an LLM-judge validated against a human-labeled subset.
- ~~**Known gap (Day 10 full run, 2026-08-12): Request Analyzer accuracy.**~~ **FIXED Day 12 (2026-08-13).** 8/24 eval tasks (33%) were misclassified as `needs_tool: False` when a tool was expected. Fix: `temperature=0.1` on the analyzer's call (the failures looked like sampling-driven inconsistency, not comprehension) plus 2 more targeted few-shot examples. Re-verified against the exact 8 failing queries before the full re-run: 7/8 fixed. The 8th (`doc_search_06`) was deliberately left as-is rather than chased with more eval-specific examples, to avoid overfitting the prompt to this one curated question. Full re-run: tool-selection accuracy 66.7% → 95.8%.
- ~~**Known bug (Day 10 full run, 2026-08-12): Tool Selector double-escapes newlines in extracted `code` args.**~~ **FIXED Day 12 (2026-08-13).** Confirmed on 2/24 tasks (`code_analysis_04`, `programming_problem_08`), both with the identical "unexpected character after line continuation character" failure signature — the model's JSON output re-encoded a real newline as the two-character sequence `\n` instead of an actual escaped newline. Fix: `tool_registry.py`'s `code` arg schema changed from a single string to a JSON array of lines, sidestepping the escaping problem structurally instead of prompting around it; `tool_executor.py` joins the array back into a real string before dispatch. Re-verified against both exact failing tasks before the full re-run: both now return clean, uncorrupted code. Full re-run: agent hallucination rate 16.7% → 0% (hand-checked against Day 9's verified ground truth across all 24 answers, not assumed).

## Verification / "done" criteria
- Orchestration loop runs end-to-end on a real query, producing a trace log through all 5 stages.
- Each of the 3 tools has at least one passing unit test.
- Baseline and agent system both run over the same eval set with saved traces.
- You can report, with numbers: tool-selection accuracy (%), groundedness score distribution, hallucination rate baseline vs. agent (%), task success rate baseline vs. agent (%).
