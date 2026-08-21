# Project Overview: LLM Agent with Tool Selection and Hallucination Mitigation

This document is a structured summary of a Master's thesis implementation
project, written for another AI agent to use as context when helping write
the thesis text. It describes what was actually built and verified, not
the original plan where the two diverge. Where a design decision has a
"why," that reasoning is included, since it's usually the part worth
discussing in a design/methodology chapter.

**Scope note**: one previously-explored pipeline stage is intentionally
out of scope for this project and should not be discussed, described, or
referenced anywhere in the thesis. Every component described below is
active and in scope.

---

## 1. What this project is

A Master's thesis implementing an LLM-based agent that:
1. Analyzes a user request and decides whether it needs an external tool.
2. Selects and invokes the right tool from a fixed registry, using
   schema-constrained LLM output rather than free-text parsing.
3. Synthesizes an answer restricted to what the tool's output actually
   supports.
4. For code-writing requests specifically, runs a bounded
   generate-then-verify-then-revise loop, where "verify" means actually
   compiling/running the code, not asking the model to judge itself.

It is evaluated against a **no-tool baseline** (the same underlying model,
prompted directly, with no tools and no pipeline) on tool-selection
accuracy, hallucination rate, and task success rate.

**Model**: `qwen2.5:7b-instruct`, served locally via Ollama. A
same-family coder variant, `qwen2.5-coder:7b-instruct`, is available as an
ablation/comparison model but is not the primary model.

**No agent framework** (no LangChain/LangGraph/etc.) - the orchestration
loop, tool registry, and dispatch logic are implemented from scratch in
Python. This is a deliberate scope decision: the orchestration logic
itself is the thesis's contribution, not a wrapper around an existing
framework's abstractions.

---

## 2. Architecture

```
User Query
    |
    v
Request Analyzer -- classifies the query into one of three actions:
    |                 "lookup_or_inspect" | "generate" | "none"
    |
    +-- "lookup_or_inspect" --> Tool Selector --> Tool Executor --+
    |     (schema-constrained tool +   (runs the tool,            |
    |      args choice)                 returns structured        |
    |                                    evidence)                |
    |                                                              v
    +-- "generate" --------> Code Generation Loop -----------> evidence
    |     (see section 4)
    |
    +-- "none" ---------------------------------------------------+
                                                                    |
                                                                    v
                                                        Answer Synthesizer
                                                (drafts the answer; if there
                                                 is evidence, restricted to
                                                 facts that evidence
                                                 actually supports)
                                                                    |
                                                                    v
                                                             Final Answer
```

Every stage's inputs/outputs are logged to a structured trace (per-run
JSON: query, classification, tool selection, evidence, answer, per-stage
timing), which is both the debugging tool and the evaluation raw data.

### Pipeline stages (source files)

- **Request Analyzer** (`src/agent/request_analyzer.py`) - one
  schema-constrained LLM call classifying the query into `none` /
  `lookup_or_inspect` / `generate`. Few-shot examples in the prompt
  distinguish the three cases (e.g. a pasted code snippet or a
  named-library question is `lookup_or_inspect`; a described-but-not-yet-
  written task is `generate`).
- **Tool Selector** (`src/agent/tool_selector.py`) - one schema-constrained
  LLM call choosing a tool name + arguments from the tool registry. The
  registry's JSON schemas are embedded directly into the selection
  prompt, so adding a new tool or a new argument to an existing tool
  requires no prompt-engineering - the model sees the current schema live.
- **Tool Executor** (`src/agent/tool_executor.py`) - dispatches to the
  chosen tool's Python function and wraps the result (or any exception)
  into a structured evidence dict: `{tool, args, result, error}`. A
  failing tool never crashes the pipeline; it becomes evidence with
  `error` set instead.
- **Answer Synthesizer** (`src/agent/answer_synthesizer.py`) - one LLM
  call that drafts the final answer. See section 3 for its grounding
  policy.

### Conversation context

`src/agent/session.py` wraps the (otherwise stateless) pipeline in a
session that keeps a sliding window of the last few `{query, answer}`
turns, threaded into the Request Analyzer, Tool Selector, and Answer
Synthesizer prompts. This lets follow-up questions ("what about X
instead?") resolve correctly without repeating context. It is
session-only, in-memory, not persisted to disk.

---

## 3. Hallucination mitigation strategies (active)

Two of the three originally-proposed mitigation layers are active in the
current system:

1. **Structured decision-making.** Every LLM call that must produce
   structured data (request classification, tool selection) uses Ollama's
   JSON-schema-constrained decoding (the `format=<schema>` parameter),
   not free-text parsing. This removes an entire class of "model produced
   malformed output" failures and is the most defensible "structured
   decision-making" evidence for the thesis, since it's used
   unconditionally, everywhere structured output is needed.

2. **Controlled access to information sources.** The Answer Synthesizer
   is instructed to answer only from the evidence a tool actually
   returned - never from the model's own training knowledge - and to
   explicitly say which part of a question is unsupported rather than
   guess. This applies uniformly to every tool's evidence, not just
   document retrieval (see `_CONTROLLED_ACCESS_POLICY` in
   `answer_synthesizer.py`).

3. **Execution-grounded verification (code generation only).** The code
   generation loop (section 4) doesn't ask the model to judge its own
   output - it actually runs static analysis and executes/compiles the
   code, and only accepts the result once real tool output confirms it
   works. It's grounded in real program execution rather than a model's
   self-assessment, scoped specifically to the one place (code
   generation) where "does this actually work" has an objective,
   checkable answer.

---

## 4. Code generation loop ("generate" action)

`src/agent/code_generation.py`. For requests that describe a coding task
to implement (not a documentation lookup, not existing code to inspect),
the orchestrator runs a bounded loop instead of the normal Tool
Selector -> Tool Executor path:

1. **Generate**: one LLM call drafts code in a fenced block, tagged with
   its language (`python` or `c`).
2. **Verify**: the generated code is run through the *same* `code_analysis`
   and `code_execution` tools that exist standalone in the registry (see
   section 5) - static analysis first, then actual execution.
3. **Revise (bounded to exactly one attempt)**: if verification found a
   real problem (syntax error, lint finding, non-zero exit code, timeout,
   or a tool itself failing), the specific problem is fed back to the
   model for one revision pass, then re-verified. There is no open-ended
   retry loop - after one revision, whatever the result is gets reported,
   including if it's still broken.

This is a deliberate design choice: an open-ended ReAct-style loop was
considered and rejected in favor of this bounded shape, because of
observed reliability limits of the small (7B) local model on repeated
self-correction - a model that fails a check twice tends to keep failing
the same way, so more attempts mostly cost latency without improving
correctness.

The final evidence reports whether a revision fired, why, and how many
iterations it took (`iterations`, `revised`, `revision_reason` in the
result), so the pipeline's behavior on a given task is inspectable, not a
black box.

**Language support**: Python (`ast` + `ruff` for analysis, subprocess
execution) and C (`gcc -fsyntax-only -Wall -Wextra` for analysis, compile-
then-run for execution) are both supported, verified against the real
model. C support demonstrates the verification step is language-
pluggable, not hardcoded to Python - a compile failure surfaces through
the exact same evidence shape (`exit_code`, `stderr`) a Python runtime
failure would, so no pipeline logic had to special-case the language.

---

## 5. Tools (registry)

Every tool is a plain, fast, local, deterministic Python function - none
of them call the LLM internally. This was a deliberate architectural
invariant, verified with real timing measurements (tool execution
consistently sub-second to a few seconds; the LLM calls, not tool
execution, dominate end-to-end latency). It's also why the code generation
loop (section 4) is a separate orchestrator branch rather than a 4th
registry tool - it genuinely does call the LLM internally (to generate/
revise), and folding that into the "tool" abstraction would have hidden a
nested agent inside what's supposed to be a deterministic action.

- **`doc_rag`** (`src/tools/doc_rag.py`) - embedding-based retrieval over
  a local documentation corpus (FastAPI's official docs, 85 files,
  markdown-aware structural chunking on headings/code fences).
  `sentence-transformers/all-MiniLM-L6-v2` embeddings, ChromaDB (cosine
  distance) for storage/search, a calibrated distance threshold to filter
  out irrelevant chunks, and same-section chunk expansion for hits that
  ranked. A known, empirically-diagnosed limitation: retrieval quality is
  sensitive to vocabulary overlap between the query and the document's
  own terminology, not purely semantic relevance - a documented
  limitation of small bi-encoder embedding models, not something chunking
  or threshold tuning can fully fix.
- **`code_analysis`** (`src/tools/code_analysis.py`) - static analysis.
  Python via `ast` (syntax) + `ruff` (lint); C via `gcc -fsyntax-only
  -Wall -Wextra`, with gcc's plain-text diagnostics parsed into the same
  structured finding shape ruff produces.
- **`code_execution`** (`src/tools/code_execution.py`) - sandboxed
  execution with a timeout, capturing stdout/stderr/exit code. Python via
  a subprocess running the interpreter directly; C via compile-to-temp-
  binary-then-run, with compile failures surfaced through the same
  stdout/stderr/exit_code shape a runtime failure would use.
- **`web_fetch`** (`src/tools/web_fetch.py`) - fetches a specific,
  user-named URL and extracts its readable text (via `requests` +
  BeautifulSoup, stripping script/style, truncating long pages). The key
  design property: the *user* names the exact URL, so the agent never
  picks an unvetted source itself - this is why an open web-search tool
  was considered and deliberately not built (the agent choosing which
  search result to trust would directly undermine the controlled-access
  policy in section 3).

---

## 6. Baseline (comparison condition)

`src/agent/baseline.py` - the same underlying model, prompted directly
with the user's query and no tools, no pipeline, no evidence grounding.
This is the control condition the tool-using agent is evaluated against.

---

## 7. Interfaces

- **CLI** (`src/cli.py`, run via `python -m src.cli`) - interactive REPL.
  Each response shows which tool was used and the answer. A `baseline: `
  prefix routes that one query through the no-tool baseline instead, for
  a quick side-by-side comparison; baseline queries are deliberately
  excluded from conversation history, to keep it a clean control
  condition.
- **Web UI** (`src/webapp.py`, run via `python -m src.webapp`) - a Gradio
  chat interface over the same `Session` class the CLI uses, so it's a
  thin presentation layer with no duplicated agent logic. One session per
  browser tab.

---

## 8. Technology stack

| Purpose | Technology |
|---|---|
| LLM serving | [Ollama](https://ollama.com), local |
| Primary model | `qwen2.5:7b-instruct` |
| Comparison model | `qwen2.5-coder:7b-instruct` |
| Structured LLM output | Ollama JSON-schema-constrained decoding (`format=`), validated with `jsonschema` |
| Language | Python 3.14 |
| Document retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) + `ChromaDB` |
| Python static analysis | `ast` (stdlib) + `ruff` |
| C static analysis / compilation | `gcc` (MinGW-w64 / MSYS2 on Windows) |
| Code execution sandboxing | `subprocess` (stdlib), with timeouts |
| Web fetching | `requests` + `BeautifulSoup` (`beautifulsoup4`) |
| Web UI | `Gradio` (`ChatInterface`) |
| Testing | `pytest` |
| Linting | `ruff` |
| No agent framework | orchestration hand-written in `src/agent/` |

---

## 9. Project layout

```
src/
  agent/
    request_analyzer.py    # stage 1: classify the query
    tool_selector.py        # stage 2: schema-constrained tool + args choice
    tool_executor.py        # stage 3: dispatches to a tool, wraps errors
    answer_synthesizer.py   # stage 4: grounded answer drafting
    code_generation.py      # "generate" action: bounded generate-verify-revise loop
    orchestrator.py         # wires the stages together per request
    tool_registry.py        # tool names, descriptions, JSON-schema args
    session.py               # per-conversation sliding-window context
    history.py               # formats prior turns into prompt text
    baseline.py               # no-tool comparison condition
    trace_log.py              # structured per-run logging
  tools/
    doc_rag.py                 # documentation retrieval tool
    code_analysis.py           # static analysis tool (Python + C)
    code_execution.py          # sandboxed execution tool (Python + C)
    web_fetch.py                # user-named URL fetch tool
    chunking.py                  # markdown-aware document chunking
  eval/
    tasks.py, runner.py, metrics.py, report.py, annotations.py
  cli.py                          # interactive CLI entrypoint
  webapp.py                       # Gradio web UI entrypoint
  config.py                       # model names, paths, feature flags
data/
  docs/        # FastAPI documentation corpus (RAG source)
  eval/        # eval task set, saved results, hand annotations
tests/         # unit tests per module above
PLAN.md        # original implementation plan + day-by-day change log,
               # including the disclosed eval-set-leakage limitation
```
