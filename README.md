# Agent System with Tool Selection and Hallucination Reduction

Master's thesis project: an LLM agent that analyzes a user request, selects an
appropriate tool from a fixed registry, and reduces hallucinations through
structured decision-making, controlled access to information sources, and a
claim-level verification pass.

See `PLAN.md` for the implementation plan.

## Stack

- Local LLM served via [Ollama](https://ollama.com) — `qwen2.5:7b-instruct`
  (primary), `qwen2.5-coder:7b-instruct` (ablation/comparison)
- Python 3.14, `chromadb` + `sentence-transformers` for retrieval
- No agent framework — orchestration loop is implemented from scratch in
  `src/agent/`

## Setup

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5-coder:7b-instruct
```

## Try it

```bash
python -m src.cli
```

Interactive REPL - type a question, see the agent's answer (which tool it
picked, its groundedness score, whether it hedged or retried). Prefix a
line with `baseline: ` to run that one query through the no-tool baseline
instead, for a side-by-side comparison. `exit` or Ctrl+C to quit.

## Project layout

```
src/
  agent/     # request analyzer, tool selector, orchestrator, synthesizer, verifier
  tools/     # doc RAG, code analysis, code execution tools
  eval/      # eval task loading, runner, metrics, report generation
  cli.py     # interactive demo entrypoint
data/
  docs/      # source documents for the RAG corpus
  eval/      # evaluation task set + reference answers
logs/        # trace logs from agent runs (jsonl)
reports/     # generated eval tables/charts
tests/       # unit tests for tools and agent components
```
