# The 3 Tools, in Plain English

The agent has exactly 3 tools it can reach for. Each one is a plain Python
function under the hood; `src/agent/tool_registry.py` just describes each
one's name, purpose, and expected inputs so the Tool Selector (the LLM) can
pick the right one and fill in its arguments correctly.

## 1. `doc_rag` — "look it up in the docs"

**What it does:** searches a stored pile of documentation and returns the
passages most relevant to a question — like Ctrl+F, but based on meaning
instead of exact word matches.

**Analogy:** you have a librarian who has already read every document in
the library and memorized what each paragraph is *about*. You ask a
question in your own words, and she hands you the 3 paragraphs most likely
to contain the answer — she doesn't answer the question herself, she just
finds the evidence.

**When the agent picks it:** any "how do I...", "what does X do", "what's
the right way to configure Y" question — the software-engineering
equivalent of looking something up instead of guessing.

**What goes in / out:**
- In: `{"query": "How do I configure connection pooling in SQLAlchemy?"}`
- Out: a handful of text passages from the doc corpus, each tagged with
  which file it came from.

## 2. `code_analysis` — "check this code without running it"

**What it does:** reads a piece of Python code and reports problems it can
find just by looking at it — syntax errors, style issues, structural
red flags — without ever executing a single line.

**Analogy:** a proofreader for code. They can tell you a sentence is
grammatically broken, or written in a confusing way, without needing to
"perform" the sentence to find that out.

**When the agent picks it:** "can you review this function", "is this code
well-written", "what's wrong with this snippet" — anything about the code's
*shape*, not its *behavior*.

**What goes in / out:**
- In: `{"code": "def add(a,b):\n    return a+b"}`
- Out: a structured list of findings (e.g. "missing space after comma",
  "unused import").

## 3. `code_execution` — "actually run this and see what happens"

**What it does:** runs a Python snippet in an isolated subprocess (with a
timeout, so nothing can hang forever) and captures whatever it printed,
any errors it raised, and its exit result.

**Analogy:** a supervised sandbox. You hand someone a piece of code, they
run it in a locked room where nothing they do can affect anything outside
that room, and they report back exactly what happened.

**When the agent picks it:** "run this", "what does this print", "does
this solve the problem" — anything where the *actual behavior* of the code
is the thing being asked about, not just its appearance.

**What goes in / out:**
- In: `{"code": "print(sum(range(5)))", "stdin": "optional input text"}`
- Out: captured stdout, stderr, and whether it raised an exception.

## Why only these 3

Each one covers a different *kind* of evidence the agent can ground an
answer in: **retrieved knowledge** (docs), **static inspection** (code
analysis), and **observed behavior** (code execution). That spread is what
lets the evaluation scenarios — code analysis, documentation search, and
programming problems — each map cleanly onto one primary tool.
