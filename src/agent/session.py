"""Conversation session: threads a sliding window of recent turns through
the stateless orchestrator, so follow-up queries can reference what was
discussed before. orchestrator.run() itself stays a pure function (history
is just an optional parameter) - this class is the only thing that
actually holds state, used by the CLI for a multi-turn conversation.
"""
from dataclasses import dataclass, field

import ollama

from src.agent.orchestrator import run as run_agent
from src.config import OLLAMA_HOST

_DEFAULT_MAX_HISTORY = 5


@dataclass
class Session:
    client: ollama.Client | None = None
    max_history: int = _DEFAULT_MAX_HISTORY
    history: list[dict] = field(default_factory=list)

    def ask(self, query: str) -> dict:
        """Run `query` through the agent pipeline with this session's
        accumulated history, then record the turn and trim to the last
        `max_history` entries."""
        client = self.client or ollama.Client(host=OLLAMA_HOST)
        # A snapshot, not the live list - self.history gets mutated right
        # below, and the callee shouldn't see this turn appear in its own
        # "prior" history if anything ever inspects it after the fact.
        trace = run_agent(query, client=client, history=list(self.history))

        self.history.append({"query": query, "answer": trace["answer"]})
        self.history = self.history[-self.max_history :]

        return trace
