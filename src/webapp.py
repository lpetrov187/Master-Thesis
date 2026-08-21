"""Web chat UI for the agent pipeline - a thin Gradio front end over
Session.ask(), which already does everything (orchestration, evidence,
conversation context). No agent logic lives here, only presentation.

Usage:
    python -m src.webapp
"""
# Must be imported before anything that touches huggingface_hub (gradio's
# own HF Hub integration included) - src.config sets HF_HUB_OFFLINE for an
# already-cached model, which only has an effect if it runs before
# huggingface_hub.constants is first imported anywhere (see config.py).
import src.config  # noqa: F401, I001

import gradio as gr

from src.agent.session import Session
from src.tools.doc_rag import ingest


def respond(message: str, history: list[dict], session: Session) -> tuple[str, Session]:
    trace = session.ask(message)
    tool = trace["selection"]["tool"] if trace["selection"] else "none"
    answer = trace["answer"] if trace["answer"] is not None else f"ERROR: {trace['error']}"
    return f"**[tool: {tool}]**\n\n{answer}", session


def main() -> None:
    print("Loading documentation corpus...")
    ingest()

    # A bare `gr.State(Session)` crashes Gradio's config serialization:
    # dataclasses.is_dataclass() returns True for the *class* itself (not
    # just instances), so Gradio tries dataclasses.asdict(Session) before
    # the factory is ever called - wrapping it in a lambda avoids that.
    session_state = gr.State(lambda: Session())
    demo = gr.ChatInterface(
        respond,
        additional_inputs=[session_state],
        additional_outputs=[session_state],
        title="Master's Thesis Agent",
        description=(
            "Tool-using LLM agent with hallucination mitigation. "
            "Tools available: doc_rag, code_analysis, code_execution, web_fetch."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
