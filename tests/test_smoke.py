"""Day-1 smoke test: confirms Ollama is reachable and returns schema-constrained JSON."""
import json

import ollama
from src.config import OLLAMA_HOST, PRIMARY_MODEL


def test_ollama_reachable():
    client = ollama.Client(host=OLLAMA_HOST)
    models = {m["model"] for m in client.list()["models"]}
    assert PRIMARY_MODEL in models


def test_structured_json_output():
    client = ollama.Client(host=OLLAMA_HOST)
    schema = {
        "type": "object",
        "properties": {"tool": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["tool", "reason"],
    }
    response = client.chat(
        model=PRIMARY_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Reply with a JSON object naming a tool called 'echo' and a reason.",
            }
        ],
        format=schema,
    )
    parsed = json.loads(response["message"]["content"])
    assert "tool" in parsed and "reason" in parsed
