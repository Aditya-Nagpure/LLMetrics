"""Prompt builder — assembles the final prompt from a Jinja2 template."""
from __future__ import annotations

from jinja2 import BaseLoader, Environment

from llmetrics.tracing.span import span

_DEFAULT_TEMPLATE = """\
You are a helpful assistant. Answer the question based only on the provided context.

Context:
{% for doc in docs %}
{{ loop.index }}. {{ doc }}
{% endfor %}
Question: {{ query }}

Answer:"""


class PromptBuilder:
    """Renders a Jinja2 prompt template from query + document list."""

    def __init__(self, template: str | None = None) -> None:
        src = template or _DEFAULT_TEMPLATE
        env = Environment(loader=BaseLoader(), autoescape=False)
        self._template = env.from_string(src)

    def build(self, query: str, docs: list[str]) -> str:
        """Render and return the assembled prompt string."""
        with span("prompt_builder", {"num_docs": len(docs)}):
            return self._template.render(query=query, docs=docs)