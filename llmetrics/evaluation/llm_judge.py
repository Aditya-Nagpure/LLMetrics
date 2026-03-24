"""LLMJudgeEvaluator — optional async LLM-based evaluation."""
from __future__ import annotations

import asyncio
import json
import re

from llmetrics.core.config import settings
from llmetrics.core.models import EvalResult, TraceRecord
from llmetrics.evaluation.base import Evaluator, merge_eval
from llmetrics.storage.base import StorageBackend

_PROMPT_TEMPLATE = """\
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the response below and return ONLY a JSON object — no other text.

Query: {query}

Context (retrieved documents):
{context}

Response: {response}

Score each criterion from 0.0 (worst) to 1.0 (best):
- faithfulness: Is the response fully supported by the context?
- context_relevance: Are the retrieved documents relevant to the query?
- answer_correctness: Does the response correctly answer the query?

Return format (JSON only):
{{"faithfulness": 0.0, "context_relevance": 0.0, "answer_correctness": 0.0}}"""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a possibly noisy LLM response."""
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in LLM judge response: {text!r}")
    return json.loads(match.group())


def _build_prompt(record: TraceRecord) -> str:
    context = "\n".join(
        f"{i + 1}. {d.get('content', '')}"
        for i, d in enumerate(record.retrieved_docs)
        if d.get("content")
    )
    return _PROMPT_TEMPLATE.format(
        query=record.query,
        context=context or "(no context)",
        response=record.response or "(no response)",
    )


def _call_llm(prompt: str) -> str:
    """Synchronous LLM call — runs in a thread executor from the async wrapper."""
    if settings.llm_provider == "groq":
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content

    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.llm_judge_model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


class LLMJudgeEvaluator(Evaluator):
    """
    LLM-based evaluator. Off by default (enable_llm_judge=False in settings).
    Runs synchronously via evaluate(); use evaluate_async() + asyncio.create_task()
    for the fire-and-forget pattern in middleware.
    """

    def evaluate(self, record: TraceRecord) -> EvalResult:
        prompt = _build_prompt(record)
        try:
            raw = _call_llm(prompt)
            data = _extract_json(raw)
            return EvalResult(
                faithfulness=float(data.get("faithfulness") or 0.0),
                context_relevance=float(data.get("context_relevance") or 0.0),
                answer_correctness=float(data.get("answer_correctness") or 0.0),
            )
        except Exception:
            return EvalResult()

    async def evaluate_async(
        self, record: TraceRecord, storage: StorageBackend
    ) -> None:
        """Fire-and-forget: evaluate in a thread, then update the stored record."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.evaluate, record)
            merged = merge_eval(record.evaluation, result)
            updated = record.model_copy(update={"evaluation": merged})
            storage.write(updated)
        except Exception:
            pass  # Never let the judge crash or block anything
