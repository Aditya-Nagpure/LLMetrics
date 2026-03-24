"""LLMetrics CLI — trace inspection, metrics summary, and on-demand evaluation."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Any

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmetrics.core.config import settings
from llmetrics.storage.base import StorageBackend

app = typer.Typer(
    name="llmetrics",
    help="LLMetrics observability CLI",
    no_args_is_help=True,
)
traces_app = typer.Typer(help="Trace inspection commands", no_args_is_help=True)
metrics_app = typer.Typer(help="Metrics commands", no_args_is_help=True)
app.add_typer(traces_app, name="traces")
app.add_typer(metrics_app, name="metrics")

console = Console()


# ---------------------------------------------------------------------------
# Storage helper
# ---------------------------------------------------------------------------


def _open_storage() -> StorageBackend:
    if settings.storage_backend == "jsonl":
        from llmetrics.storage.jsonl_store import JsonlStore

        return JsonlStore(settings.jsonl_path)
    from llmetrics.storage.sqlite_store import SqliteStore

    return SqliteStore(settings.sqlite_path)


# ---------------------------------------------------------------------------
# traces list
# ---------------------------------------------------------------------------


@traces_app.command("list")
def traces_list(
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows to show"),
) -> None:
    """List stored traces in a summary table."""
    with _open_storage() as storage:
        records = storage.read_all()

    if not records:
        console.print("[yellow]No traces found.[/yellow]")
        raise typer.Exit()

    records = records[-limit:]  # most recent N

    table = Table(title=f"Traces (showing {len(records)})", header_style="bold cyan")
    table.add_column("Trace ID", style="dim", width=8)
    table.add_column("Timestamp", width=19)
    table.add_column("Query", width=42)
    table.add_column("Latency", justify="right", width=8)
    table.add_column("Cost", justify="right", width=10)
    table.add_column("Faith.", justify="right", width=6)
    table.add_column("Rel.", justify="right", width=6)
    table.add_column("Err", width=4)

    for r in records:
        faith = (
            f"{r.evaluation.faithfulness:.2f}"
            if r.evaluation.faithfulness is not None
            else "—"
        )
        rel = (
            f"{r.evaluation.context_relevance:.2f}"
            if r.evaluation.context_relevance is not None
            else "—"
        )
        query_display = r.query[:40] + "…" if len(r.query) > 40 else r.query
        table.add_row(
            r.trace_id[:8],
            r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            query_display,
            f"{r.latency_total_s:.3f}s",
            f"${r.cost_usd:.6f}",
            faith,
            rel,
            "✗" if r.error else "",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# traces show <id>
# ---------------------------------------------------------------------------


@traces_app.command("show")
def traces_show(
    trace_id: str = typer.Argument(..., help="Full or prefix of trace ID"),
) -> None:
    """Show full details of a single trace."""
    with _open_storage() as storage:
        record = storage.read_by_id(trace_id)
        if record is None:
            # Try prefix match
            all_records = storage.read_all()
            matches = [r for r in all_records if r.trace_id.startswith(trace_id)]
            if len(matches) == 1:
                record = matches[0]
            elif len(matches) > 1:
                console.print(
                    f"[yellow]Ambiguous prefix — {len(matches)} matches. Use more characters.[/yellow]"
                )
                raise typer.Exit(code=1)

    if record is None:
        console.print(f"[red]Trace {trace_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold]{record.trace_id}[/bold]  {record.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            title="Trace",
            style="cyan",
        )
    )

    # Query / Response
    console.print(Panel(record.query, title="Query", style="green"))
    console.print(Panel(record.response or "[dim](empty)[/dim]", title="Response", style="white"))

    # Metrics
    metrics_text = (
        f"Latency total:  {record.latency_total_s:.3f}s\n"
        f"Latency LLM:    {record.latency_llm_s:.3f}s\n"
        f"Tokens:         prompt={record.tokens.prompt if record.tokens else '?'}  "
        f"completion={record.tokens.completion if record.tokens else '?'}\n"
        f"Cost:           ${record.cost_usd:.8f}"
    )
    console.print(Panel(metrics_text, title="Metrics", style="yellow"))

    # Evaluation
    ev = record.evaluation
    eval_text = (
        f"Faithfulness:      {ev.faithfulness:.4f}" if ev.faithfulness is not None else "Faithfulness:      —"
    ) + "\n" + (
        f"Context relevance: {ev.context_relevance:.4f}" if ev.context_relevance is not None else "Context relevance: —"
    ) + "\n" + (
        f"Answer correctness:{ev.answer_correctness:.4f}" if ev.answer_correctness is not None else "Answer correctness:—"
    )
    console.print(Panel(eval_text, title="Evaluation", style="magenta"))

    # Spans
    if record.spans:
        span_table = Table(header_style="bold", show_header=True)
        span_table.add_column("Span")
        span_table.add_column("Duration", justify="right")
        for s in record.spans:
            span_table.add_row(s.name, f"{s.duration_s*1000:.1f}ms")
        console.print(Panel(span_table, title="Spans", style="blue"))

    # Docs
    if record.retrieved_docs:
        doc_lines = "\n".join(
            f"[dim]{i+1}.[/dim] {d.get('content', str(d))[:120]}"
            for i, d in enumerate(record.retrieved_docs)
        )
        console.print(Panel(doc_lines, title=f"Retrieved Docs ({len(record.retrieved_docs)})", style="dim"))

    if record.error:
        console.print(Panel(record.error, title="Error", style="red"))


# ---------------------------------------------------------------------------
# metrics summary
# ---------------------------------------------------------------------------


@metrics_app.command("summary")
def metrics_summary() -> None:
    """Show latency percentiles and token/cost aggregates from stored traces."""
    with _open_storage() as storage:
        records = storage.read_all()

    if not records:
        console.print("[yellow]No traces found.[/yellow]")
        raise typer.Exit()

    latencies = [r.latency_total_s for r in records]
    costs = [r.cost_usd for r in records]
    token_counts = [r.tokens.total for r in records if r.tokens]
    errors = sum(1 for r in records if r.error)

    def _pct(data: list[float], q: float) -> float:
        if len(data) == 1:
            return data[0]
        idx = max(0, int(q * len(data)) - 1)
        return sorted(data)[idx]

    table = Table(title="LLMetrics Summary", header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total traces", str(len(records)))
    table.add_row("Error traces", str(errors))
    table.add_row("", "")
    table.add_row("P50 latency", f"{_pct(latencies, 0.50):.3f}s")
    table.add_row("P95 latency", f"{_pct(latencies, 0.95):.3f}s")
    table.add_row("Max latency", f"{max(latencies):.3f}s")
    table.add_row("", "")
    table.add_row("Total cost", f"${sum(costs):.6f}")
    table.add_row("Avg cost/req", f"${sum(costs)/len(costs):.6f}")
    if token_counts:
        table.add_row("Avg tokens/req", f"{sum(token_counts)//len(token_counts)}")

    # Eval averages
    faith_scores = [r.evaluation.faithfulness for r in records if r.evaluation.faithfulness is not None]
    rel_scores = [r.evaluation.context_relevance for r in records if r.evaluation.context_relevance is not None]
    if faith_scores:
        table.add_row("", "")
        table.add_row("Avg faithfulness", f"{sum(faith_scores)/len(faith_scores):.4f}")
    if rel_scores:
        table.add_row("Avg context rel.", f"{sum(rel_scores)/len(rel_scores):.4f}")

    console.print(table)


# ---------------------------------------------------------------------------
# evaluate --trace-id <id>
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    trace_id: str = typer.Option(..., "--trace-id", help="Trace ID to evaluate"),
    save: bool = typer.Option(False, "--save", help="Write updated scores back to storage"),
) -> None:
    """Run evaluators on a stored trace and display the results."""
    import sys

    with _open_storage() as storage:
        record = storage.read_by_id(trace_id)

    if record is None:
        console.print(f"[red]Trace {trace_id!r} not found.[/red]")
        raise typer.Exit(code=1)

    from llmetrics.evaluation.base import merge_eval
    from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator
    from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator

    evaluators = [FaithfulnessEvaluator(), ContextRelevanceEvaluator()]
    results = []
    for ev in evaluators:
        try:
            console.print(f"Running [bold]{ev.__class__.__name__}[/bold]…")
            results.append(ev.evaluate(record))
        except Exception as exc:
            console.print(f"[red]{ev.__class__.__name__} failed: {exc}[/red]")

    if not results:
        console.print("[yellow]No evaluation results produced.[/yellow]")
        raise typer.Exit(code=1)

    merged = merge_eval(*results)

    table = Table(title=f"Evaluation — {trace_id[:8]}…", header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    table.add_row("Faithfulness", f"{merged.faithfulness:.4f}" if merged.faithfulness is not None else "—")
    table.add_row("Context relevance", f"{merged.context_relevance:.4f}" if merged.context_relevance is not None else "—")
    table.add_row("Answer correctness", f"{merged.answer_correctness:.4f}" if merged.answer_correctness is not None else "—")
    console.print(table)

    if save:
        updated = record.model_copy(update={"evaluation": merged})
        with _open_storage() as storage:
            storage.write(updated)
        console.print("[green]Scores saved to storage.[/green]")
