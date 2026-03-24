#!/usr/bin/env python
"""
replay_traces.py — Re-score all stored traces with the current evaluators.

Usage:
    python scripts/replay_traces.py [--dry-run]

For SQLite, records are updated in-place (INSERT OR REPLACE).
For JSONL, a new file is written atomically (old file replaced).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmetrics.core.config import settings
from llmetrics.core.models import TraceRecord
from llmetrics.evaluation.base import merge_eval
from llmetrics.evaluation.context_relevance import ContextRelevanceEvaluator
from llmetrics.evaluation.faithfulness import FaithfulnessEvaluator
from llmetrics.storage.base import StorageBackend


def _open_storage() -> StorageBackend:
    if settings.storage_backend == "jsonl":
        from llmetrics.storage.jsonl_store import JsonlStore

        return JsonlStore(settings.jsonl_path)
    from llmetrics.storage.sqlite_store import SqliteStore

    return SqliteStore(settings.sqlite_path)


def _rescore(record: TraceRecord, evaluators: list) -> TraceRecord:
    results = []
    for ev in evaluators:
        try:
            results.append(ev.evaluate(record))
        except Exception as exc:
            print(f"  [WARN] {ev.__class__.__name__} failed for {record.trace_id[:8]}: {exc}")
    if results:
        merged = merge_eval(*results)
        return record.model_copy(update={"evaluation": merged})
    return record


def replay(dry_run: bool = False) -> None:
    evaluators = [FaithfulnessEvaluator(), ContextRelevanceEvaluator()]
    print(f"Evaluators: {[e.__class__.__name__ for e in evaluators]}")
    print(f"Storage:    {settings.storage_backend}")
    if dry_run:
        print("DRY RUN — no writes will be performed\n")

    with _open_storage() as storage:
        records = storage.read_all()
        print(f"Loaded {len(records)} trace(s)\n")

        updated: list[TraceRecord] = []
        for i, record in enumerate(records, 1):
            print(f"[{i}/{len(records)}] {record.trace_id[:8]}… q={record.query[:50]!r}")
            rescored = _rescore(record, evaluators)
            ev = rescored.evaluation
            print(
                f"         faithfulness={ev.faithfulness}  "
                f"context_relevance={ev.context_relevance}"
            )
            updated.append(rescored)

        if not dry_run:
            if settings.storage_backend == "jsonl":
                # Rewrite the JSONL file atomically
                path = Path(settings.jsonl_path)
                tmp = path.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for r in updated:
                        f.write(r.model_dump_json() + "\n")
                tmp.replace(path)
                print(f"\nRewrote {path} with {len(updated)} record(s).")
            else:
                for r in updated:
                    storage.write(r)
                print(f"\nUpdated {len(updated)} record(s) in SQLite.")
        else:
            print("\nDry run complete — nothing written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-score stored traces with current evaluators.")
    parser.add_argument("--dry-run", action="store_true", help="Print scores without writing")
    args = parser.parse_args()
    replay(dry_run=args.dry_run)