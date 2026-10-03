#!/usr/bin/env python
"""Retrieval eval harness.

Answers the question "did my change make retrieval better or worse?" with a
number instead of a vibe. Every case names the sources a correct retrieval must
surface; the harness reports top-1 accuracy and mean reciprocal rank (MRR).

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --golden evals/golden_set.json --threshold 0.8

Exits non-zero when the pass rate falls below ``--threshold``, so it can gate CI.

Note on scope: this measures *retrieval*, not generation. Generation quality
needs an LLM judge or human review -- see docs/ARCHITECTURE.md. Measuring the
part you can measure deterministically is still worth far more than eyeballing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running as a script from the repo root without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.llm.echo_provider import EchoChatModel
from app.llm.hashing_embedder import HashingEmbedder
from app.rag.pipeline import RAGPipeline
from app.rag.vectorstore import VectorStore

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class CaseResult:
    case_id: str
    question: str
    passed: bool
    rank: int | None
    top_score: float
    top_source: str
    expected: list[str]


async def build_pipeline(settings: Settings, corpus: Path) -> RAGPipeline:
    """Build a fully offline pipeline and index the corpus."""
    embedder = HashingEmbedder(dimension=settings.rag_embed_dim)
    pipeline = RAGPipeline(
        chat=EchoChatModel(),
        embedder=embedder,
        store=VectorStore(dimension=settings.rag_embed_dim),
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    for path in sorted(p for p in corpus.iterdir() if p.suffix in {".md", ".txt"}):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            await pipeline.ingest(text, source=path.name)
    return pipeline


async def evaluate(golden_path: Path, threshold: float, verbose: bool) -> int:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = golden["cases"]
    top_k = int(golden.get("top_k", 5))

    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    corpus = Path(golden.get("corpus", "data/sample_docs"))
    if not corpus.is_dir():
        print(f"{RED}corpus not found: {corpus}{RESET}", file=sys.stderr)
        return 2

    pipeline = await build_pipeline(settings, corpus)
    print(
        f"{DIM}corpus={corpus} documents={len(pipeline.store.documents())} "
        f"chunks={len(pipeline.store)} embedder={pipeline.embedder_name}{RESET}\n"
    )

    results: list[CaseResult] = []
    for case in cases:
        hits = await pipeline.search(case["question"], top_k=top_k)
        expected = set(case["expected_sources"])
        sources = [hit.chunk.source for hit in hits]

        rank = next((index + 1 for index, source in enumerate(sources) if source in expected), None)
        results.append(
            CaseResult(
                case_id=case["id"],
                question=case["question"],
                passed=rank is not None,
                rank=rank,
                top_score=round(hits[0].score, 4) if hits else 0.0,
                top_source=sources[0] if sources else "-",
                expected=case["expected_sources"],
            )
        )

    for result in results:
        marker = f"{GREEN}PASS{RESET}" if result.passed else f"{RED}FAIL{RESET}"
        rank_text = f"rank {result.rank}" if result.passed else "not retrieved"
        print(f"  {marker}  {result.question:<52} {DIM}{rank_text}{RESET}")
        if not result.passed:
            print(f"        expected {result.expected}, got top hit '{result.top_source}'")
        elif verbose:
            print(f"        top score {result.top_score} from '{result.top_source}'")

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    pass_rate = passed / total if total else 0.0
    reciprocal_ranks = [1.0 / result.rank for result in results if result.rank]
    mrr = sum(reciprocal_ranks) / total if total else 0.0

    print(
        f"\n{BOLD}{passed}/{total} passed ({pass_rate:.1%})  |  MRR {mrr:.3f}  |  "
        f"threshold {threshold:.0%}{RESET}"
    )

    if pass_rate < threshold:
        print(f"{RED}FAIL: pass rate below threshold{RESET}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval eval harness")
    parser.add_argument("--golden", default="evals/golden_set.json", help="Path to the golden set")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum pass rate to succeed")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print scores for passing cases"
    )
    args = parser.parse_args(argv)

    golden_path = Path(args.golden)
    if not golden_path.is_file():
        print(f"{RED}golden set not found: {golden_path}{RESET}", file=sys.stderr)
        return 2
    return asyncio.run(evaluate(golden_path, args.threshold, args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())
