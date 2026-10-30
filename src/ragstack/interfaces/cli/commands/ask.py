from __future__ import annotations

import json

import typer

from ._common import run


def ask(
    question: str,
    tenant: str = typer.Option("default"),
    user: str = typer.Option(..., help="who is asking; groups are resolved from the database"),
    config: str | None = typer.Option(None),
    show_trace: bool = typer.Option(False, "--trace"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Ask a question and print the grounded answer with citations."""

    async def go(c):  # type: ignore[no-untyped-def]
        r = await c.ask(tenant, user, question)
        typer.echo(r.answer.text)
        if r.answer.refused:
            typer.secho(f"(refused: {r.answer.refusal_reason})", fg="yellow")
        for cit in r.answer.citations:
            where = f"{cit.source_name}" + (f" p.{cit.page}" if cit.page else "")
            typer.secho(f"  [{cit.index}] {where}: {cit.quote[:100]!r}", fg="cyan")
        if show_trace:
            typer.echo(json.dumps(r.trace.model_dump(), indent=2, default=str))
        else:
            typer.secho(
                f"  evidence={r.answer.evidence_score:.2f} attempts={r.trace.attempts} "
                f"{r.trace.total_ms:.0f} ms tokens={r.trace.input_tokens}+{r.trace.output_tokens}",
                fg="bright_black",
            )

    run(go, config, verbose)


def search(
    query: str,
    tenant: str = typer.Option("default"),
    user: str = typer.Option(...),
    config: str | None = typer.Option(None),
) -> None:
    """Retrieval only: print the ranked chunks with their scores."""

    async def go(c):  # type: ignore[no-untyped-def]
        hits, _ = await c.search(tenant, user, query)
        for i, rc in enumerate(hits, 1):
            s = rc.scores
            typer.echo(
                f"{i:2d}. {rc.source_name} p.{rc.chunk.page} rrf={s.rrf or 0:.4f} "
                f"rerank={s.rerank if s.rerank is not None else '-'}  {rc.chunk.text[:90]!r}"
            )

    run(go, config)
