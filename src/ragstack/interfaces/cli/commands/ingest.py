from __future__ import annotations

from pathlib import Path

import typer

from ._common import run


def ingest(
    files: list[Path] = typer.Argument(..., exists=True),
    tenant: str = typer.Option("default"),
    user: list[str] = typer.Option([], help="grant to these user ids"),
    group: list[str] = typer.Option([], help="grant to these group ids"),
    everyone: bool = typer.Option(False, help="grant to the whole tenant"),
    config: str | None = typer.Option(None, help="RetrievalConfig YAML (chunking section is used)"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Ingest FILES with the given access grants (deny by default: pass at least one)."""

    async def go(c):  # type: ignore[no-untyped-def]
        for path in files:
            doc, n = await c.ingest_document(path, tenant, users=user, groups=group, everyone=everyone)
            typer.echo(f"✓ {path.name}: document {doc.id} ({n} chunks)")

    run(go, config, verbose)
