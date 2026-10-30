from __future__ import annotations

import typer

from ._common import run


def serve(host: str = "0.0.0.0", port: int = 8080, reload: bool = False) -> None:
    """Run the HTTP API (requires `ragstack[api]`)."""
    import uvicorn

    uvicorn.run("ragstack.interfaces.api.app:create_app", host=host, port=port, reload=reload, factory=True)


def migrate() -> None:
    """Apply pending migrations and exit."""

    async def go(c):  # type: ignore[no-untyped-def]
        typer.echo("✓ schema up to date")

    run(go)
