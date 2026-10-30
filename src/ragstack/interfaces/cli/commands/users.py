from __future__ import annotations

import typer

from ._common import run


def user(
    user_id: str,
    tenant: str = typer.Option("default"),
    group: list[str] = typer.Option([], help="group memberships to add"),
) -> None:
    """Create a user and its group memberships (the server resolves groups from here, never from requests)."""

    async def go(c):  # type: ignore[no-untyped-def]
        await c.access.ensure_user(tenant, user_id, group)
        typer.echo(f"✓ {user_id} in {tenant}: groups={group}")

    run(go)
