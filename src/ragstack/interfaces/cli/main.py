from __future__ import annotations

import typer

from .commands import ask, evaluate, ingest, serve, users

app = typer.Typer(help="ragstack: permission-aware hybrid RAG with evals.", no_args_is_help=True)
app.command()(ingest.ingest)
app.command()(ask.ask)
app.command()(ask.search)
app.command("eval")(evaluate.evaluate)
app.command()(users.user)
app.command()(serve.serve)
app.command()(serve.migrate)

if __name__ == "__main__":
    app()
