from __future__ import annotations

import json
from pathlib import Path

import typer

from ragstack.application.evaluation import comparison_table, load_golden
from ragstack.infrastructure.di.container import load_config

from ._common import run


def evaluate(
    golden: Path = typer.Argument(..., exists=True, help="JSONL golden set"),
    config: list[Path] = typer.Option(
        [], "--config", help="one or more RetrievalConfig YAMLs; default config when none"
    ),
    out: Path = typer.Option(Path("evals/runs"), help="where to write results"),
    fail_on_leak: bool = typer.Option(True, help="exit 1 when any config leaks a forbidden document"),
) -> None:
    """Run every config over the golden set, print the comparison table, write JSON results."""
    cases = load_golden(golden)
    configs = [load_config(p) for p in config] or [load_config(None)]

    async def go(c):  # type: ignore[no-untyped-def]
        runs = []
        for cfg in configs:
            sub = c.with_config(cfg)
            run_ = await sub.eval_runner().run(cfg.name, cfg.model_dump(), cases)
            runs.append(run_)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{run_.id}-{cfg.name}.json").write_text(run_.model_dump_json(indent=2))
            typer.secho(
                f"✓ {cfg.name}: {json.dumps({k: round(v, 3) for k, v in run_.metrics.items()})}",
                fg="green",
            )
        table = comparison_table(runs)
        typer.echo("\n" + table)
        (out / "latest.md").write_text(table)
        leaked = [r.name for r in runs if r.metrics.get("leak_rate", 0) > 0]
        if fail_on_leak and leaked:
            typer.secho(f"LEAK in {leaked}", fg="red", err=True)
            raise typer.Exit(code=1)

    run(go)
