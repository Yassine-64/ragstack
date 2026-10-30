from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ragstack.application.config import RetrievalConfig
from ragstack.infrastructure.config.logging import configure_logging
from ragstack.infrastructure.config.settings import Settings
from ragstack.infrastructure.di.container import Container, build_container, load_config


def run[T](fn: Callable[[Container], Awaitable[T]], config: str | None = None, verbose: bool = False) -> T:
    settings = Settings()
    configure_logging("DEBUG" if verbose else settings.log_level)
    cfg: RetrievalConfig = load_config(config or settings.config_path)

    async def _go() -> T:
        c = build_container(settings, cfg)
        await c.start()
        try:
            return await fn(c)
        finally:
            await c.stop()

    return asyncio.run(_go())
