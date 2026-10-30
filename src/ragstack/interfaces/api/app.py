from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ragstack import __version__
from ragstack.infrastructure.config.logging import configure_logging
from ragstack.infrastructure.config.settings import Settings
from ragstack.infrastructure.di.container import Container, build_container

from .errors import register_error_handlers
from .routers import ask, documents, health


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    s = settings or Settings()
    configure_logging(s.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or build_container(s)
        await c.start()
        app.state.container = c
        try:
            yield
        finally:
            if container is None:
                await c.stop()

    app = FastAPI(title="ragstack", version=__version__, lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router, prefix="/v1")
    app.include_router(ask.router, prefix="/v1")
    return app
