from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ragstack.domain.exceptions import DocumentNotFound, Forbidden, RagstackError

_STATUS: dict[type[RagstackError], int] = {DocumentNotFound: 404, Forbidden: 403}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RagstackError)
    async def _handle(request: Request, exc: RagstackError) -> JSONResponse:
        return JSONResponse(
            status_code=_STATUS.get(type(exc), 500),
            content={"error": type(exc).__name__, "detail": str(exc)},
        )
