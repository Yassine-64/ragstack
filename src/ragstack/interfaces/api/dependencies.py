from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ragstack.infrastructure.di.container import Container

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    c: Container = request.app.state.container
    return c


def require_api_key(
    container: Annotated[Container, Depends(get_container)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    expected = container.settings.api_key
    if expected and (credentials is None or credentials.credentials != expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")


def caller(
    x_user_id: Annotated[str, Header()], x_tenant_id: Annotated[str, Header()] = "default"
) -> tuple[str, str]:
    """Identity comes from headers set by the gateway/SSO in front of this API. Groups never do."""
    return x_tenant_id, x_user_id


ContainerDep = Annotated[Container, Depends(get_container)]
CallerDep = Annotated[tuple[str, str], Depends(caller)]
AuthDep = Depends(require_api_key)
