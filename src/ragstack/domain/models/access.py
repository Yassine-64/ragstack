from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PrincipalType(StrEnum):
    USER = "user"
    GROUP = "group"
    EVERYONE = "everyone"  # everyone in the tenant


class AccessGrant(BaseModel):
    document_id: str
    principal_type: PrincipalType
    principal_id: str = Field(default="", description="empty for EVERYONE")


class Principal(BaseModel):
    """Who is asking. Groups are resolved by the server from the database, never trusted from the request."""

    tenant_id: str
    user_id: str
    group_ids: frozenset[str] = Field(default_factory=frozenset)
