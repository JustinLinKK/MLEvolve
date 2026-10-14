"""Read-only MCP surface for lesson profile retrieval."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .client import LessonProfileClient
from .config import LessonProfileSettings


def build_mcp_server(
    settings: LessonProfileSettings | None = None,
    *,
    client: LessonProfileClient | None = None,
) -> FastMCP:
    profile_client = client or LessonProfileClient(settings or LessonProfileSettings())
    profile_client.registry.initialize()
    server = FastMCP(
        name="mlevolve_lesson_profiles",
        instructions="Read evidence-backed family/hardware baselines and role-scoped lessons.",
    )

    @server.tool()
    def get_family_hardware_profile(
        profile_key: str,
        agent_role: str = "draft",
        current_delta_or_error: str = "",
    ) -> dict[str, Any]:
        profile = profile_client.registry.profile(profile_key)
        if profile is None:
            return {"family_hardware_profile": {"profile_key": profile_key, "match_level": "none"}}
        return profile_client.get_family_hardware_profile(
            agent_role=agent_role,
            identity=profile["identity"],
            code=current_delta_or_error,
            error=current_delta_or_error,
        )

    @server.tool()
    def search_lesson_profiles(
        query: str,
        agent_role: str,
        limit: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return profile_client.search_lesson_profiles(
            query=query,
            agent_role=agent_role,
            limit=limit,
            filters=filters or {},
        )

    return server


def run_stdio(settings: LessonProfileSettings | None = None) -> None:
    build_mcp_server(settings).run("stdio")
