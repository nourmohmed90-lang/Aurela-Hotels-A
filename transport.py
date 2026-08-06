from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Callable

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.client.sse import sse_client
from agent.config import AgentConfig, TransportMode

GetSessionId = Callable[[], "str | None"]

@asynccontextmanager
async def open_transport(
    config: AgentConfig,
) -> AsyncGenerator[tuple[Any, Any, Callable[[], str | None]], None]:
    """Open transport layer based on configuration (stdio or HTTP/SSE)."""
    if config.transport_mode == "stdio":
        server_params = StdioServerParameters(
            command=config.stdio_command,
            args=list(config.stdio_args),
            request_timeout_seconds=config.request_timeout_seconds,)
        async with stdio_client(server_params) as (read, write):
            yield read, write, lambda: None

    elif config.transport_mode == TransportMode.HTTP:
        if not config.http_url:
            raise ValueError("http_url is required for HTTP transport.")

        async with streamable_http_client(
            config.http_url
        ) as (read, write, get_session_id):
            yield read, write, get_session_id

    else:
        raise ValueError(f"Unsupported transport mode: {config.transport_mode}")