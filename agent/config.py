from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv


class TransportMode(str, Enum):
    STDIO = "stdio"
    HTTP = "http"


@dataclass(frozen=True)
class AgentConfig:
    # transport
    transport_mode: TransportMode
    stdio_command: str
    stdio_args: tuple[str, ...]
    http_url: str | None
    http_headers: dict[str, str]

    # LLM credentials & model
    gemini_api_key: str | None
    gemini_model: str = "gemini-2.5-flash"

    # elicitation
    scripted_elicitation_responses: tuple[dict, ...] = field(default_factory=tuple)
    interactive_elicitation: bool = True

    # metadata
    client_name: str = "aurelia-hotels-agent"
    client_version: str = "0.3.0"
    request_timeout_seconds: float = 30.0


def _split_args(raw: str) -> tuple[str, ...]:
    return tuple(a for a in raw.split() if a)


def load_config() -> AgentConfig:
    """Build an AgentConfig from environment variables."""
    load_dotenv(override=True)

    mode_raw = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    try:
        transport_mode = TransportMode(mode_raw)
    except ValueError as exc:
        raise ValueError(
            f"MCP_TRANSPORT must be 'stdio' or 'http', got {mode_raw!r}"
        ) from exc

    http_url = os.environ.get("MCP_SERVER_URL")
    if transport_mode is TransportMode.HTTP and not http_url:
        raise ValueError("MCP_TRANSPORT=http requires MCP_SERVER_URL")

    auth_token = os.environ.get("MCP_AUTH_TOKEN")
    http_headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}

    interactive = (
        os.environ.get("MCP_ELICITATION_INTERACTIVE", "true").strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )

    gemini_key = os.environ.get("GEMINI_API_KEY")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    return AgentConfig(
        transport_mode=transport_mode,
        stdio_command=os.environ.get("MCP_STDIO_COMMAND", "python"),
        stdio_args=_split_args(
            os.environ.get("MCP_STDIO_ARGS", "-m mcp_server.server")
        ),
        http_url=http_url,
        http_headers=http_headers,
        gemini_api_key=gemini_key,
        gemini_model=gemini_model,
        request_timeout_seconds=float(os.environ.get("MCP_REQUEST_TIMEOUT", "30")),
        interactive_elicitation=interactive,
    )