from __future__ import annotations

from dataclasses import dataclass

from mcp import ClientSession
from mcp.types import InitializeResult


@dataclass(frozen=True)
class NegotiatedCapabilities:
    protocol_version: str
    server_name: str
    server_version: str
    instructions: str | None

    supports_tools: bool
    supports_tools_list_changed: bool
    supports_resources: bool
    supports_resources_list_changed: bool
    supports_prompts: bool
    supports_logging: bool

    @classmethod
    def from_initialize_result(cls, result: InitializeResult) -> NegotiatedCapabilities:
        caps = result.capabilities
        return cls(
            protocol_version=result.protocolVersion,
            server_name=result.serverInfo.name,
            server_version=result.serverInfo.version,
            instructions=result.instructions,
            supports_tools=caps.tools is not None,
            supports_tools_list_changed=bool(caps.tools and caps.tools.listChanged),
            supports_resources=caps.resources is not None,
            supports_resources_list_changed=bool(caps.resources and caps.resources.listChanged),
            supports_prompts=caps.prompts is not None,
            supports_logging=caps.logging is not None,
        )

    def require(self, capability: str, *, needed_for: str) -> None:
        """Ensure server declares feature support before execution."""
        if not getattr(self, f"supports_{capability}", False):
            raise RuntimeError(
                f"Server '{self.server_name}' does not declare '{capability}' "
                f"support (needed for: {needed_for})."
            )


async def perform_handshake(session: ClientSession) -> NegotiatedCapabilities:
    """Execute initialize/initialized MCP handshake."""
    result = await session.initialize()
    negotiated = NegotiatedCapabilities.from_initialize_result(result)

    print(
        f"[handshake] connected to '{negotiated.server_name}' "
        f"v{negotiated.server_version} (protocol {negotiated.protocol_version})"
    )
    print(
        "[handshake] declared server capabilities -> "
        f"tools.listChanged={negotiated.supports_tools_list_changed}  "
        f"resources.listChanged={negotiated.supports_resources_list_changed}  "
        f"prompts={negotiated.supports_prompts}  "
        f"logging={negotiated.supports_logging}"
    )
    if negotiated.instructions:
        print(f"[handshake] server instructions: {negotiated.instructions}")

    return negotiated