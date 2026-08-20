import os
import sqlite3
from typing import Dict, Any
from . import mcp

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DB = os.path.join(BASE_DIR, "database", "hotel.db")

# Global in-memory status tracking for active tools
ENABLED_TOOLS: Dict[str, bool] = {
    "get_reservation": True,
    "search_available_rooms": True,
    "find_alternative_branch": True,
    "search_all_branches": True,
    "analyze_reservation": True,
    "approve_guest_transfer": True,
    "resolve_overbooking": True,
    "recommend_compensation": True,
    "approve_compensation": True,
}

def get_db():
    return sqlite3.connect(DB)

# --- Admin Platform Helper Functions ---
def list_tool_statuses() -> Dict[str, bool]:
    """Returns all tools and their active status."""
    return ENABLED_TOOLS

def set_tool_status(tool_name: str, enabled: bool) -> Dict[str, Any]:
    """Dynamically enables/disables a tool at runtime."""
    if tool_name not in ENABLED_TOOLS:
        return {"success": False, "error": f"Tool '{tool_name}' not found."}
    ENABLED_TOOLS[tool_name] = enabled
    return {"success": True, "tool": tool_name, "enabled": enabled}

def is_tool_enabled(tool_name: str) -> bool:
    """Checks if a tool is currently allowed to run."""
    return ENABLED_TOOLS.get(tool_name, False)

if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        print("Starting Aurelia Hotel MCP Server...")
        mcp.run(transport="streamable-http")