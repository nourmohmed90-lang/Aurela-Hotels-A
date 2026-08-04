from __future__ import annotations
from . import mcp
# NOTIFICATIONS

current_role = "receptionist"


@mcp.tool()
def get_available_tools():

    if current_role == "receptionist":

        return [
            "get_reservation",
            "search_available_rooms",
            "find_alternative_branch",
            "recommend_compensation"
        ]

    elif current_role == "manager":

        return [
            "get_reservation",
            "search_available_rooms",
            "find_alternative_branch",
            "recommend_compensation",
            "approve_guest_transfer",
            "resolve_overbooking"
        ]

    return []
@mcp.tool()
def promote_to_manager():

    global current_role

    current_role = "manager"

    return {

        "notification":
        "tools/list_changed",

        "new_role":
        current_role,

        "available_tools":
        get_available_tools()

    }
@mcp.tool()
def demote_to_receptionist():

    global current_role

    current_role = "receptionist"

    return {

        "notification":
        "tools/list_changed",

        "new_role":
        current_role,

        "available_tools":
        get_available_tools()

    }