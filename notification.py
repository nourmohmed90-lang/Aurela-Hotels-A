from __future__ import annotations

import asyncio
from typing import Any
from mcp import types

async def handle_notification(
        agent,
        message: Any,
    ) -> None:
        """Handle incoming server notifications (e.g. tool list changes, server logs)."""
        if isinstance(message, Exception):
            print(f"[notifications] transport error: {message}")
            return

        if not isinstance(message, types.ServerNotification):
            return  # server-initiated *requests* go through their own callbacks

        notification = message.root

        if isinstance(notification, types.ToolListChangedNotification):
            print("[notifications] tools/list_changed received -> refreshing catalog")

            # fix: properly call refresh_catalog via tools module
            agent._pending_notification_tasks.append(
            asyncio.create_task(agent.tools.refresh_catalog(agent))
            )
        elif isinstance(notification, types.LoggingMessageNotification):
            print(f"[server log:{notification.params.level}] {notification.params.data}")
        