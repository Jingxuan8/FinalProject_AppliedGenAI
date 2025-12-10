# agent_graph/mcp_client.py

import json
import asyncio
from typing import Any, Dict, List
from fastmcp.client import Client


class MCPClientWrapper:
    """
    Synchronous wrapper around the FastMCP async Client.
    Compatible with FastMCP 2.13.3.
    """

    def __init__(self, host="localhost", port=8765):
        # Connect to FastMCP server via HTTP/SSE endpoint
        self.url = f"http://{host}:{port}"
        self.client = Client(self.url)

    def _run(self, coro):
        """Run async FastMCP operations synchronously."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Call an MCP tool synchronously.
        Returns a Python list of dict results.
        """
        result = self._run(self.client.call_tool(tool_name, arguments))

        if not result.content:
            return []

        part = result.content[0]

        if part.type == "text":
            try:
                return json.loads(part.text)
            except Exception:
                return []

        return result.content

    def list_tools(self) -> List[str]:
        result = self._run(self.client.list_tools())
        return [t.name for t in result]
