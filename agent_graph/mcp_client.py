import asyncio
import json
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams


class MCPClientWrapper:
    """
    Wrapper around ADK's MCP Toolset with:
    - Suppressed async generator shutdown noise
    - Clean handling of ADK tool wrappers {"content": ...}
    - Clean handling of MCP "text parts" [{type: text, text: "..."}]
    """

    def __init__(self, host="localhost", port=8765):
        url = f"http://{host}:{port}/mcp"
        self.toolset = McpToolset(
            connection_params=StreamableHTTPConnectionParams(url=url)
        )

    # ------------------------------------------------------------------
    # SAFE RUNNER: suppresses all the annoying async shutdown errors
    # ------------------------------------------------------------------
    def call_tool(self, name: str, args: dict):
        """Compatibility wrapper so tool calls work uniformly."""
        return self.call_tool_sync(name, args)

    def _safe_run(self, coro):
        try:
            return asyncio.run(coro)

        # Suppress anyio cancellation noise
        except RuntimeError as e:
            if "cancel scope" in str(e) or "different task" in str(e):
                return None
            return None

        except GeneratorExit:
            return None

        except Exception:
            return None

    # ------------------------------------------------------------------
    # TOOL LIST
    # ------------------------------------------------------------------
    def list_tools(self):
        result = self._safe_run(self._list_tools())
        return result or []

    async def _list_tools(self):
        tools = await self.toolset.get_tools()
        return [t.name for t in tools]

    # ------------------------------------------------------------------
    # MAIN TOOL CALL
    # ------------------------------------------------------------------
    def call(self, tool_name, arguments):
        """
        Calls an MCP tool synchronously and returns a CLEAN Python object.

        Handles:
        - ADK responses: {"content": [...]}
        - MCP text-part responses:
              [{"type":"text","text":"[...json...]"}]
        """
        raw = self._safe_run(self._call(tool_name, arguments))

        # Case 1: ADK's standard {"content": [...]}
        if isinstance(raw, dict) and "content" in raw:
            return raw["content"]

        # Case 2: Text-part MCP payload
        if isinstance(raw, list) and len(raw) > 0:
            first = raw[0]
            if isinstance(first, dict) and first.get("type") == "text":
                text = first.get("text", "")
                try:
                    return json.loads(text)
                except Exception:
                    return text

        # Case 3: Already final
        return raw or []
    # ------------------------------------------------------------
    # PUBLIC SYNCHRONOUS WRAPPER (retriever uses ONLY this)
    # ------------------------------------------------------------
    def call_tool_sync(self, tool_name: str, arguments: dict):
        """
        Synchronous wrapper used by retriever.
        Ensures that the result is always a clean Python list/dict.
        """

        raw = self.call(tool_name, arguments)

        if not raw:
            return []

        # ADK-style: {"content": [...]}
        if isinstance(raw, dict) and "content" in raw:
            return raw["content"]

        # Raw MCP format: [{"type": "text", "text": "..."}]
        if isinstance(raw, list) and len(raw) > 0:
            item = raw[0]
            if isinstance(item, dict) and item.get("type") == "text":
                import json
                try:
                    return json.loads(item["text"])
                except Exception:
                    return []

        return raw

    # ------------------------------------------------------------------
    # INTERNAL RAW TOOL CALL
    # ------------------------------------------------------------------
    async def _call(self, tool_name, arguments):
        tools = await self.toolset.get_tools()
        tool = next((t for t in tools if t.name == tool_name), None)

        if tool is None:
            print(f"[MCPClientWrapper] No tool named '{tool_name}'")
            return []

        try:
            raw = await tool.run_async(args=arguments, tool_context=None)

        except GeneratorExit:
            return None

        except RuntimeError as e:
            if "cancel scope" in str(e) or "different task" in str(e):
                return None
            raise

        return raw
