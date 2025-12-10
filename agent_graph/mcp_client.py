import asyncio
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

class MCPClientWrapper:
    def __init__(self, host="localhost", port=8765):
        url = "http://localhost:8765/mcp"
        self.toolset = McpToolset(
            connection_params=StreamableHTTPConnectionParams(url=url)
        )

    def list_tools(self):
        return asyncio.run(self._list_tools())

    async def _list_tools(self):
        tools = await self.toolset.get_tools()
        return [t.name for t in tools]

    def call(self, tool_name, arguments):
        return asyncio.run(self._call(tool_name, arguments))

    async def _call(self, tool_name, arguments):
        tools = await self.toolset.get_tools()

        # Find correct tool
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break

        if tool is None:
            return []

        # Run tool asynchronously
        result = await tool.run_async(args=arguments, tool_context=None)

        # Result is already JSON-compatible
        return result
