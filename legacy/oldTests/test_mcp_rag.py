from agent_graph.mcp_client import MCPClientWrapper

mcp = MCPClientWrapper(host="localhost", port=8765)

print("=== Testing RAG tool directly ===")

# 1. List tools
tools = mcp.list_tools()
print("Available tools:", tools)

# 2. Basic RAG query
query = "mechanical keyboard"
print("\nQuerying RAG for:", query)

# NOTE: Correct tool name is rag_search (underscore)
result = mcp.call("rag_search", {"query": query, "num_results": 5})
print("\nRAW result object:", result)
try:
    print("Content:", result.get("content"))
except:
    print("Result is not a dict.")
