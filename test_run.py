# test_run.py
import os
os.environ["OPENAI_API_KEY"] = ""
from agent_graph import run_pipeline
from agent_graph.mcp_client import MCPClientWrapper
from openai import OpenAI
# ---------------------------------------
# 1. Load OpenAI model
# ---------------------------------------

# Make sure your OPENAI_API_KEY is set in the environment
client = OpenAI(api_key="YOUR API KEY")

# Wrap model name (matches your config & router/planner usage)
class ModelWrapper:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.chat = client.chat

model = ModelWrapper("gpt-4o-mini")


# ---------------------------------------
# 2. Connect to MCP server
# ---------------------------------------

# Connect to server running in SSE mode
mcp = MCPClientWrapper(host="localhost", port=8765)


# ---------------------------------------
# 3. Test query
# ---------------------------------------

query = "recommend a fun cooperative board game under $30"

print("\n============================")
print("USER QUERY:", query)
print("============================\n")

# Run full pipeline
final_state = run_pipeline(query, model, mcp, return_state=True)

# ---------------------------------------
# 4. Print final answer
# ---------------------------------------

print("\n===== FINAL ANSWER =====\n")
print(final_state["final_answer"])

# ---------------------------------------
# 5. Print debug log
# ---------------------------------------

print("\n===== DEBUG LOG =====\n")
for line in final_state["debug_log"]:
    print(line)

print("\nPipeline complete.\n")
