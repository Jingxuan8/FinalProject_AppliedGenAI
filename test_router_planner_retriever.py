# test_router_planner_retriever.py

from agent_graph.router import Router
from agent_graph.planner import Planner
from agent_graph.retriever import Retriever
from agent_graph.mcp_client import MCPClientWrapper


# --------------------------------------------------------
# Create objects
# --------------------------------------------------------

router = Router("gpt-4o-mini")   # Router requires model name
planner = Planner()              # Planner does NOT take model name
mcp = MCPClientWrapper(host="localhost", port=8765)
retriever = Retriever()


# --------------------------------------------------------
# Helper for printing clean block separators
# --------------------------------------------------------

def print_block(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# --------------------------------------------------------
# Test pipeline function
# --------------------------------------------------------

def run_test(query):
    print_block(f"USER QUERY: {query}")

    # ---- step 1: router ----
    state = {"user_query": query, "debug_log": []}
    state = router(state)

    print("\n--- ROUTER OUTPUT ---")
    print("intent:", state.get("intent"))
    print("constraints:", state.get("constraints"))
    print("need_live_price:", state.get("need_live_price"))

    # ---- step 2: planner ----
    state = planner(state)

    print("\n--- PLANNER OUTPUT ---")
    print(state.get("planner_output"))

    # ---- step 3: retriever ----
    state = retriever(state)

    print("\n--- RETRIEVER OUTPUT ---")
    print("retrieval_source:", state.get("retrieval_source"))
    print("rag_results:", state.get("rag_results"))
    print("web_results:", state.get("web_results"))

    # ---- debug trace ----
    print("\n--- DEBUG LOG ---")
    for line in state.get("debug_log", []):
        print(line)


# --------------------------------------------------------
# RUN TEST CASES
# --------------------------------------------------------

test_cases = [
    "recommend a cooperative board game under 30 dollars",
    "what is the current price of a ps5 controller",
    "recommend a popular card game for 4 players",
]

for q in test_cases:
    run_test(q)
