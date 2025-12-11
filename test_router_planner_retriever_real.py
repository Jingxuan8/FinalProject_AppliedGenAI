# test_router_planner_retriever_real.py

from agent_graph.router import Router
from agent_graph.planner import Planner
from agent_graph.retriever import Retriever
from agent_graph.mcp_client import MCPClientWrapper   # <= You already use this in project


def run_test(query: str):
    print("\n====================================================")
    print("USER QUERY:", query)
    print("====================================================")

    # Agents
    router = Router()
    planner = Planner()
    retriever = Retriever()

    # IMPORTANT: attach REAL MCP client so retriever can call rag/web tools
    retriever.mcp = MCPClientWrapper()  # <-- This is how your old retriever worked

    # Initial pipeline state
    state = {
        "user_query": query,
        "debug_log": []
    }

    # -----------------------
    # 1. ROUTER
    # -----------------------
    state = router(state)

    print("\n--- ROUTER OUTPUT ---")
    print("intent:", state.get("intent"))
    print("constraints:", state.get("constraints"))
    print("safety_flag:", state.get("safety_flag"))

    # -----------------------
    # 2. PLANNER
    # -----------------------
    state = planner(state)

    print("\n--- PLANNER OUTPUT ---")
    print(state.get("planner_output"))

    # -----------------------
    # 3. RETRIEVER (REAL MCP)
    # -----------------------
    state = retriever(state)

    print("\n--- RETRIEVER OUTPUT ---")
    print("retrieval_source:", state.get("retrieval_source"))
    print("\nRAG RESULTS:")
    print(state.get("rag_results"))
    print("\nWEB RESULTS:")
    print(state.get("web_results"))

    # -----------------------
    # DEBUG LOG
    # -----------------------
    print("\n--- DEBUG LOG ---")
    for line in state["debug_log"]:
        print(line)


if __name__ == "__main__":
    tests = [
        "What is the current price of a PS5 controller?",
        "Is the PS5 controller in stock right now?",
        "Recommend a card game under 20 dollars.",
        "Recommend a cooperative board game.",
        "Tell me something interesting to buy.",
        "Compare Nintendo Switch and PS5."
    ]

    for q in tests:
        run_test(q)
