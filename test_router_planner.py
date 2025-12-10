# test_router_planner.py

from agent_graph.router import Router
from agent_graph.planner import Planner
from langgraph.graph import StateGraph, END


# ------------------------------------------------------------
# Build a minimal graph: router → planner only
# ------------------------------------------------------------
def build_test_graph():
    graph = StateGraph(dict)

    router = Router("gpt-4o-mini")
    planner = Planner()

    graph.add_node("router", router)
    graph.add_node("planner", planner)

    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", END)

    return graph.compile()


# ------------------------------------------------------------
# Test function
# ------------------------------------------------------------
def run_test(query):
    graph = build_test_graph()

    state = {
        "user_query": query,
        "intent": None,
        "constraints": {},
        "need_live_price": False,
        "planner_output": {},
        "debug_log": [],
    }

    result = graph.invoke(state)

    print("\n=== ROUTER OUTPUT ===")
    print(result.get("intent"))
    print(result.get("constraints"))
    print(result.get("need_live_price"))
    print()

    print("=== PLANNER OUTPUT ===")
    print(result.get("planner_output"))
    print()

    print("=== DEBUG LOG ===")
    for msg in result.get("debug_log", []):
        print(msg)


# ------------------------------------------------------------
# Run three test cases
# ------------------------------------------------------------
if __name__ == "__main__":
    run_test("recommend a cooperative board game under 30 dollars")
    run_test("what is the current price of a ps5 controller")
    run_test("compare logitech g502 and razer basilisk mice")
