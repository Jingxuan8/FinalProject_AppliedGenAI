# test_router_planner_v2.py

from agent_graph.router import Router
from agent_graph.planner import Planner


def run_test(query: str):
    print("\n==============================================")
    print("USER QUERY:", query)
    print("==============================================")

    # Create agents
    router = Router()
    planner = Planner()

    # Initial state
    state = {
        "user_query": query,
        "debug_log": [],
    }

    # ---- Router ----
    state = router(state)

    print("\n--- ROUTER RESULT ---")
    print("intent:", state.get("intent"))
    print("constraints:", state.get("constraints"))
    print("safety_flag:", state.get("safety_flag"))

    # ---- Planner ----
    state = planner(state)

    print("\n--- PLANNER RESULT ---")
    print(state.get("planner_output"))

    print("\n--- DEBUG LOG ---")
    for log in state.get("debug_log", []):
        print(log)


if __name__ == "__main__":
    tests = [
        "What is the current price of a PS5 controller?",
        "Is the PS5 controller in stock right now?",
        "Recommend a card game under 20 dollars.",
        "Recommend a cooperative board game.",
        "Tell me something interesting to buy.",
    ]

    for t in tests:
        run_test(t)
