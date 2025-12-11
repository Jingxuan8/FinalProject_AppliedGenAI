# test_router_only.py

from agent_graph.router import Router

def run_router_test(query: str):
    router = Router()
    state = {
        "user_query": query,
        "debug_log": []
    }
    result = router(state)
    print(result)   # print full state only


if __name__ == "__main__":
    test_cases = [
        "What is the current price of a PS5 controller?",
        "Is the PS5 controller in stock right now?",
        "Recommend a card game under 20 dollars.",
        "Recommend a cooperative board game.",
        "Tell me something interesting to buy.",
        "Compare Nintendo Switch and PS5.",
        "recommend a cooperative board game under 30 dollars"
    ]

    for q in test_cases:
        run_router_test(q)
