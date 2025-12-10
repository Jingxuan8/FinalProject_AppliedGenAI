# test_run.py

from agent_graph.graph import create_graph
from agent_graph.state import AgentState


def run_query(query: str):
    print("\n" + "=" * 60)
    print(f"USER QUERY: {query}")
    print("=" * 60)

    graph = create_graph()

    # Initial state for graph execution
    state = {"user_query": query}
    # Run graph synchronously
    result = graph.invoke(state)

    # Graph may return either dict or AgentState depending on compilation
    if isinstance(result, dict):
        answer = result.get("final_answer")
        planner_output = result.get("planner_output")
        retrieval_source = result.get("retrieval_source")
        debug_log = result.get("debug_log", [])
    else:
        answer = getattr(result, "final_answer", None)
        planner_output = getattr(result, "planner_output", None)
        retrieval_source = getattr(result, "retrieval_source", None)
        debug_log = getattr(result, "debug_log", [])

    # Debug trace
    print("\n--- DEBUG TRACE ---")
    for line in debug_log:
        print(line)

    print("\n--- PLANNER OUTPUT ---")
    print(planner_output)

    print("\n--- RETRIEVAL SOURCE ---")
    print(retrieval_source)

    print("\n--- FINAL ANSWER ---\n")
    print(answer)
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # Basic test suite
    run_query("recommend a cooperative board game under 30 dollars")
    run_query("what is the current price of a ps5 controller")
    run_query("compare logitech g502 and razer basilisk mice")
