# sequential_test.py
"""
Sequential full pipeline test:
Router -> Planner -> Retriever -> Answerer
This bypasses LangGraph and runs everything step-by-step for debugging.
"""

from agent_graph.router import Router
from agent_graph.planner import Planner
from agent_graph.retriever import Retriever
from agent_graph.answerer import Answerer


# --------------------------------------------------------
# Create objects
# --------------------------------------------------------

router = Router("gpt-4o-mini")
planner = Planner()
retriever = Retriever()
answerer = Answerer("gpt-4o-mini")   # Hybrid Answerer/Critic


# --------------------------------------------------------
# Helper for printing clean block separators
# --------------------------------------------------------

def print_block(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# --------------------------------------------------------
# Full sequential pipeline test
# --------------------------------------------------------

def sequential_test(query: str):
    print_block(f"USER QUERY: {query}")

    # ---- INITIAL STATE ----
    state = {
        "user_query": query,
        "intent": None,
        "constraints": {},
        "need_live_price": False,
        "planner_output": {},
        "rag_results": [],
        "web_results": [],
        "retrieval_source": None,
        "merged_results": [],
        "final_answer": "",
        "debug_log": [],
        "safety_flag": False,
    }

    print("\n>>> INITIAL STATE:")
    print(state)

    # ---- Step 1: Router ----
    state = router(state)
    print("\n--- ROUTER FULL OUTPUT ---")
    print(state)

    # ---- Step 2: Planner ----
    state = planner(state)
    print("\n--- PLANNER FULL OUTPUT ---")
    print(state)

    # ---- Step 3: Retriever ----
    state = retriever(state)
    print("\n--- RETRIEVER FULL OUTPUT ---")
    print(state)

    # ---- Step 4: Answerer (final LLM formatting) ----
    state = answerer(state)
    print("\n--- ANSWERER FULL OUTPUT ---")
    print(state)

    # ---- Final Answer ----
    print("\n=== FINAL ANSWER ===")
    print(state.get("final_answer"))

    # ---- Debug log ----
    print("\n=== DEBUG LOG ===")
    for line in state["debug_log"]:
        print(line)


# --------------------------------------------------------
# RUN TEST CASES
# --------------------------------------------------------

test_cases = [
    "recommend a cooperative board game under 30 dollars",
    # "what is the current price of a ps5 controller",
    # "recommend a popular card game for 4 players",
]

for q in test_cases:
    sequential_test(q)
