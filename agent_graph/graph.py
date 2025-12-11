from langgraph.graph import StateGraph, END
from agent_graph.router import Router
from agent_graph.planner import Planner
from agent_graph.retriever import Retriever
from agent_graph.answerer import Answerer    # <-- Critic removed


# ------------------------------------------------------------
# Build LangGraph pipeline (4 nodes)
# ------------------------------------------------------------
def create_graph():
    graph = StateGraph(dict)

    router = Router()
    planner = Planner()
    retriever = Retriever()
    answerer = Answerer()   # <-- Hybrid Answerer/Critic

    # -----------------------------
    # 1. Add nodes
    # -----------------------------
    graph.add_node("router", router)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("answerer", answerer)

    # -----------------------------
    # 2. Define edges (4-node pipeline)
    # -----------------------------
    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", END)

    return graph.compile()


# ------------------------------------------------------------
# Helper for testing
# ------------------------------------------------------------
def run_pipeline(graph, query: str):
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

    return graph.invoke(state)
