# agent_graph/graph.py

from langgraph.graph import StateGraph, END
from agent_graph.router import Router
from agent_graph.planner import Planner
from agent_graph.retriever import Retriever
from agent_graph.critic import Critic
from agent_graph.answerer import Answerer


# ------------------------------------------------------------
# Build LangGraph pipeline
# ------------------------------------------------------------
def create_graph():
    graph = StateGraph(dict)   # state is a dictionary

    router = Router("gpt-4o-mini")
    planner = Planner()
    retriever = Retriever()
    critic = Critic("gpt-4o-mini")
    answerer = Answerer("gpt-4o-mini")

    # -----------------------------
    # 1. Add nodes in correct order
    # -----------------------------
    graph.add_node("router", router)
    graph.add_node("planner", planner)
    graph.add_node("retriever", retriever)
    graph.add_node("critic", critic)
    graph.add_node("answerer", answerer)

    # -----------------------------
    # 2. Define edges
    # -----------------------------
    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_edge("critic", "answerer")
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
    }

    return graph.invoke(state)
