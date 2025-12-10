# agent_graph/graph.py

"""
LangGraph orchestration for the e-commerce multi-agent system.

This file wires together:
- Router
- Planner
- Retriever (MCP tool calls)
- Critic (merge catalog + web)
- Answerer (final response)

It exposes:
    build_graph(model, mcp_client)
    run_pipeline(query, model, mcp_client)

You will call `run_pipeline()` from a notebook, CLI, or Streamlit UI.
"""

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, END

from agent_graph.state import GraphState
from agent_graph.router import router_node
from agent_graph.planner import planner_node
from agent_graph.retriever import retriever_node
from agent_graph.critic import critic_node
from agent_graph.answerer import answerer_node


# ------------------------------------------------------------
# Build the LangGraph
# ------------------------------------------------------------

def build_graph(model, mcp_client):
    """
    Constructs the LangGraph graph with all nodes registered.
    """

    graph = StateGraph(GraphState)

    # Register nodes — these run sequentially
    graph.add_node("router", lambda s: router_node(s, model))
    graph.add_node("planner", lambda s: planner_node(s, model))
    graph.add_node("retriever", lambda s: retriever_node(s, mcp_client))
    graph.add_node("critic", critic_node)
    graph.add_node("answerer", lambda s: answerer_node(s, model))

    # Define execution flow
    graph.set_entry_point("router")
    graph.add_edge("router", "planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "critic")
    graph.add_edge("critic", "answerer")
    graph.add_edge("answerer", END)

    return graph


# ------------------------------------------------------------
# Run a single query through the entire pipeline
# ------------------------------------------------------------

def run_pipeline(
    query: str,
    model,
    mcp_client,
    return_state: bool = False,
) -> str | Dict[str, Any]:
    """
    Convenience function for testing the full agent pipeline.

    Args:
        query (str): user query
        model: LLM wrapper (must expose OpenAI-compatible chat.completions.create)
        mcp_client: your MCP tool caller
        return_state (bool): if True, returns full state dict

    Returns:
        final_answer (str) by default
        or full final state if return_state=True
    """

    # Initial state
    state: GraphState = {
        "user_query": query,
        "intent": "unknown",
        "constraints": {},
        "need_live_price": False,
        "plan": {},
        "rag_results": [],
        "web_results": [],
        "merged_results": [],
        "final_answer": "",
        "debug_log": [],
    }

    graph = build_graph(model, mcp_client)
    app = graph.compile()

    final_state = app.invoke(state)

    if return_state:
        return final_state

    return final_state.get("final_answer", "")


__all__ = ["build_graph", "run_pipeline"]
