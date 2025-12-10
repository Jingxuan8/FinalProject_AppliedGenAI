# agent_graph/retriever.py

"""
Retriever node for LangGraph orchestration.

This node:
- Executes MCP tool calls (rag.search, web.search) according to the Planner.
- Stores outputs in the shared GraphState.
- Handles tool call failures gracefully.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_graph.state import GraphState, AgentPlan


# ------------------------------------------------------------
# MCP Client Interface (model-agnostic)
# ------------------------------------------------------------

def call_mcp_tool(mcp_client, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Calls an MCP tool via the provided client.

    `mcp_client` must expose:
        mcp_client.call(tool_name, arguments)

    Expected tool names:
        - "rag.search"
        - "web.search"

    Returns list of dict results, or empty list on failure.
    """

    try:
        result = mcp_client.call(tool_name, arguments)
        # FastMCP returns raw JSON-friendly data, so just return it
        if isinstance(result, list):
            return result
        return []
    except Exception as e:
        print(f"[ERROR] MCP tool call failed ({tool_name}): {e}")
        return []


# ------------------------------------------------------------
# Retriever node
# ------------------------------------------------------------

def retriever_node(state: GraphState, mcp_client) -> GraphState:
    """
    Executes tool calls selected by the planner.

    The planner produces:
        plan = {
            use_rag: bool,
            use_web: bool,
            rerank: bool,
            rag_kwargs: {...},
            web_kwargs: {...},
            reason: "..."
        }

    This node uses the plan to call the appropriate tools.
    """

    plan: AgentPlan = state.get("plan", {})

    rag_results: List[Dict[str, Any]] = []
    web_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # 1. RAG SEARCH (default)
    # ------------------------------------------------------------
    if plan.get("use_rag", False):
        rag_args = plan.get("rag_kwargs", {})
        rag_args["query"] = state["user_query"]  # ensure query present
        rag_args.setdefault("num_results", 5)
        rag_args.setdefault("rerank", plan.get("rerank", False))

        rag_results = call_mcp_tool(mcp_client, "rag.search", rag_args)

    # ------------------------------------------------------------
    # 2. WEB SEARCH (real-time price/availability)
    # ------------------------------------------------------------
    if plan.get("use_web", False):
        web_args = plan.get("web_kwargs", {})
        web_args["query"] = state["user_query"]
        web_args.setdefault("num_results", 5)

        web_results = call_mcp_tool(mcp_client, "web.search", web_args)

    debug_msg = (
        f"[RETRIEVER] rag={len(rag_results)} results, "
        f"web={len(web_results)} results"
    )

    return {
        **state,
        "rag_results": rag_results,
        "web_results": web_results,
        "debug_log": state.get("debug_log", []) + [debug_msg],
    }


__all__ = ["retriever_node", "call_mcp_tool"]
