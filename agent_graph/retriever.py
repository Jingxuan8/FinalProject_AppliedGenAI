# agent_graph/retriever.py

from __future__ import annotations
from typing import Dict, Any
from .mcp_client import MCPClientWrapper


class Retriever:
    """
    Retriever bridges Planner output -> MCP tools (rag_search, web_search).
    This version matches your teammate's original schemas and safely handles
    both dict and list MCP outputs.
    """

    def __init__(self):
        self.mcp = MCPClientWrapper()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("planner_output", {})
        mode = plan.get("mode")

        query = state.get("user_query", "")
        filters = plan.get("filters", {}) or {}

        debug = state.get("debug_log", [])

        # ============================================================
        # CASE A — RAG SEARCH
        # ============================================================
        if mode == "rag":

            # Match FastMCP rag_search signature in mcp_server/server.py
            rag_request = {
                "query": query,
                "category": filters.get("category"),
                "max_price": filters.get("max_price"),
                "num_results": 5,
            }

            debug.append(f"[RETRIEVER] rag_search args={rag_request}")

            rag_results = self.mcp.call_tool("rag_search", rag_request)

            # ---- Normalize rag_results safely ----
            if isinstance(rag_results, dict):
                rag_items = rag_results.get("results", [])
            elif isinstance(rag_results, list):
                rag_items = rag_results
            else:
                rag_items = []

            debug.append(f"[RETRIEVER] rag returned {len(rag_items)} items")

            return {
                **state,
                "retrieval_source": "rag",
                "rag_results": rag_items,
                "web_results": [],
                "debug_log": debug,
            }

        # ============================================================
        # CASE B — WEB SEARCH
        # ============================================================
        elif mode == "web":

            web_request = {
                "query": query,
                "num_results": 5
            }

            debug.append(f"[RETRIEVER] web_search args={web_request}")

            web_results = self.mcp.call_tool("web_search", web_request)

            # ---- Normalize web_results safely ----
            if isinstance(web_results, dict):
                web_items = web_results.get("results", [])
            elif isinstance(web_results, list):
                web_items = web_results
            else:
                web_items = []

            debug.append(f"[RETRIEVER] web returned {len(web_items)} items")

            return {
                **state,
                "retrieval_source": "web",
                "rag_results": [],
                "web_results": web_items,
                "debug_log": debug,
            }

        # ============================================================
        # FALLBACK — NO PLAN
        # ============================================================
        debug.append("[RETRIEVER] No valid retrieval mode")

        return {
            **state,
            "retrieval_source": None,
            "rag_results": [],
            "web_results": [],
            "debug_log": debug,
        }


__all__ = ["Retriever"]
