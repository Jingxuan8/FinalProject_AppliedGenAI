# agent_graph/retriever.py

from __future__ import annotations
from typing import Dict, Any
from .mcp_client import MCPClientWrapper


class Retriever:
    """
    Retriever bridges Planner output -> MCP tools (rag_search, web_search).

    This revised version:
    - Uses Planner's flags: use_rag / use_web / filters
    - Supports dual-source retrieval for price/availability checks
    - Keeps rag_results and web_results separate for the Answerer/Critic
    - Adds clear, tool-aware debug logs
    """

    def __init__(self):
        self.mcp = MCPClientWrapper()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("planner_output", {}) or {}

        # Flags from Planner
        use_rag: bool = bool(plan.get("use_rag", False))
        use_web: bool = bool(plan.get("use_web", False))

        query: str = state.get("user_query", "") or ""
        filters: Dict[str, Any] = plan.get("filters", {}) or {}

        debug = state.get("debug_log", [])

        rag_items = []
        web_items = []

        # ============================================================
        # CASE A — RAG SEARCH (private catalog)
        # ============================================================
        if use_rag:
            rag_request = {
                "query": query,
                "category": filters.get("category"),
                "max_price": filters.get("max_price"),
                "num_results": 5,
            }

            debug.append(f"[RETRIEVER] rag_search args={rag_request}")

            rag_results = self.mcp.call_tool("rag_search", rag_request)

            # Normalize rag_results
            if isinstance(rag_results, dict):
                rag_items = rag_results.get("results", [])
            elif isinstance(rag_results, list):
                rag_items = rag_results
            else:
                rag_items = []

            debug.append(f"[RETRIEVER] rag returned {len(rag_items)} items")

        # ============================================================
        # CASE B — WEB SEARCH (live web)
        # ============================================================
        if use_web:
            web_request = {
                "query": query,
                "num_results": 5,
            }

            debug.append(f"[RETRIEVER] web_search args={web_request}")

            web_results = self.mcp.call_tool("web_search", web_request)

            # Normalize web_results
            if isinstance(web_results, dict):
                web_items = web_results.get("results", [])
            elif isinstance(web_results, list):
                web_items = web_results
            else:
                web_items = []

            # Apply max price filter if needed
            max_price = filters.get("max_price")
            if max_price is not None:
                filtered = []
                for item in web_items:
                    price = item.get("price")
                    # allow items with unknown price (None) or <= max_price
                    if price is None or not isinstance(price, (int, float)):
                        filtered.append(item)
                    elif price <= max_price:
                        filtered.append(item)
                debug.append(
                    f"[RETRIEVER] web returned {len(web_items)} items; "
                    f"{len(filtered)} after max_price filter={max_price}"
                )
                web_items = filtered
            else:
                debug.append(f"[RETRIEVER] web returned {len(web_items)} items")

        # ============================================================
        # Determine retrieval_source
        # ============================================================
        if use_rag and use_web:
            retrieval_source = "mixed"
        elif use_rag:
            retrieval_source = "rag"
        elif use_web:
            retrieval_source = "web"
        else:
            retrieval_source = None
            debug.append("[RETRIEVER] No retrieval tools selected by planner")

        # ============================================================
        # SIMPLE PRICE RECONCILIATION (Rubric Requirement)
        # ============================================================
        resolved_price = None
        resolved_price_item = None

        compare_price = plan.get("compare_price", False)

        # Only reconcile when both sources used & price is relevant
        if use_rag and use_web and compare_price:

            best_item = None
            best_price = None

            for item in web_items:
                price = item.get("price")
                if isinstance(price, (int, float)):
                    if best_price is None or price < best_price:
                        best_price = price
                        best_item = item

            resolved_price = best_price
            resolved_price_item = best_item

            debug.append(f"[RETRIEVER] Resolved price = {resolved_price} item={resolved_price_item}")

        # Store both fields for Answerer
        state["resolved_price"] = resolved_price
        state["resolved_price_item"] = resolved_price_item

        # ============================================================
        # Return updated state
        # ============================================================
        return {
            **state,
            "retrieval_source": retrieval_source,
            "rag_results": rag_items,
            "web_results": web_items,
            "debug_log": debug,
        }


__all__ = ["Retriever"]
