# agent_graph/answerer.py

from __future__ import annotations
from typing import Dict, Any, List

from .state import AgentState


class Answerer:
    """
    Final Answer Agent.
    Summarizes retrieved product results into a grounded natural-language response.

    Requirements:
    - No hallucination (only use fields present in retrieved data)
    - Respect safety_flag set by Router
    - Format responses consistently
    - Distinguish between RAG-based and Web-based results
    """

    def _format_item(self, item: Dict[str, Any]) -> str:
        """Format one product into a clean bullet point."""
        title = item.get("title", "Unknown Product")
        price = item.get("price")
        brand = item.get("brand")
        category = item.get("category")
        url = item.get("product_url") or item.get("url")

        parts = [f"**{title}**"]

        if brand:
            parts.append(f"Brand: {brand}")

        if price is not None:
            parts.append(f"Price: ${price}")

        if category:
            parts.append(f"Category: {category}")

        if url:
            parts.append(f"[Link]({url})")

        return " | ".join(parts)

    # ------------------------------------------------------------
    # Main LangGraph entry point
    # ------------------------------------------------------------
    def __call__(self, state: AgentState) -> Dict[str, Any]:

        # ==========================================================
        # SAFETY CHECK
        # ==========================================================
        if getattr(state, "safety_flag", False):
            return {
                "final_answer": (
                    "I'm sorry, but I can’t assist with that request."
                )
            }

        results: List[Dict[str, Any]] = state.merged_results or []
        source = getattr(state, "retrieval_source", None)

        # ==========================================================
        # NO RESULTS → graceful fallback
        # ==========================================================
        if not results:
            return {
                "final_answer": (
                    "I couldn't find matching products for your query. "
                    "Try adjusting your constraints or rephrasing your request."
                )
            }

        # ==========================================================
        # INTRO TEXT BASED ON SOURCE
        # ==========================================================
        if source == "rag":
            intro = "Here are the top matches from our product catalog:\n\n"
        elif source == "web":
            intro = "Here are the most relevant live web results:\n\n"
        else:
            intro = "Here are the best matched products:\n\n"

        # ==========================================================
        # FORMAT BULLET LIST
        # ==========================================================
        bullets = "\n".join(f"- {self._format_item(item)}" for item in results)

        final_answer = intro + bullets

        debug_msg = f"[ANSWERER] produced_answer_with_{source or 'unknown'}"

        return {
            "final_answer": final_answer,
            "debug_log": state.debug_log + [debug_msg],
        }


__all__ = ["Answerer"]
