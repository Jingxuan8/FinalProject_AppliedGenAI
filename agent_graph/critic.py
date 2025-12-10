# agent_graph/critic.py

from __future__ import annotations
from typing import Dict, Any, List

from .state import AgentState


class Critic:
    """
    Critic / Merger Agent.

    Responsibilities:
    - Inspect RAG and Web results
    - Prefer RAG when planner_mode == "rag"
    - Prefer Web when planner_mode == "web"
    - Fallback to the other source if the preferred one is empty
    - Deduplicate results (by doc_id or title)
    - Limit final list size
    """

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        rag_results: List[Dict[str, Any]] = state.rag_results or []
        web_results: List[Dict[str, Any]] = state.web_results or []

        planner_output = getattr(state, "planner_output", {}) or {}
        planner_mode = planner_output.get("mode", "rag")

        merged: List[Dict[str, Any]] = []
        source: str = "none"

        # ------------------------------------------------------
        # Prefer RAG if planner says "rag"
        # ------------------------------------------------------
        if planner_mode == "rag":
            if rag_results:
                merged = rag_results
                source = "rag"
            elif web_results:
                merged = web_results
                source = "web"

        # ------------------------------------------------------
        # Prefer WEB if planner says "web"
        # ------------------------------------------------------
        elif planner_mode == "web":
            if web_results:
                merged = web_results
                source = "web"
            elif rag_results:
                merged = rag_results
                source = "rag"

        # ------------------------------------------------------
        # If no mode matched (unlikely), still try to salvage
        # ------------------------------------------------------
        if not merged and (rag_results or web_results):
            # fallback: use whichever has data
            if rag_results:
                merged = rag_results
                source = "rag"
            else:
                merged = web_results
                source = "web"

        # ------------------------------------------------------
        # Deduplicate by doc_id or title
        # ------------------------------------------------------
        seen = set()
        unique_results: List[Dict[str, Any]] = []
        for item in merged:
            key = item.get("doc_id") or item.get("title")
            if key and key not in seen:
                seen.add(key)
                unique_results.append(item)

        # Limit to top N (you can tune this)
        unique_results = unique_results[:8]

        debug_msg = (
            f"[CRITIC] planner_mode={planner_mode}, "
            f"chosen_source={source}, "
            f"rag={len(rag_results)}, web={len(web_results)}, "
            f"merged={len(unique_results)}"
        )

        return {
            "merged_results": unique_results,
            "retrieval_source": source if unique_results else None,
            "debug_log": state.debug_log + [debug_msg],
        }


__all__ = ["Critic"]
