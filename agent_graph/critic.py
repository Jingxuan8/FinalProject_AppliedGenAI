# agent_graph/critic.py

"""
Critic / Merger node for LangGraph orchestration.

Responsibilities:
- Reconcile RAG (catalog) and web (live) results.
- Combine them into merged_results.
- Detect conflicting price information.
- Prefer catalog for static info.
- Prefer web for dynamic info (current price, availability).
- Produce consistent, safe, grounded data for the Answerer.

This is required by the project rubric and tool_prompts.md.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List

from agent_graph.state import GraphState


# ------------------------------------------------------------
# Utility: match catalog & web products by title similarity
# ------------------------------------------------------------

def titles_similar(t1: str, t2: str, threshold: float = 0.72) -> bool:
    """
    Fuzzy title matcher using difflib ratio.
    Titles are normalized before comparison.
    """
    if not t1 or not t2:
        return False

    t1 = t1.lower().strip()
    t2 = t2.lower().strip()

    ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
    return ratio >= threshold


# ------------------------------------------------------------
# Merge logic
# ------------------------------------------------------------

def merge_catalog_and_web(
    rag_results: List[Dict[str, Any]],
    web_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merges RAG and web search results following project rules.

    Steps:
    1. Try to align catalog & web products by fuzzy title matching.
    2. Combine static + dynamic info into a unified record.
    3. If price discrepancy > 20%, flag it.
    """

    merged: List[Dict[str, Any]] = []
    used_web = set()

    for r in rag_results:
        r_title = r.get("title", "")
        r_price = r.get("price")

        best_web = None
        best_ratio = 0.0

        # ------------------------------------------------------------
        # 1. Find matching web result
        # ------------------------------------------------------------
        for idx, w in enumerate(web_results):
            if idx in used_web:
                continue

            w_title = w.get("title", "")
            ratio = difflib.SequenceMatcher(None, r_title.lower(), w_title.lower()).ratio()

            if ratio > best_ratio and ratio >= 0.72:
                best_ratio = ratio
                best_web = (idx, w)

        # ------------------------------------------------------------
        # 2. Construct merged record
        # ------------------------------------------------------------
        merged_record = dict(r)  # start from catalog (static info)

        if best_web:
            w_idx, w = best_web
            used_web.add(w_idx)

            # Live price + availability override
            if w.get("price") is not None:
                merged_record["current_price"] = w["price"]
            if w.get("availability"):
                merged_record["availability"] = w["availability"]
            merged_record["web_source"] = w.get("source")

            # Price discrepancy detection
            if r_price and w.get("price"):
                old = float(r_price)
                new = float(w["price"])
                if abs(new - old) / max(old, 1e-6) > 0.20:
                    merged_record["price_discrepancy"] = True
                else:
                    merged_record["price_discrepancy"] = False

        merged.append(merged_record)

    # ------------------------------------------------------------
    # 3. Add remaining unmatched web results
    # ------------------------------------------------------------
    for idx, w in enumerate(web_results):
        if idx not in used_web:
            standalone = {
                "title": w.get("title", ""),
                "current_price": w.get("price"),
                "availability": w.get("availability"),
                "web_source": w.get("source"),
                "doc_id": None,  # no catalog doc_id
                "price_discrepancy": None,
            }
            merged.append(standalone)

    return merged


# ------------------------------------------------------------
# Critic Node
# ------------------------------------------------------------

def critic_node(state: GraphState) -> GraphState:
    """
    Reconciles catalog and web results into merged_results.
    """

    rag = state.get("rag_results", [])
    web = state.get("web_results", [])

    merged = merge_catalog_and_web(rag, web)

    debug_msg = f"[CRITIC] merged {len(rag)} RAG + {len(web)} WEB → {len(merged)} unified"

    return {
        **state,
        "merged_results": merged,
        "debug_log": state.get("debug_log", []) + [debug_msg],
    }


__all__ = [
    "critic_node",
    "merge_catalog_and_web",
    "titles_similar",
]
