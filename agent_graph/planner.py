# agent_graph/planner.py

"""
Planner Agent for Agentic Orchestration.

Decides:
- Which source to use (rag or web)
- What filters to apply
- What the retriever should do

We FIX:
- Proper category canonicalization
- Filtering logic for RAG
- Avoid unsupported fields for MCP tools
"""

from __future__ import annotations
from typing import Dict, Any
from .state import AgentState

# ------------------------------------------------------------
# CATEGORY NORMALIZATION TABLE
# ------------------------------------------------------------

CATEGORY_MAP = {
    "board game": "Board Games",
    "board games": "Board Games",
    "card game": "Card Games",
    "card games": "Card Games",
    "dice game": "Dice Games",
    "dice games": "Dice Games",
    "controller": "Controllers",
    "controllers": "Controllers",
    "mouse": "Mice",
    "mice": "Mice",
}

def normalize_category(cat: str | None) -> str | None:
    if not cat:
        return None
    key = cat.strip().lower()
    return CATEGORY_MAP.get(key, None)


# ------------------------------------------------------------
# PLANNER CLASS
# ------------------------------------------------------------

class Planner:
    def __init__(self):
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:

        intent = state.get("intent", "unknown")
        constraints = state.get("constraints", {}) or {}
        need_live = state.get("need_live_price", False)

        # ---------------------------
        # Build FILTERS
        # ---------------------------
        filters = {
            "max_price": constraints.get("max_price") or constraints.get("budget"),
            "category": normalize_category(constraints.get("category")),
            "brand": constraints.get("brand"),
        }

        # ---------------------------
        # Decide SOURCE
        # ---------------------------

        # If live data required → use web
        if need_live:
            mode = "web"

        # If comparing → try private first
        elif intent == "compare":
            mode = "rag"

        # Default → use rag
        else:
            mode = "rag"

        # ---------------------------
        # Build Planner Output
        # ---------------------------
        planner_output = {
            "mode": mode,
            "filters": filters,
            "intent": intent,
        }

        # Debug line for logging
        debug_msg = f"[PLANNER] mode={mode}, filters={filters}, intent={intent}"

        # Return updated state
        return {
            **state,
            "planner_output": planner_output,
            "debug_log": state.get("debug_log", []) + [debug_msg],
        }


__all__ = ["Planner"]
