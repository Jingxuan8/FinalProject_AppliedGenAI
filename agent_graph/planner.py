"""
Planner Agent for Agentic Orchestration.

Decides:
- Which sources to use (rag and/or web)
- What filters to apply
- What the retriever should do

Improved Version:
- Dual rag + web calls for price-check tasks
- Availability and price comparison flags
- Category normalization restored
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
    """Normalize user-specified categories to canonical catalog categories."""
    if not cat:
        return None
    key = cat.strip().lower()
    return CATEGORY_MAP.get(key, None)


# ------------------------------------------------------------
# PLANNER CLASS (Improved Version)
# ------------------------------------------------------------

class Planner:
    def __init__(self):
        pass

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:

        raw_intent = state.get("intent", "unknown")
        # Normalize intent for safety (handles: spaces, casing)
        intent = raw_intent.replace(" ", "_").strip().lower()

        constraints = state.get("constraints") or {}
        need_live = state.get("need_live_price", False)

        # ---------------------------
        # Build FILTERS (for MCP tools)
        # ---------------------------
        filters = {
            "max_price": constraints.get("max_price") or constraints.get("budget"),
            "category": normalize_category(constraints.get("category")),
            "brand": constraints.get("brand"),
        }

        # ============================================================
        # CASE 1 — Price-check intent: use BOTH rag + web
        # ============================================================
        if intent == "check_price":
            planner_output = {
                "intent": raw_intent,
                "use_rag": True,
                "use_web": True,
                "compare_price": True,
                "compare_availability": True,
                "filters": filters,
            }

            debug_msg = (
                "[PLANNER] Price-check intent → using BOTH rag and web. "
                f"filters={filters}"
            )

        # ============================================================
        # CASE 2 — Availability intent (Dual rag + web)
        # ============================================================
        elif intent == "check_availability":
            planner_output = {
                "intent": raw_intent,
                "use_rag": True,
                "use_web": True,
                "compare_price": False,
                "compare_availability": True,   # important!
                "filters": filters,
            }

            debug_msg = (
                "[PLANNER] Availability intent → using BOTH rag and web for stock check. "
                f"filters={filters}"
            )

        # ============================================================
        # CASE 3 — Normal product search (rag only)
        # ============================================================
        elif intent == "search":
            planner_output = {
                "intent": raw_intent,
                "use_rag": True,
                "use_web": False,
                "compare_price": False,
                "compare_availability": False,
                "filters": filters,
            }

            debug_msg = f"[PLANNER] Search intent → rag only. filters={filters}"

        # ============================================================
        # CASE 4 — "Compare" queries (optional/expandable)
        # ============================================================
        elif intent == "compare":
            planner_output = {
                "intent": raw_intent,
                "use_rag": True,
                "use_web": True,
                "compare_price": True,
                "compare_availability": True,
                "filters": filters,
            }

            debug_msg = (
                "[PLANNER] Compare intent → dual rag + web. "
                f"filters={filters}"
            )

        # ============================================================
        # CASE 5 — Fallback: preserve old logic (rag or web only)
        # ============================================================
        else:
            mode = "web" if need_live else "rag"

            planner_output = {
                "intent": raw_intent,
                "use_rag": (mode == "rag"),
                "use_web": (mode == "web"),
                "compare_price": False,
                "compare_availability": False,
                "filters": filters,
            }

            debug_msg = f"[PLANNER] Default → mode={mode}, filters={filters}"

        # ============================================================
        # Return updated state
        # ============================================================
        return {
            **state,
            "planner_output": planner_output,
            "debug_log": state.get("debug_log", []) + [debug_msg],
        }


__all__ = ["Planner"]
