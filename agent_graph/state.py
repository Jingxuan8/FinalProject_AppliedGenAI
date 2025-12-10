"""
Unified state format for the LangGraph agentic pipeline.
Fully compatible with the updated Router → Planner → Retriever → Critic → Answerer flow.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypedDict


# ------------------------------------------------------------
# Constraint structure extracted by the Router
# ------------------------------------------------------------

class Constraints(TypedDict, total=False):
    budget: float
    category: str
    brand: str
    min_rating: float
    max_price: float
    min_price: float
    availability: Literal["in_stock", "any"]


# ------------------------------------------------------------
# Planning data (replaces old “plan” field completely)
# ------------------------------------------------------------

class PlannerOutput(TypedDict, total=False):
    mode: Literal["rag", "web"]
    filters: Dict[str, Any]
    intent: str


# ------------------------------------------------------------
# GraphState — schema for the entire pipeline
# ------------------------------------------------------------

class GraphState(TypedDict, total=False):

    # Input
    user_query: str

    # Router outputs
    intent: Literal[
        "search",
        "compare",
        "check_price",
        "check_availability",
        "unknown",
    ]
    constraints: Constraints
    need_live_price: bool
    safety_flag: bool

    # Planner output
    planner_output: PlannerOutput

    # Retriever outputs
    rag_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    retrieval_source: Optional[str]

    # Critic output
    merged_results: List[Dict[str, Any]]

    # Final answer
    final_answer: str

    # Debug info
    debug_log: List[str]


# ------------------------------------------------------------
# AgentState — runtime wrapper for LangGraph
# ------------------------------------------------------------

class AgentState:
    """
    Wrapper providing:
    - dict() for LangGraph
    - attribute access
    - safe defaults for missing keys
    """

    def __init__(self, **kwargs):
        self._data: Dict[str, Any] = {
            "user_query": "",
            "intent": "unknown",
            "constraints": {},
            "need_live_price": False,
            "safety_flag": False,

            # NEW unified planner field
            "planner_output": {},

            # retriever outputs
            "rag_results": [],
            "web_results": [],
            "retrieval_source": None,

            # merged critic results
            "merged_results": [],

            "final_answer": "",
            "debug_log": [],
        }

        self._data.update(kwargs)

    def dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __getattr__(self, item):
        return self._data.get(item)

    def __setattr__(self, item, value):
        if item == "_data":
            super().__setattr__(item, value)
        else:
            self._data[item] = value

    def __repr__(self):
        return f"AgentState({self._data})"


__all__ = [
    "Constraints",
    "PlannerOutput",
    "GraphState",
    "AgentState"
]
