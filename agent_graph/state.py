# agent_graph/state.py

"""
Shared state definitions for the LangGraph orchestration.

This file defines:
- Constraints: normalized constraints extracted from the user's query.
- AgentPlan: planner output that decides which tools to call and how.
- GraphState: the full state object passed between LangGraph nodes.

All other nodes (router, planner, retriever, answerer, critic) should
import and use these types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# ------------------------------------------------------------
# Constraint structure extracted by the Router
# ------------------------------------------------------------

class Constraints(TypedDict, total=False):
    """
    Normalized constraints extracted from the user query.

    These directly map to rag_search / web_search parameters.

    - budget:  overall max budget (shorthand for max_price)
    - category: product category / type (e.g. 'Board Games')
    - brand:    preferred brand
    - min_rating: minimum star rating (1–5)
    - max_price / min_price: explicit price bounds
    - availability: used for web.search ('in_stock' or 'any')
    """
    budget: float
    category: str
    brand: str
    min_rating: float
    max_price: float
    min_price: float
    availability: Literal["in_stock", "any"]


# ------------------------------------------------------------
# Planner output structure
# ------------------------------------------------------------

class AgentPlan(TypedDict, total=False):
    """
    Planner's decision about which tools to call and why.

    - use_rag: whether to call rag.search
    - use_web: whether to call web.search
    - rerank:  whether to ask rag.search for LLM-based reranking
    - reason:  short natural-language explanation (useful for logging / UI)
    - rag_kwargs: arguments to pass into rag.search MCP call
    - web_kwargs: arguments to pass into web.search MCP call
    """
    use_rag: bool
    use_web: bool
    rerank: bool
    reason: str
    rag_kwargs: Dict[str, Any]
    web_kwargs: Dict[str, Any]


# ------------------------------------------------------------
# Main graph state
# ------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    Shared state object for the LangGraph workflow.

    Keys:
    - user_query: raw natural language query from the user.
    - intent:     high-level task type identified by Router.
    - constraints: normalized constraints extracted by Router.
    - need_live_price: whether Router believes we need current / live data.
    - plan:       Planner's decision on which tools to call.
    - rag_results: raw list of dicts returned by rag.search MCP tool.
    - web_results: raw list of dicts returned by web.search MCP tool.
    - merged_results: reconciled list produced by Critic / merger.
    - final_answer: final natural-language answer produced by Answerer.
    - debug_log: list of short debug messages for UI / logging.
    """

    # Input
    user_query: str

    # Router outputs
    intent: Literal[
        "search",             # general product search / recommend
        "compare",            # comparison of two or more products
        "check_price",        # focus on price (often "current")
        "check_availability", # in-stock / availability oriented
        "unknown",
    ]
    constraints: Constraints
    need_live_price: bool

    # Planner outputs
    plan: AgentPlan

    # Tool results
    rag_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    merged_results: List[Dict[str, Any]]

    # Final outputs
    final_answer: str

    # Optional debug / trace information
    debug_log: List[str]


__all__ = ["Constraints", "AgentPlan", "GraphState"]
