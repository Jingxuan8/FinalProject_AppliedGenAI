# agent_graph/planner.py

"""
Planner node for LangGraph orchestration.

Responsibilities:
- Interpret router output (intent + constraints + need_live_price)
- Decide which tools to call this turn:
    * rag.search (default)
    * web.search (if live price or availability)
    * both (if comparing catalog vs real-time data)
- Construct argument payloads for each tool
- Decide whether LLM-based reranking should be enabled
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent_graph.state import GraphState, AgentPlan


# ------------------------------------------------------------
# Core planner rules (summarized from tool_prompts.md)
# ------------------------------------------------------------

PLANNER_SYSTEM_RULES = """
You are the Planner Agent for an e-commerce product system.

Your job:
1. Decide which tools to call: rag.search, web.search, or both.
2. Prepare arguments for each tool.
3. Follow these rules:

RULES:
- Prefer rag.search for detailed product search:
    * features, specs, ratings, product info
    * category-based search (e.g., board games, card games)
    * price filtering, brand filtering, rating filtering

- Use web.search when the user asks about:
    * "current price"
    * "now"
    * "latest"
    * "availability"
    * "in stock"

- When comparing catalog vs market price → call BOTH:
    rag.search first, then web.search.

- Enable rag.search rerank=True only when user wants:
    * "best"
    * "most relevant"
    * "top recommendation"
    * "optimal"
    * "ranked"

- Do NOT hallucinate fields. Only include arguments the tools accept.

OUTPUT JSON ONLY:
{
  "use_rag": true/false,
  "use_web": true/false,
  "rerank": true/false,
  "rag_kwargs": { ... },
  "web_kwargs": { ... },
  "reason": "short human explanation"
}
"""


# ------------------------------------------------------------
# LLM call wrapper
# ------------------------------------------------------------

def call_llm_planner(model, router_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asks the LLM to produce a tool-use plan.
    `router_info` includes intent, constraints, need_live_price.
    """

    import json

    response = model.chat.completions.create(
        model=model.model_name,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_RULES},
            {"role": "user", "content": json.dumps(router_info, indent=2)},
        ],
        temperature=0,
        max_tokens=400,
    )

    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)
        return parsed
    except Exception:
        # Fail-safe
        return {
            "use_rag": True,
            "use_web": router_info.get("need_live_price", False),
            "rerank": False,
            "rag_kwargs": {},
            "web_kwargs": {},
            "reason": "fallback plan",
        }


# ------------------------------------------------------------
# Planner node
# ------------------------------------------------------------

def planner_node(state: GraphState, model) -> GraphState:
    """
    Takes router output and decides tool usage.
    """

    router_info = {
        "intent": state["intent"],
        "constraints": state["constraints"],
        "need_live_price": state["need_live_price"],
        "user_query": state["user_query"],
    }

    plan_dict = call_llm_planner(model, router_info)

    # Ensure missing fields don't break the system
    plan: AgentPlan = {
        "use_rag": bool(plan_dict.get("use_rag", True)),
        "use_web": bool(plan_dict.get("use_web", False)),
        "rerank": bool(plan_dict.get("rerank", False)),
        "rag_kwargs": plan_dict.get("rag_kwargs", {}) or {},
        "web_kwargs": plan_dict.get("web_kwargs", {}) or {},
        "reason": plan_dict.get("reason", "no reason provided"),
    }

    debug_msg = f"[PLANNER] {plan}"

    return {
        **state,
        "plan": plan,
        "debug_log": state.get("debug_log", []) + [debug_msg],
    }


__all__ = ["planner_node", "call_llm_planner", "PLANNER_SYSTEM_RULES"]
