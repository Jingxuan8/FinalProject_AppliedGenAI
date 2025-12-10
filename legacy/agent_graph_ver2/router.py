# agent_graph/router.py

"""
Router node for LangGraph orchestration.

Responsibilities:
- Classify user's intent (search / compare / check_price / check_availability)
- Extract usable constraints (budget, category, brand, rating, etc.)
- Detect whether live pricing or availability is required
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_graph.state import GraphState, Constraints


# ------------------------------------------------------------
# Router prompt template
# ------------------------------------------------------------

ROUTER_PROMPT = """
You are the Router Agent for an e-commerce product assistant.

Your goals:
1. Identify the user's high-level intent.
2. Extract structured constraints.
3. Determine if the query requires live (real-time) price or availability info.

Possible intents:
- "search"              → user wants product recommendations
- "compare"             → user wants differences / ranking between items
- "check_price"         → user wants price info, especially CURRENT price
- "check_availability"  → user asks about stock status
- "unknown"             → cannot classify

Constraints to extract (if present):
- budget (max dollar amount)
- category (e.g., board game, dice, card game)
- brand
- min_rating
- max_price / min_price
- availability (if user says “in stock”, “available now”)

Rules:
- If user mentions “current price”, “now”, “latest price”, or “availability”, set need_live_price = true.
- Category should be a short noun phrase (“board game”, “card game”, “dice game”).
- Keep constraints minimal — only include fields actually mentioned.

Output JSON only:
{
  "intent": "...",
  "constraints": { ... },
  "need_live_price": true/false
}
"""


# ------------------------------------------------------------
# Call LLM (model-agnostic)
# ------------------------------------------------------------

def call_llm_router(model, query: str) -> Dict[str, Any]:
    """
    Calls the LLM to classify the query.

    `model` must provide:
        model.chat.completions.create(
            model=model_name,
            messages=[...]
        )

    This matches the OpenAI 1.x API and many other providers.
    """

    response = model.chat.completions.create(
        model=model.model_name,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=300,
    )

    # Router must ALWAYS return valid JSON
    content = response.choices[0].message.content

    import json
    try:
        parsed = json.loads(content)
        return parsed
    except Exception:
        # Fallback safety wrapper
        return {
            "intent": "unknown",
            "constraints": {},
            "need_live_price": False
        }


# ------------------------------------------------------------
# Router node for LangGraph
# ------------------------------------------------------------

def router_node(state: GraphState, model) -> GraphState:
    """
    LangGraph node: updates the state with routing information.
    """

    query = state["user_query"]

    parsed = call_llm_router(model, query)

    intent = parsed.get("intent", "unknown")
    constraints: Constraints = parsed.get("constraints", {}) or {}
    need_live_price = bool(parsed.get("need_live_price", False))

    debug_msg = f"[ROUTER] intent={intent}, need_live_price={need_live_price}, constraints={constraints}"

    return {
        **state,
        "intent": intent,
        "constraints": constraints,
        "need_live_price": need_live_price,
        "debug_log": state.get("debug_log", []) + [debug_msg],
    }


__all__ = ["router_node", "call_llm_router", "ROUTER_PROMPT"]
