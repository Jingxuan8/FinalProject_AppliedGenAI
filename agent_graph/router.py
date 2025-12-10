# agent_graph/router.py

"""
Router Agent (LLM-powered) for Agentic Orchestration.
Extracts:
- intent (search / compare / check_price / check_availability / unknown)
- constraints (budget, category, brand, material, rating, etc.)
- need_live_price (true if real-time info is needed)
- safety_flag (true if unsafe or prohibited)
"""

from __future__ import annotations
from typing import Dict, Any
from openai import OpenAI


# ------------------------------------------------------------
# ROUTER SYSTEM PROMPT
# ------------------------------------------------------------

ROUTER_PROMPT = """
You are the Router Agent in an agentic e-commerce assistant.

Your job:
1. Identify the user's INTENT.
2. Extract CONSTRAINTS relevant to product filtering.
3. Determine if LIVE DATA is required (current price, “now”, availability).
4. Check for SAFETY issues.

INTENT CATEGORIES:
- "search"               → product suggestions / recommendations
- "compare"              → comparing items / pros/cons
- "check_price"          → price inquiries, especially current/latest price
- "check_availability"   → stock status or availability
- "unknown"              → fallback

Constraints to extract ONLY if present:
- category
- brand
- material
- budget
- max_price / min_price
- rating
- features
- audience
- style

Rules:
- If query contains “current price”, "now", “latest price”, “in stock”, “available now”
  → need_live_price = true.
- Otherwise need_live_price = false.
- safety_flag = true only if user asks for harmful, violent, illegal, or unsafe content.

Respond ONLY with:

{
  "intent": "...",
  "constraints": {...},
  "need_live_price": true/false,
  "safety_flag": true/false
}
"""


# ------------------------------------------------------------
# ROUTER CLASS
# ------------------------------------------------------------

class Router:
    """
    Router node for LangGraph.
    Uses OpenAI gpt-4o-mini to classify user intent and extract constraints.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model_name = model_name

    def _call_llm(self, query: str) -> Dict[str, Any]:
        """Send query to LLM and safely parse JSON."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=350,
        )

        content = response.choices[0].message.content

        import json
        try:
            return json.loads(content)
        except Exception:
            return {
                "intent": "unknown",
                "constraints": {},
                "need_live_price": False,
                "safety_flag": False,
            }

    def __call__(self, state: dict):
        """Stateless LangGraph node → takes a dict, returns a dict."""
        query = state.get("user_query", "")

        parsed = self._call_llm(query)

        intent = parsed.get("intent", "unknown")
        constraints = parsed.get("constraints", {}) or {}
        need_live_price = bool(parsed.get("need_live_price", False))
        safety_flag = bool(parsed.get("safety_flag", False))

        debug_msg = f"[ROUTER] intent={intent}, constraints={constraints}, live_price={need_live_price}, safety={safety_flag}"

        # CRITICAL: MERGE WITH PRIOR STATE (do NOT overwrite)
        return {
            **state,
            "intent": intent,
            "constraints": constraints,
            "need_live_price": need_live_price,
            "safety_flag": safety_flag,
            "debug_log": state.get("debug_log", []) + [debug_msg],
        }


__all__ = ["Router"]
