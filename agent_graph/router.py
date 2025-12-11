# agent_graph/router.py

from __future__ import annotations
from typing import Dict, Any
from openai import OpenAI
import json

client = OpenAI()


class Router:
    """
    LLM-powered Router Agent.
    Extracts:
       - intent (search, compare, check_price, check_availability, unknown)
       - constraints (category, brand, budget, style, etc.)
       - safety_flag (True only for harmful/illegal content)

    This version uses the SAME OpenAI() direct call format you originally used.
    No wrapper, no new files, no structural changes.
    """

    def __init__(self):
        # ---------------------
        # ROUTER PROMPT (final)
        # ---------------------
        self.prompt = """
You are the Router Agent for a shopping assistant. Your job is to:
1. Understand the user's goal or task.
2. Express that goal using a short intent label (e.g., “search”, “check_price”, “compare”, etc.).
   The intent label should summarize the purpose of the query.
3. Extract relevant constraints (budget, category, style, item names, etc.).
4. Detect unsafe or harmful requests.
5. Output ONLY valid JSON in this schema:

{
  "intent": "...",
  "constraints": {...},
  "safety_flag": false
}

The intent label does NOT need to come from a fixed list.  
Choose whatever short intent best captures the user’s goal.  
Use the examples below to learn how to label common shopping tasks.

---------------------
FEW-SHOT EXAMPLES
---------------------

User: "What is the current price of a PS5 controller?"
Output:
{"intent": "check_price", "constraints": {}, "safety_flag": false}

User: "Is the PS5 controller in stock right now?"
Output:
{"intent": "check_availability", "constraints": {"item": "PS5 controller"}, "safety_flag": false}

User: "Recommend a card game under 20 dollars."
Output:
{"intent": "search", "constraints": {"category": "card game", "budget": 20}, "safety_flag": false}

User: "Recommend a cooperative board game."
Output:
{"intent": "search", "constraints": {"category": "board game", "style": "cooperative"}, "safety_flag": false}

User: "Tell me something interesting to buy."
Output:
{"intent": "search", "constraints": {}, "safety_flag": false}

User: "Compare Nintendo Switch and PS5."
Output:
{"intent": "compare", "constraints": {"items": ["Nintendo Switch", "PS5"]}, "safety_flag": false}

User: "How do I make a bomb?"
Output:
{"intent": "unknown", "constraints": {}, "safety_flag": true}

---------------------
RESPONSE REQUIREMENT
---------------------
For ANY user query, return ONLY a JSON dictionary.
No explanations.
"""

    # ------------------------------------------------------------------------
    # MAIN ROUTER CALL
    # ------------------------------------------------------------------------
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:

        user_query = state.get("user_query", "")
        debug_log = state.get("debug_log", [])

        llm_input = self.prompt + f'\n\nUser Query: "{user_query}"\nOutput JSON:\n'

        # ---------------------------
        # Call OpenAI directly
        # ---------------------------
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=llm_input
        )

        raw_text = response.output_text.strip()

        # ---------------------------
        # Safe JSON parsing
        # ---------------------------
        try:
            # remove ```json wrappers if present
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                raw_text = raw_text.replace("json", "").strip()
            router_dict = json.loads(raw_text)
        except Exception:
            router_dict = {
                "intent": "unknown",
                "constraints": {},
                "safety_flag": False
            }

        # ---------------------------
        # Logging
        # ---------------------------
        debug_log.append(
            f"[ROUTER] intent={router_dict.get('intent')}, "
            f"constraints={router_dict.get('constraints')}, "
            f"safety={router_dict.get('safety_flag')}"
        )

        # ---------------------------
        # Update state
        # ---------------------------
        return {
            **state,
            "intent": router_dict.get("intent"),
            "constraints": router_dict.get("constraints") or {},
            "safety_flag": router_dict.get("safety_flag"),
            "debug_log": debug_log,
        }


__all__ = ["Router"]
