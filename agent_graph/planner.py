# agent_graph/planner.py

from __future__ import annotations
from typing import Dict, Any
from openai import OpenAI
import json

client = OpenAI()


class Planner:
    """
    LLM-powered Planner Agent.

    Determines:
    - Whether to use RAG and/or live web data
    - Whether price or availability comparisons are needed
    - What filters (max_price, category, brand) should be set

    Output schema MUST match retriever expectations:
    {
      "intent": "...",
      "use_rag": true/false,
      "use_web": true/false,
      "compare_price": true/false,
      "compare_availability": true/false,
      "filters": {
          "max_price": <number|null>,
          "category": <string|null>,
          "brand": <string|null>
      }
    }
    """

    def __init__(self):
        # Final prompt used for every planning decision
        self.prompt = """
You are the Planner Agent in a shopping assistant pipeline.

Your responsibility:
Given the Router output (intent + constraints), decide HOW the system should retrieve information.

You must output ONLY a JSON dictionary with this exact schema:

{
  "intent": "...",
  "use_rag": true/false,
  "use_web": true/false,
  "compare_price": true/false,
  "compare_availability": true/false,
  "filters": {
      "max_price": <number|null>,
      "category": <string|null>,
      "brand": <string|null>
  }
}

Do NOT add extra keys.  
Do NOT add explanations.  
Return ONLY a JSON object.

------------------------------------
BEHAVIOR GUIDANCE (Few-shot style)
------------------------------------

The Planner infers decisions from intent + constraints.  
The examples below are NOT strict rules, but demonstrations of correct behavior.

### Example A — Price check
Input:
{"intent":"check_price","constraints":{}}

Output:
{
  "intent": "check_price",
  "use_rag": true,
  "use_web": true,
  "compare_price": true,
  "compare_availability": true,
  "filters": { "max_price": null, "category": null, "brand": null }
}

### Example B — Availability check
Input:
{"intent":"check_availability","constraints":{"item":"PS5 controller"}}

Output:
{
  "intent": "check_availability",
  "use_rag": true,
  "use_web": true,
  "compare_price": false,
  "compare_availability": true,
  "filters": { "max_price": null, "category": null, "brand": null }
}

### Example C — Search with category + budget
Input:
{"intent":"search","constraints":{"category":"card game","budget":20}}

Output:
{
  "intent": "search",
  "use_rag": true,
  "use_web": false,
  "compare_price": false,
  "compare_availability": false,
  "filters": { "max_price": 20, "category": "card game", "brand": null }
}

### Example D — General search
Input:
{"intent":"search","constraints":{}}

Output:
{
  "intent": "search",
  "use_rag": true,
  "use_web": false,
  "compare_price": false,
  "compare_availability": false,
  "filters": { "max_price": null, "category": null, "brand": null }
}

### Example E — Compare products
Input:
{"intent":"compare","constraints":{"items":["Nintendo Switch","PS5"]}}

Output:
{
  "intent": "compare",
  "use_rag": true,
  "use_web": true,
  "compare_price": true,
  "compare_availability": true,
  "filters": { "max_price": null, "category": null, "brand": null }
}

### Example F — Unknown
Input:
{"intent":"unknown","constraints":{}}

Output:
{
  "intent": "unknown",
  "use_rag": false,
  "use_web": false,
  "compare_price": false,
  "compare_availability": false,
  "filters": { "max_price": null, "category": null, "brand": null }
}

------------------------------------
RESPONSE REQUIREMENTS
------------------------------------
Return ONLY valid JSON using the schema above.
"""

    # ============================================================
    # MAIN PLANNER CALL
    # ============================================================
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:

        intent = state.get("intent", "unknown")
        constraints = state.get("constraints") or {}
        debug_log = state.get("debug_log", [])

        # Compose the LLM query payload
        planner_input = {
            "intent": intent,
            "constraints": constraints,
        }

        llm_query = self.prompt + f"\n\nRouter Output:\n{json.dumps(planner_input)}\n\nReturn JSON:\n"

        # Call LLM
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=llm_query
        )

        raw_text = response.output_text.strip()

        # Clean ```json wrappers if present
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            raw_text = raw_text.replace("json", "", 1).strip()

        # Parse JSON
        try:
            planner_dict = json.loads(raw_text)
        except Exception:
            # Fail-safe defaults (unknown)
            planner_dict = {
                "intent": intent,
                "use_rag": False,
                "use_web": False,
                "compare_price": False,
                "compare_availability": False,
                "filters": {
                    "max_price": None,
                    "category": None,
                    "brand": None,
                }
            }

        # ------------------------------------------------------------
        # POST-PROCESSING SAFETY (ensure all required keys exist)
        # ------------------------------------------------------------

        # Ensure filters exist
        filters = planner_dict.get("filters", {})
        planner_dict["filters"] = {
            "max_price": filters.get("max_price"),
            "category": filters.get("category"),
            "brand": filters.get("brand"),
        }

        # Add debug log entry
        debug_log.append(f"[PLANNER] {planner_dict}")

        # Update state
        return {
            **state,
            "planner_output": planner_dict,
            "debug_log": debug_log
        }


__all__ = ["Planner"]
