from openai import OpenAI

class Answerer:
    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

    def __call__(self, state: dict):
        intent = state.get("intent")
        rag_results = state.get("rag_results", [])
        web_results = state.get("web_results", [])
        user_query = state.get("user_query", "")
        retrieval_source = state.get("retrieval_source")

        selected = []

        # ----------------------------------------
        # RULE-BASED SELECTION
        # ----------------------------------------
        if intent == "check_price":
            # pick lowest priced item from web search
            priced = [w for w in web_results if isinstance(w.get("price"), (int, float))]
            if priced:
                best = min(priced, key=lambda x: x["price"])
                selected = [best]
            else:
                selected = web_results[:3] if web_results else []

        elif intent == "search":
            # Prefer RAG results
            if rag_results:
                # top 3 by relevance_score
                ranked = sorted(rag_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
                selected = ranked[:3]
            else:
                # fallback to web
                selected = web_results[:3]

        else:
            # unknown intent fallback
            selected = (rag_results or web_results)[:3]

        # handle no results
        if not selected:
            state["final_answer"] = "I could not find results for your query based on available data."
            return state

        # ----------------------------------------
        # LLM FORMATTING
        # ----------------------------------------
        llm_prompt = f"""
You are the Answerer agent in a retrieval-based system.

Your job is to answer the user_query ONLY using the retrieved items provided.
Do NOT hallucinate information not present in the items.

CRITICAL FORMAT RULES:
- The answer must sound like natural spoken dialogue.
- No bullet points, no numbering, no lists.
- No URLs or Markdown formatting of any kind.
- No square brackets.
- Summaries should be smooth, verbal, and under 15 seconds when read aloud.
- Treat product titles as plain speech (e.g., “a game called Tic Tac Two”).
- Pricing should be spoken naturally, like “about twelve dollars.”
- Outputs must be comfortable for TTS to read directly, without visual anchors.

user_query: "{user_query}"
intent: "{intent}"
retrieval_source: "{retrieval_source}"

retrieved_items:
{selected}

Write the final answer directly for the user in fluent spoken English.
If checking price, say the price naturally.
If search intent, give two or three spoken recommendations in a single coherent paragraph.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": llm_prompt}]
        )

        final = response.choices[0].message.content.strip()
        state["final_answer"] = final
        state["debug_log"].append("[ANSWERER] Completed final answer.")

        return state
