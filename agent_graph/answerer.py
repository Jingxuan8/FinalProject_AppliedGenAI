import json
from typing import Dict, Any, List
from openai import OpenAI


class Answerer:

    def __init__(self):
        self.client = OpenAI()  # Uses global API key

    # -------------------------------
    # Helper: call LLM
    # -------------------------------
    def _llm(self, messages, max_tokens=150):
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens
        )
        # IMPORTANT: attribute access, not dict
        return resp.choices[0].message.content

    # -------------------------------
    # Clean the product title via LLM
    # -------------------------------
    def _clean_title(self, raw_title: str) -> str | None:
        if not raw_title:
            return None

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are cleaning product titles for a shopping assistant. "
                    "Remove store names, marketing fluff, and trailing site names like 'Amazon.com'. "
                    "Keep only a concise, human-friendly product name. "
                    "Return ONLY the cleaned title, nothing else."
                )
            },
            {
                "role": "user",
                "content": f"Raw title: {raw_title}\nCleaned title:"
            }
        ]
        return self._llm(prompt, max_tokens=30).strip()

    # -------------------------------
    # Infer availability (LLM + hints)
    # -------------------------------
    def _infer_availability(self, title: str, snippet: str | None) -> str:
        """
        Use simple rules + LLM to infer availability.
        - If snippet has no availability hints at all -> 'available'
        - Otherwise ask LLM to classify as 'available' or 'unavailable'
        """
        snippet_text = (snippet or "").lower()

        # If snippet has zero hints -> assume available
        hints = [
            "unavailable", "out of stock", "coming soon",
            "temporarily unavailable", "sold out", "backordered", "pre-order"
        ]
        if not any(h in snippet_text for h in hints):
            return "available"

        # Otherwise ask LLM to decide
        prompt = [
            {
                "role": "system",
                "content": (
                    "Decide if a product is AVAILABLE or UNAVAILABLE based on the title and snippet. "
                    "Return EXACTLY one lowercase word: 'available' or 'unavailable'. "
                    "If the text suggests 'currently unavailable', 'out of stock', 'coming soon', "
                    "'temporarily unavailable', 'sold out', or similar, choose 'unavailable'. "
                    "If it sounds purchasable or in stock, choose 'available'."
                )
            },
            {
                "role": "user",
                "content": f"TITLE:\n{title}\n\nSNIPPET:\n{snippet or ''}\n\nAvailability?"
            }
        ]
        ans = self._llm(prompt, max_tokens=5).strip().lower()

        if "unavailable" in ans:
            return "unavailable"
        if "available" in ans:
            return "available"

        # Fallback: never mark unavailable without clear signal
        return "available"

    # -------------------------------
    # Short product description (LLM)
    # -------------------------------
    def _short_description(self, title: str | None, snippet: str | None) -> str:
        """
        Generate a short 1–2 sentence description of the *physical product*,
        grounded in the title and snippet.
        """
        title_text = title or "this product"
        snippet_text = snippet or ""

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are helping a shopping assistant describe physical products. "
                    "Write 1–2 short sentences describing what the product is and why it might be interesting. "
                    "Base your description ONLY on the product title and snippet. "
                    "DO NOT invent apps, software, or unrelated services if none are mentioned. "
                    "Assume this is a physical product being sold to consumers."
                )
            },
            {
                "role": "user",
                "content": (
                    f"TITLE: {title_text}\n\n"
                    f"SNIPPET: {snippet_text}\n\n"
                    "Short description (1–2 sentences):"
                )
            }
        ]
        return self._llm(prompt, max_tokens=60).strip()

    # -------------------------------
    # Build paper + speech answers
    # -------------------------------
    def _compose_answers(self, user_query: str, items: List[Dict[str, Any]]) -> (str, str):
        # PAPER ANSWER
        paper_lines = [f"Here are three product suggestions based on your request: '{user_query}':", ""]
        for it in items:
            title = it.get("title") or it.get("raw_title") or "Unnamed product"
            price = it.get("price")
            price_str = f"${price}" if price is not None else "price not available"
            desc = it.get("description") or "No additional details available."
            source_type = it.get("source_type") or "unknown source"
            paper_lines.append(f"- {title} ({price_str}) — {desc} [Source: {source_type}]")
        paper_answer = "\n".join(paper_lines)

        # SPEECH ANSWER (shorter)
        speech_parts = [ "Here are a few options I found." ]
        for it in items:
            title = it.get("title") or it.get("raw_title") or "a product"
            price = it.get("price")
            if price is not None:
                speech_parts.append(f"{title} is around ${price}.")
            else:
                speech_parts.append(f"{title} is another option to consider.")
        speech_answer = " ".join(speech_parts)

        return paper_answer, speech_answer

    # -------------------------------
    # MAIN CALL
    # -------------------------------
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:

        debug_log = state.setdefault("debug_log", [])

        user_query = state.get("user_query", "")
        intent = state.get("intent", "search")

        rag_results = state.get("rag_results", []) or []
        web_results = state.get("web_results", []) or []
        source_type = state.get("retrieval_source", "unknown")  # e.g. "rag", "web", "mixed"

        # -------------------------------------------------
        # 1) Choose ordering of results (web priority for price/availability)
        # -------------------------------------------------
        if intent in ("check_price", "check_availability") and web_results:
            # Web first, then rag, for PS5 controller type questions
            combined: List[Dict[str, Any]] = list(web_results) + list(rag_results)
            debug_log.append("[ANSWERER] Using web-first ordering for price/availability intent.")
        else:
            # Default: catalog (RAG) first, then web
            combined = list(rag_results) + list(web_results)
            debug_log.append("[ANSWERER] Using rag-first ordering for search/recommendation intent.")

        # Ensure at least 3 results exist (pad with empty placeholders if needed)
        while len(combined) < 3:
            combined.append({
                "title": None,
                "price": None,
                "availability": None,
                "url": None,
                "source": None,
                "doc_id": None,
                "snippet": None,
                "category": None,
                "brand": None,
            })

        top3 = combined[:3]

        # -------------------------------------------------
        # 2) Normalize & enrich each selected item
        # -------------------------------------------------
        selected_items: List[Dict[str, Any]] = []

        for item in top3:
            raw_title = item.get("title")
            snippet = item.get("snippet")

            cleaned_title = self._clean_title(raw_title) if raw_title else None
            availability = self._infer_availability(raw_title or "", snippet)
            description = self._short_description(cleaned_title or raw_title, snippet)

            selected_items.append({
                "title": cleaned_title,
                "raw_title": raw_title,
                "price": item.get("price"),
                "availability": availability,
                "description": description,
                "source_type": source_type,
                "url": item.get("url"),
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "snippet": snippet,
                "category": item.get("category"),
                "brand": item.get("brand"),
            })

        # -------------------------------------------------
        # 3) Compose paper & speech answers
        # -------------------------------------------------
        paper_answer, speech_answer = self._compose_answers(user_query, selected_items)

        # -------------------------------------------------
        # 4) Write back to state
        # -------------------------------------------------
        state["selected_items"] = selected_items
        state["paper_answer"] = paper_answer
        state["speech_answer"] = speech_answer
        state["final_answer"] = paper_answer  # backward compatibility

        debug_log.append("[ANSWERER] Completed final answer.")
        state["debug_log"] = debug_log

        return state

__all__ = ["Answerer"]