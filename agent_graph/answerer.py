import json
from urllib.parse import urlparse

class SimpleLLMWrapper:
    """Very small wrapper so .invoke(prompt) always returns a text string."""
    def __init__(self, client=None, model="gpt-4o-mini"):
        self.client = client
        self.model = model

    def invoke(self, prompt: str) -> str:
        if self.client is None:
            return prompt
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return completion.choices[0].message["content"]


class Answerer:
    """
    The Answerer post-processes the retriever output, enforces selection logic,
    and generates final paper/speech answers. This version enforces *plain text*
    for all paper answers and uses natural spoken text for speech answers.
    """
    def __init__(self, llm=None):
        self.llm = llm if llm else SimpleLLMWrapper()

    # ---------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------
    def __call__(self, state):
        debug = state.get("debug_log", [])
        debug.append("[ANSWERER] Starting answerer logic")

        # ------------------------------------------------------------
        # SAFETY OVERRIDE — short-circuit everything
        # ------------------------------------------------------------
        if state.get("safety_flag", False) is True:
            state["selected_items"] = []
            state["paper_answer"] = (
                "I’m sorry, but your question appears to contain unsafe or prohibited content. "
                "For safety reasons, I cannot provide assistance with this request."
            )
            state["speech_answer"] = "Sorry, I can’t help with that."
            debug.append("[ANSWERER] Safety override triggered — refusing to answer.")
            return state

        # ------------------------------------------------------------
        # Preprocess both rag_results and web_results for price/availability
        # ------------------------------------------------------------
        state = self._preprocess_results(state)

        intent = state.get("intent")
        if intent == "check_price":
            items = self._select_items_for_price(state)
            state = self._compose_price_answer(state, items)

        elif intent == "check_availability":
            items = self._select_items_for_availability(state)
            state = self._compose_availability_answer(state, items)

        elif intent == "search":
            items = self._select_items_for_search(state)
            state = self._compose_search_answer(state, items)

        else:
            state["selected_items"] = []
            state["paper_answer"] = (
                "I’m sorry, but I’m not sure how to help with that request."
            )
            state["speech_answer"] = (
                "I’m not sure how to help with that."
            )
            debug.append("[ANSWERER] Unknown intent, fallback answer used.")

        return state

    # ---------------------------------------------------------------
    # Preprocessing of RAG + Web results to fill price & availability
    # ---------------------------------------------------------------
    def _preprocess_results(self, state):
        rag = state.get("rag_results", [])
        web = state.get("web_results", [])

        # Mark all RAG as available
        for item in rag:
            item["availability"] = "available"

        # Process web results
        for w in web:
            snippet = w.get("snippet", "") or ""
            price = w.get("price", None)

            safe_price = self._extract_real_price_from_snippet(snippet, price)
            w["price"] = safe_price

            avail = self._infer_availability(snippet)
            w["availability"] = avail

        state["rag_results"] = rag
        state["web_results"] = web
        state["debug_log"].append(f"[ANSWERER] Preprocessing complete: {len(rag)} rag items, {len(web)} web items.")
        return state

    # ---------------------------------------------------------------
    # Extract real price
    # ---------------------------------------------------------------
    def _extract_real_price_from_snippet(self, snippet, fallback_price):
        if not snippet:
            return fallback_price

        if " to " in snippet and "$" in snippet:
            return None
        if "$" in snippet and "-" in snippet and "–" in snippet:
            return None

        prompt = f"""
Extract a *single real numeric price* from the snippet below, ignoring ranges or filter text.
If no single real price is present, return: 0

Snippet:
{snippet}

Return JSON only:
{{
  "price": number
}}
"""
        resp = self.llm.invoke(prompt)
        try:
            data = json.loads(resp)
            val = data.get("price", 0)
            if isinstance(val, (int, float)) and val > 0:
                return val
            return None
        except:
            return fallback_price

    # ---------------------------------------------------------------
    # Infer availability for web
    # ---------------------------------------------------------------
    def _infer_availability(self, snippet):
        if not snippet:
            return "available"

        s = snippet.lower()
        deny = ["out of stock", "sold out", "unavailable", "coming soon", "temporarily out of stock", "low stock"]
        for key in deny:
            if key in s:
                return "unavailable"
        return "available"

    # ---------------------------------------------------------------
    # Clean title using LLM
    # ---------------------------------------------------------------
    def _clean_title(self, raw_title):
        prompt = f"""
Clean the product title to be short and natural.

Raw title:
{raw_title}

Return only the cleaned title.
"""
        return self.llm.invoke(prompt).strip()

    # ---------------------------------------------------------------
    # RAG relevance check
    # ---------------------------------------------------------------
    def _rag_is_relevant(self, user_query, rag_title):
        prompt = f"""
User asked: "{user_query}"

Is this RAG item the *same or strongly matching* product?

Title: "{rag_title}"

Return only "yes" or "no".
"""
        resp = self.llm.invoke(prompt).strip().lower()
        return (resp == "yes")

    # ---------------------------------------------------------------
    # Short description for search items
    # ---------------------------------------------------------------
    def _make_short_description(self, title):
        prompt = f"""
Write a short, plain-text, 1–2 sentence description of this product:
"{title}"

Do NOT use markdown or special formatting.
Return only plain text.
"""
        return self.llm.invoke(prompt).strip()

    # ---------------------------------------------------------------
    # Unified schema builder
    # ---------------------------------------------------------------
    def _unify_schema(self, item, source_type):
        url = item.get("url") or item.get("product_url")
        snippet = item.get("snippet")
        category = item.get("category")
        doc_id = item.get("doc_id")
        price = item.get("price")
        avail = item.get("availability", "unknown")
        raw_title = item.get("title")
        cleaned_title = self._clean_title(raw_title)

        return {
            "title": cleaned_title,
            "raw_title": raw_title,
            "price": price if price is not None else None,
            "availability": avail,
            "description": None,
            "url": url,
            "doc_id": doc_id,
            "source_type": source_type,
            "snippet": snippet,
            "category": category
        }

    # ---------------------------------------------------------------
    # Item selection logic
    # ---------------------------------------------------------------
    def _select_items_for_price(self, state):
        web = state.get("web_results", [])
        unified = [self._unify_schema(w, "web") for w in web]

        with_price = [x for x in unified if isinstance(x["price"], (int, float))]
        without = [x for x in unified if x["price"] is None]

        with_price_sorted = sorted(with_price, key=lambda x: x["price"])

        final_list = with_price_sorted + without
        final_list = final_list[:3]

        state["debug_log"].append("[ANSWERER] Price order: " + str([x["price"] for x in final_list]))
        return final_list

    def _select_items_for_availability(self, state):
        user_query = state.get("user_query", "")
        rag = state.get("rag_results", [])
        web = state.get("web_results", [])

        relevant_rag = []
        fallback_rag = []

        for r in rag:
            if self._rag_is_relevant(user_query, r.get("title", "")):
                relevant_rag.append(self._unify_schema(r, "rag"))
            else:
                fallback_rag.append(self._unify_schema(r, "rag"))

        web_u = [self._unify_schema(w, "web") for w in web]
        web_avail = [x for x in web_u if x["availability"] == "available"]
        web_unavail = [x for x in web_u if x["availability"] == "unavailable"]

        combined = relevant_rag + web_avail + web_unavail + fallback_rag
        combined = combined[:3]
        return combined

    def _select_items_for_search(self, state):
        rag = state.get("rag_results", [])
        unified = [self._unify_schema(r, "rag") for r in rag[:3]]

        for item in unified:
            item["description"] = self._make_short_description(item["title"])

        return unified[:3]

    # ---------------------------------------------------------------
    # Answer Composition (PLAIN TEXT ONLY)
    # ---------------------------------------------------------------
    def _compose_price_answer(self, state, items):
        query = state.get("user_query", "")
        state["selected_items"] = items

        lines = []
        lines.append(f"Here are the current prices related to your request: {query}")
        lines.append("")

        for idx, it in enumerate(items, 1):
            title = it.get("title", "")
            price = it.get("price")
            price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "Price unavailable"
            avail = it.get("availability", "unknown")
            url = it.get("url", "")
            lines.append(f"{idx}. {title}")
            lines.append(f"   Price: {price_text}")
            lines.append(f"   Availability: {avail}")
            lines.append(f"   Link: {url}")
            lines.append("")

        plain_text = "\n".join(lines)
        state["paper_answer"] = plain_text

        # ---------------- Speech Answer ----------------
        first = items[0]
        speech_prompt = f"""
User asked: "{query}"

Create a SHORT spoken-style answer (1–2 sentences).
Use ONLY this item:
{json.dumps(first, indent=2)}

Rules:
- Extract store name from the hostname in the item's URL.
- Examples:
  amazon.com → Amazon
  gamestop.com → GameStop
  bestbuy.com → Best Buy
  direct.playstation.com → PlayStation Direct
  xbox.com → Xbox Official Store
- Do NOT say any URL.
- Mention price if available.
- Sound natural, like a salesperson.

Return ONLY the spoken sentence.
"""
        state["speech_answer"] = self.llm.invoke(speech_prompt)

        return state

    def _compose_availability_answer(self, state, items):
        query = state.get("user_query", "")
        state["selected_items"] = items

        lines = []
        lines.append(f"Here is what I found about availability for your request: {query}")
        lines.append("")

        for idx, it in enumerate(items, 1):
            title = it["title"]
            price = it["price"]
            price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "Price unavailable"
            avail = it.get("availability", "unknown")
            url = it.get("url", "")
            lines.append(f"{idx}. {title}")
            lines.append(f"   Availability: {avail}")
            lines.append(f"   Price: {price_text}")
            lines.append(f"   Link: {url}")
            lines.append("")

        plain_text = "\n".join(lines)
        state["paper_answer"] = plain_text

        # ---------------- Speech Answer ----------------
        first = items[0]
        speech_prompt = f"""
User asked: "{query}"

Create a SHORT spoken-style answer (1–2 sentences).
Use ONLY this item:
{json.dumps(first, indent=2)}

Rules:
- Extract store name from the hostname.
- No URLs.
- State clearly if the item appears available or unavailable.
- Sound like a real spoken sentence.

Return ONLY the spoken sentence.
"""
        state["speech_answer"] = self.llm.invoke(speech_prompt)

        return state

    def _compose_search_answer(self, state, items):
        query = state.get("user_query", "")
        state["selected_items"] = items

        lines = []
        lines.append(f"Here are some product suggestions related to your request: {query}")
        lines.append("")

        for idx, it in enumerate(items, 1):
            title = it["title"]
            desc = it.get("description", "")
            price = it["price"]
            price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "Price unavailable"
            url = it.get("url", "")
            lines.append(f"{idx}. {title}")
            lines.append(f"   Price: {price_text}")
            lines.append(f"   Description: {desc}")
            lines.append(f"   Link: {url}")
            lines.append("")

        plain_text = "\n".join(lines)
        state["paper_answer"] = plain_text

        # ---------------- Speech Answer ----------------
        first = items[0]
        speech_prompt = f"""
User asked: "{query}"

Create a SHORT spoken suggestion (1–2 sentences).
Use ONLY this item:
{json.dumps(first, indent=2)}

Rules:
- Extract store name from URL hostname.
- Do NOT mention the URL.
- Sound natural and conversational, like a salesperson.

Return ONLY the spoken sentence.
"""
        state["speech_answer"] = self.llm.invoke(speech_prompt)

        return state
