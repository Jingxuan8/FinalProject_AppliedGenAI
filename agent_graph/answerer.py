import json
from typing import Dict, Any
from openai import OpenAI


# =============================================================================
# SIMPLE LLM WRAPPER — to unify prompt interface (.invoke(prompt))
# =============================================================================
class SimpleLLMWrapper:
    """
    Provides .invoke(prompt) → text, matching what Answerer expects.
    Uses OpenAI ChatCompletion behind the scenes.
    """

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def invoke(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()


# =============================================================================
# ANSWERER — FULL VERSION WITH PREPROCESSING + SELECTION + COMPOSITION
# =============================================================================
class Answerer:
    def __init__(self, llm=None):
        self.llm = llm or SimpleLLMWrapper()

    # -------------------------------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------------------------------
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("safety_flag", False) is True:
            # Override everything and return refusal responses
            state["selected_items"] = []
            state["paper_answer"] = (
                "I’m sorry, but your question appears to contain unsafe or prohibited content. "
                "For safety reasons, I cannot provide assistance with this request."
            )
            state["speech_answer"] = (
                "Sorry, I can’t help with that."
            )
            state["debug_log"].append("[ANSWERER] Safety override triggered — refusing to answer.")
            return state

        debug = state["debug_log"]
        intent = state.get("intent")

        # Preprocessing
        state = self._preprocess_results(state)

        # Selection layer
        if intent == "check_price":
            state = self._select_items_for_price(state)
            state = self._compose_price_answer(state)

        elif intent == "check_availability":
            state = self._select_items_for_availability(state)
            state = self._compose_availability_answer(state)

        elif intent == "search":
            state = self._select_items_for_search(state)
            state = self._compose_search_answer(state)

        debug.append("[ANSWERER] Finished (full pipeline).")
        return state

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------
    def _preprocess_results(self, state):
        rag = state.get("rag_results", [])
        web = state.get("web_results", [])
        debug = state["debug_log"]

        # ----- RAG ITEMS -----
        processed_rag = []
        for it in rag:
            new_it = dict(it)
            new_it["availability"] = "available"
            processed_rag.append(new_it)

        # ----- WEB ITEMS -----
        processed_web = []
        for it in web:
            new_it = dict(it)
            snippet = new_it.get("snippet") or ""

            # Extract real price
            price = self._extract_real_price_from_snippet(snippet)
            new_it["price"] = price

            # Infer availability
            avail = self._infer_availability_web(snippet)
            new_it["availability"] = avail

            processed_web.append(new_it)

        state["rag_results"] = processed_rag
        state["web_results"] = processed_web

        debug.append(f"[ANSWERER] Preprocessing complete: "
                     f"{len(processed_rag)} rag items, {len(processed_web)} web items.")
        return state

    # -------------------------------------------------------------------------
    # UNIFIED ITEM SCHEMA
    # -------------------------------------------------------------------------
    def _build_unified_item(self, item, source_type, cleaned_title=None, description=None):
        return {
            "title": cleaned_title or item.get("title"),
            "raw_title": item.get("title"),

            "price": item.get("price"),
            "availability": item.get("availability"),
            "description": description,

            "url": item.get("url") or item.get("product_url"),
            "doc_id": item.get("doc_id"),
            "source_type": source_type,

            "snippet": item.get("snippet"),
            "category": item.get("category"),
        }

    # -------------------------------------------------------------------------
    # TITLE CLEANING
    # -------------------------------------------------------------------------
    def _clean_title(self, raw_title: str):
        prompt = f"""
        Clean this product title so it contains only the essential product name.
        Do NOT include store names, marketing phrases, or URLs.
        Return ONLY the cleaned title.

        Title:
        {raw_title}
        """
        resp = self.llm.invoke(prompt).strip()
        return resp if resp else raw_title

    # -------------------------------------------------------------------------
    # RAG RELEVANCE
    # -------------------------------------------------------------------------
    def _is_relevant_rag_item(self, user_query, rag_title):
        prompt = f"""
        Determine whether this catalog title refers to the SAME real-world item
        that the user is asking about.

        Respond with ONLY "yes" or "no".

        User query: "{user_query}"
        Catalog title: "{rag_title}"
        """
        resp = self.llm.invoke(prompt).lower()
        return "yes" in resp

    # -------------------------------------------------------------------------
    # SHORT DESCRIPTION
    # -------------------------------------------------------------------------
    def _generate_short_description(self, item):
        title = item.get("title") or ""
        category = item.get("category") or ""

        prompt = f"""
        Write a brief 1–2 sentence description of this product using ONLY its title
        and category. Do not invent features.

        Title: {title}
        Category: {category}
        """
        return self.llm.invoke(prompt).strip()

    # -------------------------------------------------------------------------
    # PRICE EXTRACTION
    # -------------------------------------------------------------------------
    def _extract_real_price_from_snippet(self, snippet):
        prompt = f"""
        Extract the REAL item price from the snippet.

        Return JSON: {{"price": number or null}}

        Reject:
        - ranges ($1–$1250)
        - filters ($15 to $25)
        - discount text ("Save $20")

        Accept:
        - single product prices like "$49.99", "From $29.99", "Now $19.99"

        Snippet:
        {snippet}

        Return ONLY the JSON.
        """
        resp = self.llm.invoke(prompt)
        try:
            data = json.loads(resp)
            return data.get("price")
        except:
            return None

    # -------------------------------------------------------------------------
    # AVAILABILITY
    # -------------------------------------------------------------------------
    def _infer_availability_web(self, snippet):
        s = snippet.lower()
        if any(k in s for k in ["out of stock", "unavailable", "sold out", "coming soon", "temporarily"]):
            return "unavailable"
        return "available"

    # =============================================================================
    # SELECTION LAYERS
    # =============================================================================

    # -------------------------------------------------------------------------
    # PRICE
    # -------------------------------------------------------------------------
    def _select_items_for_price(self, state):
        debug = state["debug_log"]
        web_items = state.get("web_results", [])

        processed = []
        for it in web_items:
            cleaned = self._clean_title(it.get("title", ""))
            unified = self._build_unified_item(it, "web", cleaned_title=cleaned)
            processed.append(unified)

        def sort_key(x):
            p = x.get("price")
            return (0, p) if p is not None else (1, float("inf"))

        sorted_items = sorted(processed, key=sort_key)
        debug.append(f"[ANSWERER] Price order: {[i['price'] for i in sorted_items]}")

        selected = sorted_items[:3]

        while len(selected) < 3:
            selected.append({
                "title": None, "raw_title": None,
                "price": None, "availability": "unknown",
                "description": None, "url": None,
                "doc_id": None, "source_type": "web",
                "snippet": None, "category": None
            })

        state["selected_items"] = selected
        return state

    # -------------------------------------------------------------------------
    # AVAILABILITY
    # -------------------------------------------------------------------------
    def _select_items_for_availability(self, state):
        user_query = state["user_query"]
        rag = state.get("rag_results", [])
        web = state.get("web_results", [])

        # 1. Relevant rag
        relevant_rag = []
        for it in rag:
            if self._is_relevant_rag_item(user_query, it.get("title", "")):
                unified = self._build_unified_item(it, "rag", cleaned_title=it.get("title"))
                relevant_rag.append(unified)

        # 2. Web items
        processed_web = []
        for it in web:
            cleaned = self._clean_title(it.get("title", ""))
            unified = self._build_unified_item(it, "web", cleaned_title=cleaned)
            processed_web.append(unified)

        web_available = [w for w in processed_web if w.get("availability") == "available"]
        web_unavailable = [w for w in processed_web if w.get("availability") == "unavailable"]

        ordered = relevant_rag + web_available + web_unavailable
        selected = ordered[:3]

        while len(selected) < 3:
            selected.append({
                "title": None, "raw_title": None,
                "price": None, "availability": "unknown",
                "description": None, "url": None,
                "doc_id": None, "source_type": "mixed",
                "snippet": None, "category": None
            })

        state["selected_items"] = selected
        return state

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------
    def _select_items_for_search(self, state):
        rag = state.get("rag_results", [])
        selected = []

        for it in rag[:3]:
            desc = self._generate_short_description(it)
            unified = self._build_unified_item(it, "rag", cleaned_title=it.get("title"), description=desc)
            selected.append(unified)

        while len(selected) < 3:
            selected.append({
                "title": None, "raw_title": None,
                "price": None, "availability": "unknown",
                "description": None, "url": None,
                "doc_id": None, "source_type": "rag",
                "snippet": None, "category": None
            })

        state["selected_items"] = selected
        return state

    # =============================================================================
    # ANSWER COMPOSITION LAYER
    # =============================================================================

    # -------------------------------------------------------------------------
    # PRICE → paper + speech
    # -------------------------------------------------------------------------
    def _compose_price_answer(self, state):
        items = state["selected_items"]
        query = state["user_query"]

        paper_prompt = f"""
        You are writing a simple, clean product recommendation response in plain text.

        The user asked: "{query}"

        Write:
        1) One short introductory sentence.
        2) A numbered list (1., 2., 3.) of the three selected items.
        3) Each item should be written as 1–2 natural sentences that clearly include:
           - the product title,
           - its price (or the phrase "price unavailable"),
           - its availability status,
           - and a brief helpful description summarizing what the item is.

        RULES:
        - Absolutely no markdown (no *, no -, no bold, no italics).
        - Use plain text only.
        - Do not include URLs.
        - Do not mention JSON fields.
        - Do not invent features not present in the item data.
        - Keep it simple and readable.

        Here are three EXAMPLES of the desired style (DO NOT COPY EXACT WORDING, just follow the tone):

        Example A:
        1. The Xbox Wireless Controller ($49.99, available) is a versatile gamepad that works with Xbox consoles and PCs, offering responsive controls for casual and competitive play.
        2. The PowerA Wired Controller ($24.99, available) provides an affordable option with solid build quality, ideal for everyday gaming.
        3. The Elite Series 2 Controller ($179.99, unavailable) is a premium upgrade with swappable components for players seeking advanced customization.

        Example B:
        1. The Settlers of Catan Board Game ($44.99, available) is a classic strategy title where players trade resources and build settlements.
        2. The Ticket to Ride Board Game ($37.99, available) offers easy-to-learn rules and relaxing map-building gameplay.
        3. Pandemic ($39.99, unavailable) is a cooperative game where players work together to stop global outbreaks.

        Example C:
        1. The Bicycle Standard Playing Cards ($3.99, available) are durable everyday cards great for family games.
        2. The Uno Card Game ($6.99, available) is a fast, colorful party game that’s simple for all ages.
        3. The Exploding Kittens Card Game ($19.99, unavailable) is a humorous strategy game with quick rounds.

        Now generate your answer using the three selected items below.

        Items to summarize:
        {json.dumps(items, indent=2)}

        Return only the final plain-text answer.
        """
        state["paper_answer"] = self.llm.invoke(paper_prompt)

        # Speech answer uses ONLY the first item.
        first = items[0]

        speech_prompt = f"""
        User asked: "{query}"

        You will produce a SHORT, natural spoken-style answer (1–2 sentences max).

        Use ONLY the first selected item shown below:
        {json.dumps(first, indent=2)}

        You MUST:
        - Refer to the store by interpreting the URL hostname.
        - DO NOT include the URL itself.
        - Use natural store names. Examples:
          amazon.com → Amazon
          gamestop.com → GameStop
          bestbuy.com → Best Buy
          walmart.com → Walmart
          direct.playstation.com → PlayStation Direct
          xbox.com → Xbox Official Store
        - Mention price if available.
        - Mention availability if relevant to the question.
        - Sound like a salesperson speaking casually, not a report.
        - Do NOT mention "item one", "JSON", "link", or formatting details.

        Return ONLY the spoken sentence(s).
        """
        state["speech_answer"] = self.llm.invoke(speech_prompt)

        return state

    # -------------------------------------------------------------------------
    # AVAILABILITY → paper + speech
    # -------------------------------------------------------------------------
    def _compose_availability_answer(self, state):
        items = state["selected_items"]
        query = state["user_query"]

        paper_prompt = f"""
        You are writing a simple, clean product recommendation response in plain text.

        The user asked: "{query}"

        Write:
        1) One short introductory sentence.
        2) A numbered list (1., 2., 3.) of the three selected items.
        3) Each item should be written as 1–2 natural sentences that clearly include:
           - the product title,
           - its price (or the phrase "price unavailable"),
           - its availability status,
           - and a brief helpful description summarizing what the item is.

        RULES:
        - Absolutely no markdown (no *, no -, no bold, no italics).
        - Use plain text only.
        - Do not include URLs.
        - Do not mention JSON fields.
        - Do not invent features not present in the item data.
        - Keep it simple and readable.

        Here are three EXAMPLES of the desired style (DO NOT COPY EXACT WORDING, just follow the tone):

        Example A:
        1. The Xbox Wireless Controller ($49.99, available) is a versatile gamepad that works with Xbox consoles and PCs, offering responsive controls for casual and competitive play.
        2. The PowerA Wired Controller ($24.99, available) provides an affordable option with solid build quality, ideal for everyday gaming.
        3. The Elite Series 2 Controller ($179.99, unavailable) is a premium upgrade with swappable components for players seeking advanced customization.

        Example B:
        1. The Settlers of Catan Board Game ($44.99, available) is a classic strategy title where players trade resources and build settlements.
        2. The Ticket to Ride Board Game ($37.99, available) offers easy-to-learn rules and relaxing map-building gameplay.
        3. Pandemic ($39.99, unavailable) is a cooperative game where players work together to stop global outbreaks.

        Example C:
        1. The Bicycle Standard Playing Cards ($3.99, available) are durable everyday cards great for family games.
        2. The Uno Card Game ($6.99, available) is a fast, colorful party game that’s simple for all ages.
        3. The Exploding Kittens Card Game ($19.99, unavailable) is a humorous strategy game with quick rounds.

        Now generate your answer using the three selected items below.

        Items to summarize:
        {json.dumps(items, indent=2)}

        Return only the final plain-text answer.
        """
        state["paper_answer"] = self.llm.invoke(paper_prompt)

        first = items[0]
        speech_prompt = f"""
        User asked: "{query}"

        You will produce a SHORT, natural spoken-style answer (1–2 sentences max).

        Use ONLY the first selected item shown below:
        {json.dumps(first, indent=2)}

        You MUST:
        - Refer to the store by interpreting the URL hostname.
        - DO NOT include the URL itself.
        - Use natural store names. Examples:
          amazon.com → Amazon
          gamestop.com → GameStop
          bestbuy.com → Best Buy
          walmart.com → Walmart
          direct.playstation.com → PlayStation Direct
          xbox.com → Xbox Official Store
        - Mention price if available.
        - Mention availability if relevant to the question.
        - Sound like a salesperson speaking casually, not a report.
        - Do NOT mention "item one", "JSON", "link", or formatting details.

        Return ONLY the spoken sentence(s).
        """
        state["speech_answer"] = self.llm.invoke(speech_prompt)

        return state

    # -------------------------------------------------------------------------
    # SEARCH → paper + speech
    # -------------------------------------------------------------------------
    def _compose_search_answer(self, state):
        items = state["selected_items"]
        query = state["user_query"]

        paper_prompt = f"""
        You are writing a simple, clean product recommendation response in plain text.

        The user asked: "{query}"

        Write:
        1) One short introductory sentence.
        2) A numbered list (1., 2., 3.) of the three selected items.
        3) Each item should be written as 1–2 natural sentences that clearly include:
           - the product title,
           - its price (or the phrase "price unavailable"),
           - its availability status,
           - and a brief helpful description summarizing what the item is.

        RULES:
        - Absolutely no markdown (no *, no -, no bold, no italics).
        - Use plain text only.
        - Do not include URLs.
        - Do not mention JSON fields.
        - Do not invent features not present in the item data.
        - Keep it simple and readable.

        Here are three EXAMPLES of the desired style (DO NOT COPY EXACT WORDING, just follow the tone):

        Example A:
        1. The Xbox Wireless Controller ($49.99, available) is a versatile gamepad that works with Xbox consoles and PCs, offering responsive controls for casual and competitive play.
        2. The PowerA Wired Controller ($24.99, available) provides an affordable option with solid build quality, ideal for everyday gaming.
        3. The Elite Series 2 Controller ($179.99, unavailable) is a premium upgrade with swappable components for players seeking advanced customization.

        Example B:
        1. The Settlers of Catan Board Game ($44.99, available) is a classic strategy title where players trade resources and build settlements.
        2. The Ticket to Ride Board Game ($37.99, available) offers easy-to-learn rules and relaxing map-building gameplay.
        3. Pandemic ($39.99, unavailable) is a cooperative game where players work together to stop global outbreaks.

        Example C:
        1. The Bicycle Standard Playing Cards ($3.99, available) are durable everyday cards great for family games.
        2. The Uno Card Game ($6.99, available) is a fast, colorful party game that’s simple for all ages.
        3. The Exploding Kittens Card Game ($19.99, unavailable) is a humorous strategy game with quick rounds.

        Now generate your answer using the three selected items below.

        Items to summarize:
        {json.dumps(items, indent=2)}

        Return only the final plain-text answer.
        """
        state["paper_answer"] = self.llm.invoke(paper_prompt)

        first = items[0]
        speech_prompt = f"""
        User asked: "{query}"

        You will produce a SHORT, natural spoken-style answer (1–2 sentences max).

        Use ONLY the first selected item shown below:
        {json.dumps(first, indent=2)}

        You MUST:
        - Refer to the store by interpreting the URL hostname.
        - DO NOT include the URL itself.
        - Use natural store names. Examples:
          amazon.com → Amazon
          gamestop.com → GameStop
          bestbuy.com → Best Buy
          walmart.com → Walmart
          direct.playstation.com → PlayStation Direct
          xbox.com → Xbox Official Store
        - Mention price if available.
        - Mention availability if relevant to the question.
        - Sound like a salesperson speaking casually, not a report.
        - Do NOT mention "item one", "JSON", "link", or formatting details.

        Return ONLY the spoken sentence(s).
        """
        state["speech_answer"] = self.llm.invoke(speech_prompt)

        return state
