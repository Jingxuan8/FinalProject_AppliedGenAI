from __future__ import annotations

from typing import Dict, Any, List, Optional
import json
from openai import OpenAI


def _extract_title(item: Dict[str, Any]) -> Optional[str]:
    for key in ("title", "name", "product_name"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    for key in ("sku", "doc_id"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_url(item: Dict[str, Any]) -> Optional[str]:
    for key in ("url", "product_url", "link"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _extract_price(item: Dict[str, Any]) -> Optional[float]:
    price = item.get("price")
    if isinstance(price, (int, float)):
        return float(price)
    return None


def _normalize_item_for_prompt(item: Dict[str, Any], source_type: str) -> Dict[str, Any]:
    if not item:
        return {}
    return {
        "source_type": source_type,
        "title": _extract_title(item),
        "url": _extract_url(item),
        "price": _extract_price(item),
        "availability": item.get("availability"),
        "category": item.get("category"),
        "brand": item.get("brand"),
    }


def _build_citations(
    rag_items: List[Dict[str, Any]],
    web_items: List[Dict[str, Any]],
    resolved_item: Optional[Dict[str, Any]],
    max_rag: int = 2,
    max_web: int = 3,
) -> List[Dict[str, str]]:
    """Build a small, deterministic citation list.

    Each citation is a dict with:
        { "title": ..., "url": ... }
    The Answerer text will refer to them as [1], [2], ...
    """
    citations: List[Dict[str, str]] = []
    seen = set()

    def add_item(item: Dict[str, Any], fallback_title: str):
        if not item:
            return
        title = _extract_title(item) or fallback_title
        url = _extract_url(item) or ""
        key = (title, url)
        if key in seen:
            return
        seen.add(key)
        citations.append({"title": title, "url": url})

    # 1) Resolved price item first (if present)
    if resolved_item:
        add_item(resolved_item, "Lowest-price web source")

    # 2) A couple of RAG items
    for it in rag_items[:max_rag]:
        add_item(it, "Catalog item")

    # 3) A few web items
    for it in web_items[:max_web]:
        add_item(it, "Web source")

    return citations


class Answerer:
    """LLM-based Answerer/Critic.

    New behavior (compared to the old version):
    - Produces TWO answers:
        * paper_answer: report-style text with citations [1], [2], ...
        * speech_answer: short spoken answer for TTS.
    - Still uses the same model_name + OpenAI client pattern as before.
    - Keeps compatibility by also setting state["final_answer"] = speech_answer.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model_name

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        intent = state.get("intent")
        rag_results: List[Dict[str, Any]] = state.get("rag_results", []) or []
        web_results: List[Dict[str, Any]] = state.get("web_results", []) or []
        user_query: str = state.get("user_query", "") or ""
        retrieval_source = state.get("retrieval_source")
        debug_log: List[str] = state.get("debug_log", [])

        # Safety flag from Router
        if state.get("safety_flag"):
            paper_answer = (
                "I’m not able to help with this request because it was flagged as potentially unsafe. "
                "Please try asking about a different product or question."
            )
            speech_answer = (
                "I’m sorry, but I can’t help with that request because it might be unsafe. "
                "Please ask about a different product instead."
            )
            state["paper_answer"] = paper_answer
            state["speech_answer"] = speech_answer
            state["final_answer"] = speech_answer
            state["citations"] = []
            debug_log.append("[ANSWERER] Skipped LLM due to safety_flag=True.")
            state["debug_log"] = debug_log
            return state

        # ----------------------------------------
        # 1) SELECT A SMALL SET OF ITEMS (as before)
        # ----------------------------------------
        selected: List[Dict[str, Any]] = []

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
        if not selected and not rag_results and not web_results:
            paper_text = "I could not find results for this query based on either the catalog or live web data."
            speech_text = "I wasn’t able to find anything useful for that request based on the data I have."
            state["paper_answer"] = paper_text
            state["speech_answer"] = speech_text
            state["final_answer"] = speech_text
            state["citations"] = []
            debug_log.append("[ANSWERER] No results available; returned fallback messages.")
            state["debug_log"] = debug_log
            return state

        # ----------------------------------------
        # 2) GATHER EXTRA CONTEXT FROM PLANNER + RETRIEVER
        # ----------------------------------------
        planner_output: Dict[str, Any] = state.get("planner_output", {}) or {}
        constraints: Dict[str, Any] = state.get("constraints", {}) or {}

        resolved_price: Optional[float] = state.get("resolved_price")
        resolved_item: Optional[Dict[str, Any]] = state.get("resolved_price_item")

        compare_price: bool = bool(planner_output.get("compare_price"))
        compare_availability: bool = bool(planner_output.get("compare_availability"))
        use_rag: bool = bool(planner_output.get("use_rag"))
        use_web: bool = bool(planner_output.get("use_web"))

        # Normalize a compact view of items for the prompt
        rag_view = [_normalize_item_for_prompt(it, "catalog") for it in rag_results[:3]]
        web_view = [_normalize_item_for_prompt(it, "web") for it in web_results[:3]]
        resolved_view = _normalize_item_for_prompt(resolved_item, "web") if resolved_item else None
        selected_view = [_normalize_item_for_prompt(it, "selected") for it in selected]

        context = {
            "user_query": user_query,
            "intent": intent,
            "constraints": constraints,
            "retrieval_source": retrieval_source,
            "planner": {
                "use_rag": use_rag,
                "use_web": use_web,
                "compare_price": compare_price,
                "compare_availability": compare_availability,
            },
            "resolved_price": resolved_price,
            "resolved_price_item": resolved_view,
            "selected_items": selected_view,
            "rag_items": rag_view,
            "web_items": web_view,
        }

        context_json = json.dumps(context, ensure_ascii=False, indent=2)

        # ----------------------------------------
        # 3) LLM CALL — PAPER / REPORT ANSWER
        # ----------------------------------------
        paper_prompt = f"""You are the Answerer/Critic agent in a shopping assistant.

You receive:
- The user's original query.
- Summaries of products from a private catalog ("catalog items").
- Summaries of products from live web search ("web items").
- Optionally, a resolved lowest live price for a product.

Your goals:
1. Provide a clear, concise written answer suitable for a report or on-screen text.
2. Distinguish between information coming from:
   - our private catalog, and
   - live web data.
3. If a resolved lowest price is provided, highlight it as the best current live price.
4. If price comparison is requested, briefly discuss any discrepancy between catalog pricing and live web pricing,
   and explain that live web prices are more up-to-date.
5. If availability comparison is requested, comment on availability based on the web items, or say that availability is unclear.
6. Be honest about limitations; if you don't see strong matches, say so.
7. Use numeric citations [1], [2], [3] whenever you refer to a specific product or web page.
   - Assume that [1], [2], ... refer to a separate citation list shown under the answer.
   - Do NOT output the citation list itself here.

Write 1–3 short paragraphs. Do NOT include raw URLs in the text.

Here is the structured context (JSON):

{context_json}

Now write only the report-style answer text with citations."""

        paper_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": paper_prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        paper_answer = paper_response.choices[0].message.content.strip()

        # ----------------------------------------
        # 4) LLM CALL — SPOKEN / TTS ANSWER
        # ----------------------------------------
        speech_prompt = f"""You are the voice of the same shopping assistant.

You will be given the same structured context as above. Your job is to produce a short, natural spoken answer
that could be read aloud by text-to-speech.

Requirements:
- Speak as if you are talking directly to the user.
- Summarize the key recommendation(s) and any important pricing or availability notes.
- If there is a resolved lowest live price, mention it in natural language (for example, "around 45 dollars").
- DO NOT include URLs, citation markers, or bracketed references of any kind.
- Keep it brief: ideally 2–5 sentences.

Here is the structured context (JSON):

{context_json}

Now write only the spoken-style answer text, suitable for TTS."""

        speech_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": speech_prompt}],
            temperature=0.3,
            max_tokens=350,
        )
        speech_answer = speech_response.choices[0].message.content.strip()

        # ----------------------------------------
        # 5) Build citations list (non-LLM, deterministic)
        # ----------------------------------------
        citations = _build_citations(rag_results, web_results, resolved_item)

        # Update state
        state["paper_answer"] = paper_answer
        state["speech_answer"] = speech_answer
        # Backwards-compat: final_answer = speech version (for TTS)
        state["final_answer"] = speech_answer
        state["citations"] = citations

        debug_log.append("[ANSWERER] Generated paper_answer, speech_answer, and citations.")
        state["debug_log"] = debug_log

        return state


__all__ = ["Answerer"]
