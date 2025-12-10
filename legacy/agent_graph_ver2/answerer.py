# agent_graph/answerer.py

"""
Answerer node for LangGraph orchestration.

Responsibilities:
- Convert retrieved product results into a grounded, concise final answer.
- Include citations (doc_id for RAG results, source domain for web results).
- Avoid hallucinations: only reference retrieved fields.
- Produce a short, user-friendly summary suitable for TTS.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent_graph.state import GraphState


# ------------------------------------------------------------
# ANSWERER SYSTEM PROMPT
# ------------------------------------------------------------

ANSWERER_PROMPT = """
You are the Answerer Agent for an e-commerce product assistant.

Your rules:
1. Use ONLY the data provided in the tool results. Do NOT hallucinate.
2. All claims MUST be grounded in the retrieved products.
3. Cite sources:
   - For catalog (rag.search) products: cite as [doc_id: XXXXX].
   - For web results: cite as [source: domain].
4. Keep the answer concise (aim for < 10 seconds of speech).
5. If both catalog and web results are present, prefer catalog for:
      title, brand, rating, features
   Prefer web for:
      current price, availability
6. If no results: apologize briefly and ask the user to clarify.

Output: final answer text only (no JSON).
"""


# ------------------------------------------------------------
# Utility: Format a single product line
# ------------------------------------------------------------

def format_catalog_product(p: Dict[str, Any]) -> str:
    """
    Format a catalog (RAG) product into 1–2 short sentences.
    """
    title = p.get("title", "Unknown Product")
    price = p.get("price")
    brand = p.get("brand")
    rating = p.get("rating")
    doc_id = p.get("doc_id")

    parts = [title]
    if brand:
        parts.append(f"by {brand}")
    if price:
        parts.append(f"priced around ${price:.2f}")
    if rating:
        parts.append(f"rated {rating} stars")

    summary = ", ".join(parts)
    return f"{summary} [doc_id: {doc_id}]"


def format_web_product(p: Dict[str, Any]) -> str:
    """
    Format a web product into 1–2 short sentences.
    """
    title = p.get("title", "Unknown Product")
    price = p.get("price")
    source = p.get("source", "web")
    availability = p.get("availability")

    parts = [title]
    if price:
        parts.append(f"currently ${price:.2f}")
    if availability:
        parts.append(f"{availability}")

    summary = ", ".join(parts)
    return f"{summary} [source: {source}]"


# ------------------------------------------------------------
# LLM call for answer generation
# ------------------------------------------------------------

def call_llm_answerer(model, query: str, rag_results, web_results) -> str:
    """
    LLM decides how to combine the retrieved results into a final grounded answer.
    """

    import json

    response = model.chat.completions.create(
        model=model.model_name,
        messages=[
            {"role": "system", "content": ANSWERER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "catalog_results": rag_results,
                        "web_results": web_results,
                    },
                    indent=2
                ),
            },
        ],
        temperature=0,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


# ------------------------------------------------------------
# Answerer Node
# ------------------------------------------------------------

def answerer_node(state: GraphState, model) -> GraphState:
    """
    Produces the final natural-language answer using LLM and retrieved results.
    """

    rag = state.get("rag_results", [])
    web = state.get("web_results", [])

    # If no results at all:
    if not rag and not web:
        final = (
            "I couldn’t find matching products. "
            "Could you rephrase or include more details?"
        )
        return {
            **state,
            "final_answer": final,
            "debug_log": state.get("debug_log", []) + ["[ANSWERER] no results"],
        }

    # LLM generates the final answer
    final = call_llm_answerer(
        model,
        state["user_query"],
        rag,
        web,
    )

    debug_msg = f"[ANSWERER] produced response ({len(final)} chars)"

    return {
        **state,
        "final_answer": final,
        "debug_log": state.get("debug_log", []) + [debug_msg],
    }


__all__ = [
    "answerer_node",
    "call_llm_answerer",
    "format_catalog_product",
    "format_web_product",
    "ANSWERER_PROMPT",
]
