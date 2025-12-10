"""
MCP Server Implementation using FastMCP
Supports:
- web.search
- rag.search (dummy index if dummy_products1.json exists)
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import logging

from fastmcp import FastMCP

from .tools.web_search import WebSearchTool
from .tools.rag_search import RAGSearchTool
from .utils.logger import setup_logging

logger = setup_logging()

# ============================================================
# Set up FastMCP server
# ============================================================

mcp = FastMCP(
    name="ecommerce-mcp-server",
    instructions="E-commerce product search server with optional dummy RAG index."
)

# ============================================================
# STEP 1 — Load dummy RAG dataset FIRST
# ============================================================

DUMMY_PATH = Path(__file__).resolve().parents[1] / "dummy_products1.json"

_dummy_index: List[Dict[str, Any]] = []


def load_dummy_index():
    global _dummy_index

    if not DUMMY_PATH.exists():
        logger.info("dummy_products1.json not found — using real vector store for RAG.")
        _dummy_index = []
        return

    try:
        with DUMMY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                _dummy_index = data
                logger.info(f"Loaded dummy RAG dataset with {len(_dummy_index)} items.")
            else:
                logger.warning("dummy_products1.json is not a list — ignoring.")
                _dummy_index = []
    except Exception as exc:
        logger.error(f"Failed to load dummy RAG file: {exc}")
        _dummy_index = []


# Load dummy data now
load_dummy_index()


# ============================================================
# STEP 2 — Initialize tools
# ============================================================

_web_tool = WebSearchTool()

if _dummy_index:
    logger.info("Dummy index active — skipping vector store initialization.")
    _rag_tool = None
else:
    logger.info("No dummy index — initializing real vector store RAGSearchTool.")
    _rag_tool = RAGSearchTool()


# ============================================================
# STEP 3 — Dummy RAG implementation
# ============================================================

def dummy_rag_search(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    num_results: int = 5
) -> List[Dict[str, Any]]:

    q = query.lower()
    results = []

    for item in _dummy_index:
        text = f"{item.get('title','')} {item.get('description','')}".lower()

        # simple token scoring
        score = sum(1 for token in q.split() if token in text)
        if score == 0:
            continue

        price = item.get("price")
        if max_price is not None and price is not None and price > max_price:
            continue

        if category and category.lower() not in item.get("category", "").lower():
            continue

        results.append((score, item))

    results.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in results[:num_results]]


# ============================================================
# STEP 4 — MCP Tool functions (NO **kwargs allowed)
# ============================================================

def web_search(
    query: str,
    max_price: Optional[float] = None,
    num_results: int = 5
):
    return _web_tool.search(
        query=query,
        filters=None,
        num_results=num_results
    )


def rag_search(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    num_results: int = 5
):
    if _dummy_index:
        logger.debug("Using dummy RAG search")
        return dummy_rag_search(
            query=query,
            category=category,
            max_price=max_price,
            num_results=num_results,
        )

    logger.debug("Using REAL vector store RAGSearchTool")
    return _rag_tool.search(
        query=query,
        filters={"category": category, "max_price": max_price},
        num_results=num_results,
        rerank=False,
    )


# ============================================================
# STEP 5 — Register tools
# ============================================================

mcp.tool()(web_search)
mcp.tool()(rag_search)


# ============================================================
# STEP 6 — Entry point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http", "sse"], default="stdio")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    logger.info("Starting MCP server with transport %s", args.transport)

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
