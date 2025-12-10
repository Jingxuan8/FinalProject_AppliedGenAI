"""
MCP Server Implementation using FastMCP
Exposes two tools:
- web_search: Real-time web search via Serper/Brave API
- rag_search: Vector DB search over Amazon Product Dataset 2020
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import logging

from fastmcp import FastMCP

from .tools.web_search import WebSearchTool, WebSearchConfigError
from .tools.rag_search import RAGSearchTool
from .utils.logger import setup_logging

logger = setup_logging()

# ============================================================
# FastMCP Server Setup
# ============================================================

mcp = FastMCP(
    name="ecommerce-mcp-server",
    instructions="""
    E-commerce Product Search MCP Server.
    
    This server provides two tools for product search:
    
    1. web_search: Search the web for real-time product information, prices, and availability.
       Use this when the user asks about current prices, stock availability, or latest deals.
    
    2. rag_search: Search the private Amazon Product Dataset 2020 catalog (Games & Accessories).
       Use this for detailed product information, ratings, reviews, and grounded recommendations.
    
    For best results:
    - Use rag_search first for product details, ratings, and features
    - Use web_search when user needs real-time/current information
    - Combine both for price comparisons between catalog and live data
    """
)

# ============================================================
# Initialize Tools
# ============================================================

# Web search tool - requires API key
_web_tool: Optional[WebSearchTool] = None
try:
    _web_tool = WebSearchTool()
    logger.info("WebSearchTool initialized successfully")
except WebSearchConfigError as e:
    logger.warning(f"WebSearchTool not available: {e}")
    _web_tool = None

# RAG search tool - uses local vector store
_rag_tool: Optional[RAGSearchTool] = None
try:
    _rag_tool = RAGSearchTool()
    doc_count = _rag_tool.get_document_count()
    logger.info(f"RAGSearchTool initialized with {doc_count} documents")
except Exception as e:
    logger.error(f"RAGSearchTool initialization failed: {e}")
    _rag_tool = None


# ============================================================
# MCP Tool Functions
# Compatible with agent_graph/retriever.py format
# ============================================================

@mcp.tool()
def web_search(
    query: str,
    num_results: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search the web for real-time product information.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
        filters: Optional filters (max_price, min_price, brand, availability)
    
    Returns:
        List of search results with title, url, snippet, price, availability
    """
    if _web_tool is None:
        raise RuntimeError(
            "Web search is not available. Please configure SERPER_API_KEY or BRAVE_API_KEY."
        )
    
    return _web_tool.search(
        query=query,
        filters=filters,
        num_results=num_results
    )


@mcp.tool()
def rag_search(
    query: str,
    config: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    num_results: Optional[int] = None,
    rerank: Optional[bool] = None
) -> List[Dict[str, Any]]:
    """
    Search the Amazon Product Dataset 2020 (Games & Accessories) catalog.
    
    Args:
        query: Natural language search query
        config: Search configuration (alternative to direct params)
            - num_results: Number of results (1-20, default: 5)
            - rerank: Whether to apply LLM reranking (default: False)
        filters: Metadata filters
            - max_price: Maximum price in USD
            - min_price: Minimum price in USD
            - category: Product category filter
            - brand: Brand name filter
        num_results: Number of results (direct param, overrides config)
        rerank: Whether to apply LLM reranking (direct param, overrides config)
    
    Returns:
        List of products with sku, title, price, rating, brand, category, relevance_score
    """
    if _rag_tool is None:
        raise RuntimeError(
            "RAG search is not available. Please run 'python rag/build_index.py' first."
        )
    
    # Parse config - support both config dict and direct params
    config = config or {}
    final_num_results = num_results if num_results is not None else config.get("num_results", 5)
    final_rerank = rerank if rerank is not None else config.get("rerank", False)
    
    # Parse filters
    filters = filters or {}
    
    return _rag_tool.search(
        query=query,
        budget=filters.get("max_price"),
        filters={
            "max_price": filters.get("max_price"),
            "min_price": filters.get("min_price"),
            "category": filters.get("category"),
            "brand": filters.get("brand"),
        },
        num_results=final_num_results,
        rerank=final_rerank
    )


# ============================================================
# Health Check Tool
# ============================================================

@mcp.tool()
def health_check() -> Dict[str, Any]:
    """
    Check the health status of the MCP server and its tools.
    
    Returns:
        Status of web_search and rag_search tools
    """
    status = {
        "server": "running",
        "web_search": {
            "available": _web_tool is not None,
            "provider": _web_tool.provider if _web_tool else None
        },
        "rag_search": {
            "available": _rag_tool is not None,
            "document_count": _rag_tool.get_document_count() if _rag_tool else 0
        }
    }
    return status


# ============================================================
# Entry Point
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="E-commerce MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP/SSE transport")
    parser.add_argument("--port", type=int, default=8765, help="Port for HTTP/SSE transport")
    args = parser.parse_args()

    logger.info(f"Starting MCP server with transport: {args.transport}")
    
    if _web_tool:
        logger.info(f"  - web_search: enabled (provider: {_web_tool.provider})")
    else:
        logger.warning("  - web_search: DISABLED (no API key)")
    
    if _rag_tool:
        logger.info(f"  - rag_search: enabled ({_rag_tool.get_document_count()} docs)")
    else:
        logger.warning("  - rag_search: DISABLED (no vector store)")

    if args.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
