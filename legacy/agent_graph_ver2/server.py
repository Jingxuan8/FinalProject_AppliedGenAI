"""
MCP Server Implementation using FastMCP
Exposes web.search and rag.search tools
"""

from typing import Optional
from fastmcp import FastMCP

from .tools.web_search import WebSearchTool
from .tools.rag_search import RAGSearchTool
from .utils.logger import setup_logging

logger = setup_logging()

# Initialize FastMCP server
mcp = FastMCP(
    name="ecommerce-mcp-server",
    instructions="""
    E-commerce Product Search MCP Server.
    
    This server provides two tools for product search:
    
    1. web_search: Search the web for real-time product information, prices, and availability.
    Use this when the user asks about current prices, stock availability, or latest deals.
    
    2. rag_search: Search the private Amazon Product Dataset 2020 catalog.
    Use this for detailed product information, ratings, reviews, and grounded recommendations.
    
    For best results:
    - Use rag_search first for product details, ratings, and features
    - Use web_search when user needs real-time/current information
    - Combine both for price comparisons between catalog and live data
    """
)

# Initialize tool instances
_web_search_tool = WebSearchTool()
_rag_search_tool = RAGSearchTool()


# ============================================================
# Core Tool Functions (callable directly)
# ============================================================

def web_search(
    query: str,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    brand: Optional[str] = None,
    availability: Optional[str] = None,
    num_results: int = 5
) -> list[dict]:
    """
    Search the web for product information, prices, and availability.
    
    Use this for real-time pricing, current availability checks, or when
    the user asks about 'current', 'now', or 'latest' information.
    
    Args:
        query: Search query for finding products (e.g., 'board game cooperative family')
        max_price: Maximum price in USD
        min_price: Minimum price in USD
        brand: Filter by brand name
        availability: Filter by availability ('in_stock' or 'any')
        num_results: Number of results to return (1-10, default: 5)
    
    Returns:
        List of search results with title, url, snippet, price, availability, source
    """
    filters = {}
    if max_price is not None:
        filters["max_price"] = max_price
    if min_price is not None:
        filters["min_price"] = min_price
    if brand:
        filters["brand"] = brand
    if availability:
        filters["availability"] = availability
    
    results = _web_search_tool.search(
        query=query,
        filters=filters if filters else None,
        num_results=min(max(num_results, 1), 10)
    )
    
    return results


def rag_search(
    query: str,
    budget: Optional[float] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    min_price: Optional[float] = None,
    num_results: int = 5,
    rerank: bool = False
) -> list[dict]:
    """
    Search the private Amazon Product Dataset 2020 catalog for product recommendations.
    
    Use this for detailed product information, ratings, reviews, and grounded recommendations.
    Each result includes a doc_id for citation purposes.
    
    Args:
        query: Natural language query describing the product you're looking for
        budget: Maximum budget in USD (shorthand for max_price)
        category: Product category (e.g., 'Board Games', 'Card Games')
        brand: Filter by brand name
        min_rating: Minimum star rating (1-5)
        max_price: Maximum price in USD
        min_price: Minimum price in USD
        num_results: Number of results to return (1-20, default: 5)
        rerank: Whether to apply LLM-based reranking for better relevance
    
    Returns:
        List of products with sku, doc_id (for citations), title, price, rating, brand, 
        category, features, ingredients, relevance_score
    """
    filters = {}
    if category:
        filters["category"] = category
    if brand:
        filters["brand"] = brand
    if min_rating is not None:
        filters["min_rating"] = min_rating
    if max_price is not None:
        filters["max_price"] = max_price
    if min_price is not None:
        filters["min_price"] = min_price
    
    results = _rag_search_tool.search(
        query=query,
        budget=budget,
        filters=filters if filters else None,
        num_results=min(max(num_results, 1), 20),
        rerank=rerank
    )
    
    return results


def get_cache_stats() -> dict:
    """
    Get statistics about the search caches.
    
    Returns:
        Cache statistics including hit rates and entry counts
    """
    from .utils.cache import web_search_cache, rag_search_cache
    
    return {
        "web_search_cache": web_search_cache.get_stats(),
        "rag_search_cache": rag_search_cache.get_stats()
    }


def get_tool_logs(limit: int = 10) -> list[dict]:
    """
    Get recent tool call logs for auditing.
    
    Args:
        limit: Maximum number of log entries to return (default: 10)
    
    Returns:
        List of recent tool call log entries
    """
    from .utils.logger import tool_logger
    
    return tool_logger.get_recent_logs(limit=limit)


# ============================================================
# Register tools with FastMCP
# ============================================================

# Register the functions as MCP tools
mcp.tool()(web_search)
mcp.tool()(rag_search)
mcp.tool()(get_cache_stats)
mcp.tool()(get_tool_logs)


# ============================================================
# Server entry point
# ============================================================

def main():
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server for E-commerce Product Search")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse","http"],
        default="stdio",
        help="Transport mode (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="SSE server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="SSE server port (default: 8765)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting MCP server with {args.transport} transport")

    if args.transport == "stdio":
        mcp.run()

    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)

    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
