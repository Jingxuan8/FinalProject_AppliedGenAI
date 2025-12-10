"""
RAG Search Tool for MCP Server
Queries the local vector database (Amazon Product Dataset 2020)
Integrates with Person A's GamesRAG implementation
"""

import sys
import time
import json
from typing import Any, Optional
from pathlib import Path

# Add project root to path to import Person A's rag module
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # mcp_server/tools -> project root
sys.path.insert(0, str(PROJECT_ROOT))

from ..config import VECTOR_STORE_PATH, COLLECTION_NAME
from ..utils.cache import rag_search_cache
from ..utils.logger import tool_logger, setup_logging

# Import Person A's RAG implementation
from rag.rag_search import GamesRAG, ProductResult

logger = setup_logging()


class RAGSearchTool:
    """
    RAG Search Tool implementation.
    
    Wraps Person A's GamesRAG for MCP server integration.
    
    Features:
    - Hybrid retrieval (vector similarity + metadata filters)
    - TTL-based caching
    - Full logging for auditing
    """
    
    def __init__(
        self,
        vector_store_path: str = VECTOR_STORE_PATH,
        collection_name: str = COLLECTION_NAME
    ):
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        
        # Initialize Person A's GamesRAG
        self._rag: Optional[GamesRAG] = None
        self._init_rag()
    
    def _init_rag(self) -> None:
        """Initialize the GamesRAG instance."""
        try:
            self._rag = GamesRAG(
                vector_dir=self.vector_store_path,
                collection_name=self.collection_name
            )
            doc_count = self._rag.collection.count()
            logger.info(f"Connected to vector store with {doc_count} documents")
        except Exception as e:
            logger.error(f"Failed to initialize GamesRAG: {e}")
            self._rag = None
    
    def search(
        self,
        query: str,
        budget: Optional[float] = None,
        filters: Optional[dict] = None,
        num_results: int = 5,
        rerank: bool = False
    ) -> list[dict]:
        """
        Execute RAG search against the product catalog.
        
        Args:
            query: Natural language search query
            budget: Maximum price (shorthand for max_price filter)
            filters: Metadata filters (category, brand, min_rating, etc.)
            num_results: Number of results to return (1-20)
            rerank: Whether to apply LLM-based reranking (reserved for future use)
            
        Returns:
            List of matching products with citations
        """
        start_time = time.time()
        filters = filters or {}
        num_results = min(max(num_results, 1), 20)
        
        # Merge budget into filters
        if budget is not None:
            filters["budget"] = min(budget, filters.get("budget", float("inf")))
        
        # Also handle max_price as budget
        if filters.get("max_price") is not None:
            filters["budget"] = min(
                filters.get("max_price"),
                filters.get("budget", float("inf"))
            )
        
        # Check cache
        cache_key_data = {"query": query, "filters": filters, "num_results": num_results, "rerank": rerank}
        cached_result = rag_search_cache.get(json.dumps(cache_key_data, sort_keys=True))
        
        if cached_result is not None:
            duration_ms = (time.time() - start_time) * 1000
            tool_logger.log_tool_call(
                tool_name="rag.search",
                request_payload={"query": query, "budget": budget, "filters": filters, "num_results": num_results},
                response_data=cached_result,
                duration_ms=duration_ms,
                success=True,
                cache_hit=True
            )
            return cached_result
        
        try:
            if self._rag is None:
                raise RuntimeError("RAG not initialized. Please run build_index.py first.")
            
            # Map filters to GamesRAG format
            rag_budget = filters.get("budget")
            rag_min_price = filters.get("min_price")
            rag_brand = filters.get("brand")
            rag_category_contains = filters.get("category")
            
            # Execute search using Person A's GamesRAG
            results: list[ProductResult] = self._rag.rag_search(
                query=query,
                top_k=num_results,
                budget=rag_budget,
                min_price=rag_min_price,
                brand=rag_brand,
                category_contains=rag_category_contains
            )
            
            # Convert ProductResult to MCP output format
            result_dicts = []
            for r in results:
                d = r.to_dict()
                result_dicts.append({
                    "sku": d.get("id", d.get("doc_id", "")),
                    "doc_id": d.get("doc_id", d.get("id", "")),
                    "title": d.get("title", ""),
                    "price": d.get("price"),
                    "rating": d.get("rating"),
                    "brand": d.get("brand"),
                    "category": d.get("category"),
                    "features": [],  # Features are embedded in the vector, not returned separately
                    "ingredients": d.get("ingredients"),
                    "relevance_score": round(d.get("score", r.score), 4),
                    "product_url": d.get("product_url")
                })
            
            # Cache results
            rag_search_cache.set(json.dumps(cache_key_data, sort_keys=True), result_dicts)
            
            # Log the call
            duration_ms = (time.time() - start_time) * 1000
            tool_logger.log_tool_call(
                tool_name="rag.search",
                request_payload={"query": query, "budget": budget, "filters": filters, "num_results": num_results},
                response_data=result_dicts,
                duration_ms=duration_ms,
                success=True,
                cache_hit=False,
                metadata={"reranked": rerank, "source": "vector_store", "doc_count": len(result_dicts)}
            )
            
            return result_dicts
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            tool_logger.log_tool_call(
                tool_name="rag.search",
                request_payload={"query": query, "budget": budget, "filters": filters, "num_results": num_results},
                response_data=None,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            logger.error(f"RAG search failed: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if self._rag:
            return self._rag.collection.count()
        return 0


# Global instance
_rag_search_tool: Optional[RAGSearchTool] = None


def get_rag_search_tool() -> RAGSearchTool:
    """Get or create the global RAGSearchTool instance."""
    global _rag_search_tool
    if _rag_search_tool is None:
        _rag_search_tool = RAGSearchTool()
    return _rag_search_tool
