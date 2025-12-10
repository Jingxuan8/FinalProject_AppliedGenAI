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

from ..config import VECTOR_STORE_PATH, COLLECTION_NAME, LLM_API_KEY, LLM_MODEL
from ..utils.cache import rag_search_cache
from ..utils.logger import tool_logger, setup_logging

# Import Person A's RAG implementation
from rag.rag_search import GamesRAG, ProductResult

# Try to import OpenAI for reranking
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = setup_logging()


class RAGSearchTool:
    """
    RAG Search Tool implementation.
    
    Wraps Person A's GamesRAG for MCP server integration.
    
    Features:
    - Hybrid retrieval (vector similarity + metadata filters)
    - LLM-based reranking for improved relevance
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
        self._openai_client: Optional[OpenAI] = None
        self._init_rag()
        self._init_openai()
    
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
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client for reranking."""
        if OPENAI_AVAILABLE and LLM_API_KEY:
            try:
                self._openai_client = OpenAI(api_key=LLM_API_KEY)
                logger.info("OpenAI client initialized for LLM reranking")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self._openai_client = None
        else:
            # Log but don't fail - will fail when rerank=True is requested
            if not OPENAI_AVAILABLE:
                logger.info("OpenAI package not installed - reranking will fail if requested")
            elif not LLM_API_KEY:
                logger.info("No LLM_API_KEY set - reranking will fail if requested")
    
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
            rerank: Whether to apply LLM-based reranking for better relevance
            
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
            
            # Get more results if reranking (to have better candidates)
            fetch_k = num_results * 3 if rerank else num_results
            
            # Execute search using Person A's GamesRAG
            results: list[ProductResult] = self._rag.rag_search(
                query=query,
                top_k=fetch_k,
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
            
            # Apply LLM reranking
            reranked = False
            if rerank:
                result_dicts = self._rerank_results(query, result_dicts, num_results)
                reranked = True
            
            # Trim to requested size
            result_dicts = result_dicts[:num_results]
            
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
                metadata={"reranked": reranked, "source": "vector_store", "doc_count": len(result_dicts)}
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
    
    def _rerank_results(
        self,
        query: str,
        results: list[dict],
        num_results: int
    ) -> list[dict]:
        """
        Apply LLM-based reranking to improve relevance.
        
        Args:
            query: Original search query
            results: List of search results to rerank
            num_results: Number of results to return
            
        Returns:
            Reranked list of results
            
        Raises:
            RuntimeError: If OpenAI client is not configured
        """
        if not self._openai_client:
            raise RuntimeError(
                "LLM reranking requires OpenAI API key!\n"
                "Please set OPENAI_API_KEY environment variable.\n"
                "Get your key at: https://platform.openai.com"
            )
        
        if len(results) <= 1:
            return results
        
        try:
            # Build product list for the prompt (limit to top 10 for efficiency)
            candidates = results[:min(len(results), 10)]
            products_text = "\n".join([
                f"{i+1}. {r['title']} (${r['price']}, {r.get('rating', 'N/A')}★, {r.get('brand', 'Unknown')})"
                for i, r in enumerate(candidates)
            ])
            
            prompt = f"""You are a product relevance expert. Given a user's search query, rank the following products from most to least relevant.

User Query: "{query}"

Products:
{products_text}

Instructions:
- Consider how well each product matches the user's intent
- Consider price, rating, and brand if mentioned in the query
- Return ONLY the product numbers in order of relevance, comma-separated
- Example response: 3, 1, 5, 2, 4

Most relevant to least relevant:"""

            response = self._openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0
            )
            
            # Parse the ranking response
            ranking_text = response.choices[0].message.content.strip()
            logger.info(f"LLM reranking response: {ranking_text}")
            
            # Extract numbers from the response
            import re
            numbers = re.findall(r'\d+', ranking_text)
            ranking = [int(n) - 1 for n in numbers if n.isdigit()]  # Convert to 0-indexed
            
            # Reorder results based on ranking
            reranked = []
            seen = set()
            for idx in ranking:
                if 0 <= idx < len(candidates) and idx not in seen:
                    reranked.append(candidates[idx])
                    seen.add(idx)
            
            # Add any missing results at the end
            for i, r in enumerate(candidates):
                if i not in seen:
                    reranked.append(r)
            
            # Update relevance scores to reflect new ranking
            for i, r in enumerate(reranked):
                r["relevance_score"] = round(1.0 - (i * 0.05), 4)  # Descending scores
            
            logger.info(f"Reranked {len(reranked)} results using LLM")
            return reranked[:num_results]
            
        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            raise RuntimeError(f"LLM reranking failed: {e}")
    
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
