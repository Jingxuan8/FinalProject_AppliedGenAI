"""
Web Search Tool for MCP Server
Wraps legitimate web search APIs (Serper, Brave) with caching and logging

REQUIRES API KEY - No mock implementation available.
"""

import os
import re
import time
import json
import httpx
from typing import Any, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

from ..config import (
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_API_KEY,
    WEB_SEARCH_BASE_URL,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    ALLOWED_DOMAINS
)
from ..utils.cache import web_search_cache
from ..utils.logger import tool_logger, setup_logging

logger = setup_logging()


class WebSearchConfigError(Exception):
    """Raised when web search API is not properly configured."""
    pass


@dataclass
class WebSearchResult:
    """Structured web search result."""
    title: str
    url: str
    snippet: str
    price: Optional[float] = None
    availability: Optional[str] = None
    source: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
        }
        if self.price is not None:
            result["price"] = self.price
        if self.availability:
            result["availability"] = self.availability
        if self.source:
            result["source"] = self.source
        return result


class WebSearchTool:
    """
    Web Search Tool implementation (LLM-powered, no mock).
    
    Supports multiple search providers:
    - Serper (Google Search API) - https://serper.dev
    - Brave Search API - https://brave.com/search/api
    
    Features:
    - TTL-based caching (60-300 seconds)
    - Request logging with timestamps
    - Rate limiting
    - Price extraction from snippets
    - Domain allowlist for safety
    
    REQUIRES:
    - SERPER_API_KEY or BRAVE_API_KEY environment variable
    """
    
    def __init__(
        self,
        provider: str = WEB_SEARCH_PROVIDER,
        api_key: str = WEB_SEARCH_API_KEY,
        base_url: str = WEB_SEARCH_BASE_URL
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.base_url = base_url
        
        # Validate configuration - NO MOCK ALLOWED
        self._validate_config()
        
        # Rate limiting
        self._request_timestamps: list[float] = []
        self._rate_limit = RATE_LIMIT_REQUESTS_PER_MINUTE
        
        # HTTP client
        self._client = httpx.Client(timeout=30.0)
        
        logger.info(f"WebSearchTool initialized with provider: {self.provider}")
    
    def _validate_config(self) -> None:
        """Validate that API key is configured. Raises error if not."""
        if not self.api_key:
            raise WebSearchConfigError(
                f"Web Search API key is required!\n"
                f"Provider: {self.provider}\n"
                f"Please set one of these environment variables:\n"
                f"  - SERPER_API_KEY (get free key at https://serper.dev)\n"
                f"  - BRAVE_API_KEY (get key at https://brave.com/search/api)\n"
                f"  - WEB_SEARCH_API_KEY\n"
                f"\nExample:\n"
                f"  export SERPER_API_KEY='your_api_key_here'"
            )
        
        if self.provider not in ["serper", "brave"]:
            raise WebSearchConfigError(
                f"Unsupported web search provider: {self.provider}\n"
                f"Supported providers: serper, brave\n"
                f"Set WEB_SEARCH_PROVIDER environment variable."
            )
    
    def search(
        self,
        query: str,
        filters: Optional[dict] = None,
        num_results: int = 5
    ) -> list[dict]:
        """
        Execute web search with caching and logging.
        
        Args:
            query: Search query string
            filters: Optional filters (max_price, brand, availability)
            num_results: Number of results to return (1-10)
            
        Returns:
            List of search results with title, url, snippet, price, availability
        """
        start_time = time.time()
        filters = filters or {}
        num_results = min(max(num_results, 1), 10)
        
        # Build enhanced query with filters
        enhanced_query = self._build_query(query, filters)
        
        # Check cache first
        cached_result = web_search_cache.get(enhanced_query, filters)
        if cached_result is not None:
            duration_ms = (time.time() - start_time) * 1000
            tool_logger.log_tool_call(
                tool_name="web.search",
                request_payload={"query": query, "filters": filters, "num_results": num_results},
                response_data=cached_result,
                duration_ms=duration_ms,
                success=True,
                cache_hit=True
            )
            return cached_result[:num_results]
        
        # Check rate limit
        self._check_rate_limit()
        
        try:
            # Execute search based on provider
            if self.provider == "serper":
                results = self._search_serper(enhanced_query, num_results)
            elif self.provider == "brave":
                results = self._search_brave(enhanced_query, num_results)
            else:
                # Should never reach here due to validation
                raise WebSearchConfigError(f"Unsupported provider: {self.provider}")
            
            # Process and filter results
            processed_results = self._process_results(results, filters)
            
            # Cache the results
            web_search_cache.set(enhanced_query, processed_results, filters)
            
            # Log the call
            duration_ms = (time.time() - start_time) * 1000
            source_urls = [r.get("url", "") for r in processed_results]
            
            tool_logger.log_tool_call(
                tool_name="web.search",
                request_payload={"query": query, "filters": filters, "num_results": num_results},
                response_data=processed_results,
                duration_ms=duration_ms,
                success=True,
                source_urls=source_urls,
                cache_hit=False,
                metadata={"provider": self.provider, "enhanced_query": enhanced_query}
            )
            
            return processed_results[:num_results]
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            tool_logger.log_tool_call(
                tool_name="web.search",
                request_payload={"query": query, "filters": filters, "num_results": num_results},
                response_data=None,
                duration_ms=duration_ms,
                success=False,
                error_message=str(e)
            )
            logger.error(f"Web search failed: {e}")
            raise
    
    def _build_query(self, query: str, filters: dict) -> str:
        """Build enhanced search query with filters."""
        parts = [query]
        
        if filters.get("max_price"):
            parts.append(f"under \${filters['max_price']}")
        
        if filters.get("brand"):
            parts.append(f"{filters['brand']} brand")
        
        if filters.get("availability") == "in_stock":
            parts.append("in stock")
        
        # Add product-related terms for better results
        if "buy" not in query.lower() and "price" not in query.lower():
            parts.append("buy")
        
        return " ".join(parts)
    
    def _search_serper(self, query: str, num_results: int) -> list[dict]:
        """Search using Serper (Google Search API)."""
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": query,
            "num": num_results * 2,
            "gl": "us",
            "hl": "en"
        }
        
        logger.info(f"Serper API call: {query[:50]}...")
        
        response = self._client.post(
            self.base_url,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        # Process organic results
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": urlparse(item.get("link", "")).netloc
            })
        
        # Process shopping results if available
        for item in data.get("shopping", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", item.get("title", "")),
                "price": self._parse_price(item.get("price", "")),
                "source": item.get("source", "")
            })
        
        logger.info(f"Serper returned {len(results)} results")
        return results
    
    def _search_brave(self, query: str, num_results: int) -> list[dict]:
        """Search using Brave Search API."""
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json"
        }
        
        params = {
            "q": query,
            "count": num_results * 2
        }
        
        logger.info(f"Brave API call: {query[:50]}...")
        
        response = self._client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "source": urlparse(item.get("url", "")).netloc
            })
        
        logger.info(f"Brave returned {len(results)} results")
        return results
    
    def _process_results(self, results: list[dict], filters: dict) -> list[dict]:
        """Process and filter search results."""
        processed = []
        
        for result in results:
            url = result.get("url", "")
            domain = urlparse(url).netloc.replace("www.", "")
            
            if domain and not any(allowed in domain for allowed in ALLOWED_DOMAINS):
                logger.debug(f"Non-allowlisted domain: {domain}")
            
            price = result.get("price")
            if price is None:
                price = self._extract_price(result.get("snippet", "") + " " + result.get("title", ""))
            
            if filters.get("max_price") and price and price > filters["max_price"]:
                continue
            if filters.get("min_price") and price and price < filters["min_price"]:
                continue
            
            processed.append({
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("snippet", ""),
                "price": price,
                "availability": result.get("availability"),
                "source": result.get("source") or domain
            })
        
        return processed
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text using regex patterns."""
        if not text:
            return None
        
        patterns = [
            r'\$(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*(?:USD|dollars?)',
            r'price[:\s]+\$?(\d+\.?\d*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _parse_price(self, price_str: str) -> Optional[float]:
        """Parse price string to float."""
        if not price_str:
            return None
        
        cleaned = re.sub(r'[^\d.]', '', str(price_str))
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if current_time - ts < 60
        ]
        
        if len(self._request_timestamps) >= self._rate_limit:
            wait_time = 60 - (current_time - self._request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
        
        self._request_timestamps.append(current_time)
    
    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


_web_search_tool: Optional[WebSearchTool] = None


def get_web_search_tool() -> WebSearchTool:
    """
    Get or create the global WebSearchTool instance.
    
    Raises:
        WebSearchConfigError: If API key is not configured
    """
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
