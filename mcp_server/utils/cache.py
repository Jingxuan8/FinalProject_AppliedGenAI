"""
Caching utility for MCP Server
Implements TTL-based caching (60-300 seconds as per requirements)
"""

import json
import hashlib
import time
from typing import Any, Optional
from pathlib import Path
from collections import OrderedDict
import threading

from ..config import CACHE_TTL_SECONDS, CACHE_MAX_SIZE, CACHE_DIR


class Cache:
    """
    Thread-safe LRU cache with TTL support.
    
    Features:
    - TTL (Time-To-Live) expiration: 60-300 seconds
    - LRU eviction when max size reached
    - Optional disk persistence
    - Thread-safe operations
    """
    
    def __init__(
        self,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        max_size: int = CACHE_MAX_SIZE,
        persist_to_disk: bool = False,
        cache_name: str = "default"
    ):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.persist_to_disk = persist_to_disk
        self.cache_name = cache_name
        
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()
        
        # Disk cache path
        self._disk_cache_path = CACHE_DIR / f"{cache_name}_cache.json"
        
        # Load from disk if persistence enabled
        if persist_to_disk:
            self._load_from_disk()
    
    def _generate_key(self, query: str, filters: Optional[dict] = None) -> str:
        """Generate a unique cache key from query and filters."""
        key_data = {"query": query, "filters": filters or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, query: str, filters: Optional[dict] = None) -> Optional[Any]:
        """
        Retrieve cached result if exists and not expired.
        
        Args:
            query: Search query string
            filters: Optional filter parameters
            
        Returns:
            Cached data if valid, None otherwise
        """
        key = self._generate_key(query, filters)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # Check TTL expiration
            if current_time - entry["timestamp"] > self.ttl_seconds:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            return entry["data"]
    
    def set(self, query: str, data: Any, filters: Optional[dict] = None) -> None:
        """
        Store data in cache with current timestamp.
        
        Args:
            query: Search query string
            data: Data to cache
            filters: Optional filter parameters
        """
        key = self._generate_key(query, filters)
        
        with self._lock:
            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = {
                "data": data,
                "timestamp": time.time(),
                "query": query,
                "filters": filters
            }
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Persist to disk if enabled
            if self.persist_to_disk:
                self._save_to_disk()
    
    def invalidate(self, query: str, filters: Optional[dict] = None) -> bool:
        """
        Remove a specific entry from cache.
        
        Returns:
            True if entry was removed, False if not found
        """
        key = self._generate_key(query, filters)
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self.persist_to_disk:
                    self._save_to_disk()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            if self.persist_to_disk:
                self._save_to_disk()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed_count = 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if current_time - entry["timestamp"] > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
            
            if removed_count > 0 and self.persist_to_disk:
                self._save_to_disk()
        
        return removed_count
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            valid_entries = sum(
                1 for entry in self._cache.values()
                if current_time - entry["timestamp"] <= self.ttl_seconds
            )
            
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_entries,
                "expired_entries": len(self._cache) - valid_entries,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "cache_name": self.cache_name
            }
    
    def _load_from_disk(self) -> None:
        """Load cache from disk file."""
        try:
            if self._disk_cache_path.exists():
                with open(self._disk_cache_path, "r") as f:
                    data = json.load(f)
                    self._cache = OrderedDict(data)
                    # Clean up expired entries after loading
                    self.cleanup_expired()
        except (json.JSONDecodeError, IOError) as e:
            # Start with empty cache if file is corrupted
            self._cache = OrderedDict()
    
    def _save_to_disk(self) -> None:
        """Save cache to disk file."""
        try:
            with open(self._disk_cache_path, "w") as f:
                json.dump(dict(self._cache), f)
        except IOError:
            pass  # Silently fail disk writes


# Global cache instances for different tools
web_search_cache = Cache(
    ttl_seconds=CACHE_TTL_SECONDS,
    max_size=CACHE_MAX_SIZE,
    persist_to_disk=True,
    cache_name="web_search"
)

rag_search_cache = Cache(
    ttl_seconds=CACHE_TTL_SECONDS * 2,  # RAG results can be cached longer
    max_size=CACHE_MAX_SIZE,
    persist_to_disk=True,
    cache_name="rag_search"
)

