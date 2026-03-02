"""In-memory TTL cache for market data."""
import time
import logging
from typing import Any, Optional, Dict
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with TTL."""
    value: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl


class TTLCache:
    """Thread-safe in-memory cache with TTL expiry."""

    def __init__(self, default_ttl: int = 1800):
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._cache[key]
                logger.debug(f"Cache expired for key: {key}")
                return None
            logger.debug(f"Cache hit for key: {key}")
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL."""
        with self._lock:
            ttl = ttl or self.default_ttl
            self._cache[key] = CacheEntry(value=value, timestamp=time.time(), ttl=ttl)
            logger.debug(f"Cache set for key: {key}, TTL: {ttl}s")

    def delete(self, key: str) -> None:
        """Remove a key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for e in self._cache.values() if e.is_expired())
            return {"total_entries": total, "expired": expired, "active": total - expired}


# Global cache instance
market_cache = TTLCache(default_ttl=1800)
