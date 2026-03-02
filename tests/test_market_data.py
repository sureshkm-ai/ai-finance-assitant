"""Tests for market data utilities."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.cache import TTLCache, CacheEntry
import time

class TestTTLCache:
    def test_set_and_get(self):
        cache = TTLCache(default_ttl=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_miss_returns_none(self):
        cache = TTLCache()
        assert cache.get("nonexistent") is None

    def test_expired_returns_none(self):
        cache = TTLCache(default_ttl=1)
        cache.set("key", "value", ttl=0.01)
        time.sleep(0.05)
        assert cache.get("key") is None

    def test_delete(self):
        cache = TTLCache()
        cache.set("key", "value")
        cache.delete("key")
        assert cache.get("key") is None

    def test_clear(self):
        cache = TTLCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stats(self):
        cache = TTLCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["active"] == 2

    def test_custom_ttl(self):
        cache = TTLCache(default_ttl=60)
        cache.set("key", "value", ttl=300)
        assert cache.get("key") == "value"

class TestGoalCalculations:
    def test_future_value_no_payments(self):
        from src.agents.goal_planning_agent import calculate_future_value
        fv = calculate_future_value(10000, 7.0, 10, 0)
        assert abs(fv - 19671.51) < 10

    def test_future_value_zero_rate(self):
        from src.agents.goal_planning_agent import calculate_future_value
        fv = calculate_future_value(10000, 0.0, 10, 1000)
        assert fv == 20000

    def test_portfolio_data_format(self):
        """Test that portfolio data is well-structured."""
        import json
        sample_path = os.path.join(os.path.dirname(__file__), "../src/data/sample_portfolios.json")
        with open(sample_path) as f:
            data = json.load(f)
        assert "sample_portfolios" in data
        assert len(data["sample_portfolios"]) > 0
        for portfolio in data["sample_portfolios"]:
            assert "holdings" in portfolio
            for h in portfolio["holdings"]:
                assert "symbol" in h
                assert "shares" in h
                assert "avg_cost" in h

    def test_glossary_format(self):
        """Test that glossary data is well-structured."""
        import json
        glossary_path = os.path.join(os.path.dirname(__file__), "../src/data/glossary.json")
        with open(glossary_path) as f:
            data = json.load(f)
        assert "terms" in data
        assert len(data["terms"]) >= 40
        for term in data["terms"]:
            assert "term" in term
            assert "definition" in term
