"""Market data client with yfinance (primary) and alpha_vantage (fallback)."""
import os
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from .cache import market_cache

load_dotenv(override=True)
logger = logging.getLogger(__name__)

MAJOR_INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",
    "^RUT": "Russell 2000",
    "^VIX": "VIX (Volatility)"
}


class MarketDataClient:
    """Fetches market data using yfinance with alpha_vantage fallback."""

    def __init__(self):
        self.av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.cache = market_cache
        logger.info("MarketDataClient initialized")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive info for a single stock symbol."""
        symbol = symbol.upper().strip()
        cache_key = f"stock_info_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d")

            if hist.empty:
                return self._get_fallback_data(symbol)

            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0
            prev_price = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price * 100) if prev_price != 0 else 0

            result = {
                "symbol": symbol,
                "name": info.get("longName", info.get("shortName", symbol)),
                "current_price": round(current_price, 2),
                "previous_close": round(prev_price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "volume": info.get("volume", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "52_week_high": info.get("fiftyTwoWeekHigh", None),
                "52_week_low": info.get("fiftyTwoWeekLow", None),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "")[:300] if info.get("longBusinessSummary") else "",
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", None),
                "source": "yfinance",
                "timestamp": datetime.now().isoformat(),
                "data_freshness": "live"
            }
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}, trying fallback")
            return self._get_fallback_data(symbol)

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get quotes for multiple symbols."""
        cache_key = f"multi_quotes_{'_'.join(sorted(symbols))}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_stock_info(symbol)
                time.sleep(0.1)  # small delay between requests
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = {"symbol": symbol, "error": str(e)}

        self.cache.set(cache_key, results, ttl=1800)
        return results

    def get_market_overview(self) -> Dict[str, Any]:
        """Get major market indices overview."""
        cache_key = "market_overview"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        indices_data = {}
        for symbol, name in MAJOR_INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if not hist.empty:
                    current = float(hist["Close"].iloc[-1])
                    prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
                    change_pct = ((current - prev) / prev * 100) if prev != 0 else 0
                    indices_data[name] = {
                        "symbol": symbol,
                        "value": round(current, 2),
                        "change_pct": round(change_pct, 2),
                        "direction": "up" if change_pct >= 0 else "down"
                    }
            except Exception as e:
                logger.warning(f"Could not fetch {symbol}: {e}")
                indices_data[name] = {"symbol": symbol, "error": "unavailable"}

        result = {
            "indices": indices_data,
            "timestamp": datetime.now().isoformat(),
            "market_status": self._get_market_status()
        }
        self.cache.set(cache_key, result, ttl=300)  # 5 min for overview
        return result

    def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data for a symbol."""
        cache_key = f"historical_{symbol}_{period}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period)
            if not hist.empty:
                self.cache.set(cache_key, hist, ttl=3600)
            return hist
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return pd.DataFrame()

    def search_news(self, query: str = "", symbols: List[str] = None, limit: int = 5) -> List[Dict]:
        """Get financial news for symbols or general market news."""
        cache_key = f"news_{query}_{str(symbols)}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        news_items = []
        try:
            if symbols:
                for symbol in symbols[:2]:
                    ticker = yf.Ticker(symbol.upper())
                    news = ticker.news or []
                    for item in news[:limit]:
                        news_items.append({
                            "title": item.get("title", ""),
                            "summary": item.get("summary", item.get("title", "")),
                            "url": item.get("link", ""),
                            "source": item.get("publisher", "Yahoo Finance"),
                            "published": datetime.fromtimestamp(item.get("providerPublishTime", time.time())).strftime("%Y-%m-%d %H:%M"),
                            "symbol": symbol
                        })
            else:
                # General market news from SPY
                ticker = yf.Ticker("SPY")
                news = ticker.news or []
                for item in news[:limit]:
                    news_items.append({
                        "title": item.get("title", ""),
                        "summary": item.get("summary", item.get("title", "")),
                        "url": item.get("link", ""),
                        "source": item.get("publisher", "Yahoo Finance"),
                        "published": datetime.fromtimestamp(item.get("providerPublishTime", time.time())).strftime("%Y-%m-%d %H:%M"),
                        "symbol": "Market"
                    })
        except Exception as e:
            logger.error(f"News fetch failed: {e}")

        self.cache.set(cache_key, news_items, ttl=900)  # 15 min for news
        return news_items[:limit]

    def get_portfolio_data(self, holdings: List[Dict]) -> Dict[str, Any]:
        """Enrich portfolio holdings with current market data.

        Args:
            holdings: List of dicts with keys: symbol, shares, avg_cost

        Returns:
            Dict with enriched holdings and portfolio totals
        """
        enriched = []
        total_value = 0.0
        total_cost = 0.0

        for holding in holdings:
            symbol = holding.get("symbol", "").upper()
            shares = float(holding.get("shares", 0))
            avg_cost = float(holding.get("avg_cost", 0))

            try:
                info = self.get_stock_info(symbol)
                current_price = info.get("current_price", 0)
                current_value = current_price * shares
                cost_basis = avg_cost * shares
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0

                enriched.append({
                    "symbol": symbol,
                    "name": info.get("name", symbol),
                    "shares": shares,
                    "avg_cost": avg_cost,
                    "current_price": current_price,
                    "current_value": round(current_value, 2),
                    "cost_basis": round(cost_basis, 2),
                    "gain_loss": round(gain_loss, 2),
                    "gain_loss_pct": round(gain_loss_pct, 2),
                    "sector": info.get("sector", "N/A"),
                    "change_pct": info.get("change_pct", 0),
                    "beta": info.get("beta", 1.0),
                    "dividend_yield": info.get("dividend_yield", 0),
                })
                total_value += current_value
                total_cost += cost_basis
            except Exception as e:
                logger.error(f"Portfolio enrichment failed for {symbol}: {e}")

        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0

        return {
            "holdings": enriched,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain_loss": round(total_gain_loss, 2),
            "total_gain_loss_pct": round(total_gain_loss_pct, 2),
            "num_holdings": len(enriched),
            "timestamp": datetime.now().isoformat()
        }

    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Try alpha_vantage as fallback or return placeholder."""
        if self.av_api_key and self.av_api_key != "your_alpha_vantage_api_key_here":
            try:
                import requests
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.av_api_key}"
                resp = requests.get(url, timeout=10)
                data = resp.json().get("Global Quote", {})
                if data:
                    price = float(data.get("05. price", 0))
                    prev = float(data.get("08. previous close", price))
                    change = float(data.get("09. change", 0))
                    change_pct = float(data.get("10. change percent", "0%").replace("%", ""))
                    return {
                        "symbol": symbol,
                        "name": symbol,
                        "current_price": round(price, 2),
                        "previous_close": round(prev, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "volume": int(data.get("06. volume", 0)),
                        "source": "alpha_vantage",
                        "timestamp": datetime.now().isoformat(),
                        "data_freshness": "live"
                    }
            except Exception as e:
                logger.error(f"Alpha Vantage fallback failed for {symbol}: {e}")

        return {
            "symbol": symbol,
            "name": symbol,
            "current_price": 0,
            "change_pct": 0,
            "error": "Data temporarily unavailable",
            "source": "fallback",
            "timestamp": datetime.now().isoformat(),
            "data_freshness": "unavailable"
        }

    def _get_market_status(self) -> str:
        """Determine if market is open (US Eastern time simplified check)."""
        now = datetime.now()
        # Simple heuristic: weekday 9:30-16:00 ET (approximate)
        if now.weekday() >= 5:
            return "closed"
        hour = now.hour
        if 9 <= hour < 16:
            return "open"
        return "closed"


# Global client instance
market_client = MarketDataClient()
