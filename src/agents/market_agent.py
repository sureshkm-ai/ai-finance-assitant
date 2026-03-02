"""Market Analysis Agent: Provides real-time market insights."""
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseFinanceAgent
from ..core.state import AgentState
from ..utils.market_data import market_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Finnie's Market Analysis specialist. You provide real-time market insights and help users understand market conditions.

Your role:
- Explain market data in plain English
- Put market movements in historical and educational context
- Avoid making predictions or specific investment recommendations
- Help users understand what market data means for their investment strategy
- Explain concepts like bull/bear markets, volatility, sector rotation

{user_profile_context}
"""

class MarketAnalysisAgent(BaseFinanceAgent):
    """Provides real-time market insights and analysis."""

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_name = "Market Analysis Agent"
        self.agent_description = "Provides real-time market insights"

    def process(self, state: AgentState) -> AgentState:
        try:
            query = state.get("user_query", "")
            user_profile_ctx = self.get_user_profile_context(state)

            # Extract symbols from query (simple heuristic)
            symbols_in_query = re.findall(r'\b[A-Z]{1,5}\b', query)
            # Filter likely stock tickers (exclude common English words)
            stop_words = {"A", "I", "IT", "IS", "IN", "AT", "TO", "OR", "AND", "THE", "FOR", "ETF", "S&P", "US", "UK"}
            symbols = [s for s in symbols_in_query if s not in stop_words and len(s) >= 2][:3]

            # Get market overview
            market_overview = market_client.get_market_overview()

            # Get specific stock data if symbols found
            stock_data = {}
            if symbols:
                stock_data = market_client.get_multiple_quotes(symbols)

            # Build market data context
            indices = market_overview.get("indices", {})
            market_summary = f"Market Status: {market_overview.get('market_status', 'unknown').upper()}\n\n"
            market_summary += "Major Indices:\n"
            for name, data in indices.items():
                if "error" not in data:
                    direction = "📈" if data.get("direction") == "up" else "📉"
                    market_summary += f"  {direction} {name}: {data.get('value', 'N/A'):,} ({data.get('change_pct', 0):+.2f}%)\n"

            stock_summary = ""
            if stock_data:
                stock_summary = "\nRequested Securities:\n"
                for symbol, data in stock_data.items():
                    if "error" not in data:
                        direction = "📈" if data.get("change_pct", 0) >= 0 else "📉"
                        stock_summary += f"  {direction} {symbol} ({data.get('name', symbol)}): ${data.get('current_price', 0):,.2f} ({data.get('change_pct', 0):+.2f}%)\n"
                        if data.get("52_week_high"):
                            stock_summary += f"     52-week range: ${data.get('52_week_low', 0):,.2f} - ${data.get('52_week_high', 0):,.2f} | Beta: {data.get('beta', 'N/A')} | P/E: {data.get('pe_ratio', 'N/A')}\n"

            rag_context = self.get_context(query, k=3)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(user_profile_context=user_profile_ctx)),
                HumanMessage(content=f"""User question: {query}

Current Market Data:
{market_summary}{stock_summary}

Educational context:
{rag_context}

Provide market insights that address the user's question, explaining what the data means and putting it in educational context.""")
            ]

            response = self.llm.invoke(messages)
            return self.build_response(state, response.content, {
                "market_data_fetched": True,
                "symbols_analyzed": symbols,
                "market_status": market_overview.get("market_status")
            })

        except Exception as e:
            return self.handle_error(state, e)
