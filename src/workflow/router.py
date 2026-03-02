"""Query router: classifies user queries and routes to appropriate agent."""
import logging
import re
from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from ..core.state import AgentState

logger = logging.getLogger(__name__)

AgentType = Literal["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]

ROUTER_SYSTEM_PROMPT = """You are a query router for a financial education assistant. Your job is to classify user queries into one of these categories:

- finance_qa: General financial education questions (what is a stock, how does compound interest work, explain ETFs, investment strategies, financial concepts)
- portfolio: Portfolio analysis requests (analyze my portfolio, review my holdings, portfolio performance, diversification check, portfolio rebalancing)
- market: Market data and analysis (stock prices, market performance today, how is AAPL doing, market trends, sector performance)
- goal_planning: Financial goal setting and planning (save for retirement, buy a house in 5 years, reach $1M, how much to save monthly, projection calculations)
- news: Financial news and recent events (latest market news, what happened with Tesla, recent economic news, earnings reports)
- tax: Tax-related questions (capital gains tax, 401k vs Roth IRA, tax-loss harvesting, tax-efficient investing, IRA contribution limits)

Respond with ONLY the category name, nothing else. Choose the most specific matching category.
"""

# Keyword-based fallback routing (faster, no API call)
ROUTING_KEYWORDS = {
    "portfolio": ["portfolio", "holdings", "my stocks", "my investments", "analyze my", "rebalance", "asset allocation", "diversif"],
    "market": ["stock price", "market today", "how is", "trading at", "current price", "market performance", "bull market", "bear market", "S&P", "nasdaq", "dow jones", "index"],
    "goal_planning": ["save for", "reach", "retirement goal", "buy a house", "how much should i save", "projection", "how long will it take", "investment goal", "financial goal", "by age"],
    "news": ["news", "latest", "recent", "today's market", "what happened", "earnings", "announcement", "fed", "inflation report"],
    "tax": ["tax", "401k", "ira", "roth", "capital gains", "tax-loss", "wash sale", "deductible", "tax bracket", "after-tax", "pre-tax", "hsa"],
}

def route_by_keywords(query: str) -> AgentType:
    """Fast keyword-based routing as fallback."""
    query_lower = query.lower()

    scores = {agent: 0 for agent in ROUTING_KEYWORDS}
    for agent, keywords in ROUTING_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[agent] += 1

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "finance_qa"

def create_router(llm):
    """Create a router function that uses LLM for classification."""

    def route(state: AgentState) -> AgentState:
        """Route the user query to the appropriate agent."""
        query = state.get("user_query", "")

        if not query.strip():
            return {**state, "agent_type": "finance_qa", "needs_routing": False}

        # Try LLM-based routing first
        try:
            messages = [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"Classify this query: {query}")
            ]
            response = llm.invoke(messages)
            agent_type = response.content.strip().lower().replace(" ", "_")

            valid_types = ["finance_qa", "portfolio", "market", "goal_planning", "news", "tax"]
            if agent_type not in valid_types:
                agent_type = route_by_keywords(query)

            logger.info(f"LLM Router classified '{query[:50]}...' as: {agent_type}")

        except Exception as e:
            logger.warning(f"LLM routing failed, using keyword fallback: {e}")
            agent_type = route_by_keywords(query)

        return {**state, "agent_type": agent_type, "needs_routing": False}

    return route
