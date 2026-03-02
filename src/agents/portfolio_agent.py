"""Portfolio Analysis Agent: Reviews and analyzes user portfolios."""
import logging
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseFinanceAgent
from ..core.state import AgentState
from ..utils.market_data import market_client

logger = logging.getLogger(__name__)

def calculate_portfolio_metrics(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate key portfolio metrics from enriched portfolio data."""
    holdings = portfolio_data.get("holdings", [])
    total_value = portfolio_data.get("total_value", 0)

    if not holdings or total_value == 0:
        return {}

    # Sector allocation
    sector_allocation = {}
    for h in holdings:
        sector = h.get("sector", "Unknown")
        value = h.get("current_value", 0)
        sector_allocation[sector] = sector_allocation.get(sector, 0) + value
    sector_pct = {k: round(v / total_value * 100, 1) for k, v in sector_allocation.items()}

    # Diversification score (simple heuristic: 0-100)
    num_holdings = len(holdings)
    num_sectors = len(sector_allocation)
    largest_pct = max(h.get("current_value", 0) / total_value * 100 for h in holdings) if holdings else 100

    div_score = min(100, (num_holdings * 5) + (num_sectors * 10) - max(0, largest_pct - 20))

    # Weighted beta
    total_beta = sum(h.get("beta", 1.0) * h.get("current_value", 0) for h in holdings if h.get("beta"))
    weighted_beta = round(total_beta / total_value, 2) if total_value > 0 else 1.0

    # Dividend yield (weighted)
    total_div = sum((h.get("dividend_yield", 0) or 0) * h.get("current_value", 0) for h in holdings)
    weighted_div_yield = round(total_div / total_value * 100, 2) if total_value > 0 else 0

    # Top performers and laggards
    sorted_by_gain = sorted(holdings, key=lambda x: x.get("gain_loss_pct", 0), reverse=True)

    return {
        "sector_allocation": sector_pct,
        "diversification_score": round(div_score),
        "weighted_beta": weighted_beta,
        "weighted_dividend_yield": weighted_div_yield,
        "num_holdings": num_holdings,
        "num_sectors": num_sectors,
        "largest_position_pct": round(largest_pct, 1),
        "top_performer": sorted_by_gain[0] if sorted_by_gain else None,
        "worst_performer": sorted_by_gain[-1] if sorted_by_gain else None,
        "total_value": total_value,
        "total_gain_loss": portfolio_data.get("total_gain_loss", 0),
        "total_gain_loss_pct": portfolio_data.get("total_gain_loss_pct", 0),
    }

SYSTEM_PROMPT = """You are Finnie's Portfolio Analysis specialist. You analyze investment portfolios and provide educational insights about portfolio composition, risk, diversification, and performance.

Your analysis should:
- Explain what the numbers mean in plain English
- Highlight strengths and potential areas for improvement
- Use the calculated metrics to support your analysis
- Provide educational context (e.g., what a beta of 1.5 means)
- Be encouraging while being honest about risks
- Make specific, actionable suggestions while noting this is educational only

{user_profile_context}
"""

class PortfolioAnalysisAgent(BaseFinanceAgent):
    """Analyzes user portfolios and provides educational insights."""

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_name = "Portfolio Analysis Agent"
        self.agent_description = "Reviews and analyzes user portfolios"

    def process(self, state: AgentState) -> AgentState:
        try:
            query = state.get("user_query", "")
            portfolio_data = state.get("portfolio_data")

            if not portfolio_data or not portfolio_data.get("holdings"):
                return self.build_response(state,
                    "To analyze your portfolio, please provide your holdings. You can input them in the Portfolio tab with your stock symbols, number of shares, and average purchase price.",
                    {"portfolio_analysis": False})

            # Enrich with live market data
            holdings_input = [
                {"symbol": h.get("symbol"), "shares": h.get("shares", 0), "avg_cost": h.get("avg_cost", 0)}
                for h in portfolio_data.get("holdings", [])
            ]

            if holdings_input:
                enriched = market_client.get_portfolio_data(holdings_input)
            else:
                enriched = portfolio_data

            metrics = calculate_portfolio_metrics(enriched)

            user_profile_ctx = self.get_user_profile_context(state)

            # Build analysis prompt
            holdings_summary = ""
            for h in enriched.get("holdings", []):
                gain_sign = "+" if h.get("gain_loss", 0) >= 0 else ""
                holdings_summary += f"- {h['symbol']} ({h.get('name', '')}): {h.get('shares')} shares @ ${h.get('current_price', 0):.2f} | Value: ${h.get('current_value', 0):,.2f} | Gain/Loss: {gain_sign}${h.get('gain_loss', 0):,.2f} ({gain_sign}{h.get('gain_loss_pct', 0):.1f}%) | Sector: {h.get('sector', 'N/A')}\n"

            metrics_summary = f"""
Portfolio Metrics:
- Total Value: ${metrics.get('total_value', 0):,.2f}
- Total Gain/Loss: ${metrics.get('total_gain_loss', 0):,.2f} ({metrics.get('total_gain_loss_pct', 0):.1f}%)
- Number of Holdings: {metrics.get('num_holdings', 0)}
- Sectors Represented: {metrics.get('num_sectors', 0)}
- Diversification Score: {metrics.get('diversification_score', 0)}/100
- Portfolio Beta: {metrics.get('weighted_beta', 1.0)} (market sensitivity)
- Dividend Yield: {metrics.get('weighted_dividend_yield', 0):.2f}%
- Largest Position: {metrics.get('largest_position_pct', 0):.1f}% of portfolio
- Sector Allocation: {json.dumps(metrics.get('sector_allocation', {}), indent=2)}
"""

            rag_context = self.get_context("portfolio diversification asset allocation risk management", k=3)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(user_profile_context=user_profile_ctx)),
                HumanMessage(content=f"""Analyze this portfolio and respond to: "{query}"

Holdings:
{holdings_summary}
{metrics_summary}

Relevant financial education context:
{rag_context}

Provide a thorough portfolio analysis addressing the user's specific question, explaining what the metrics mean, and offering educational insights about their portfolio composition and potential improvements.""")
            ]

            response = self.llm.invoke(messages)

            return self.build_response(state, response.content, {
                "portfolio_analysis": True,
                "metrics": metrics,
                "enriched_portfolio": enriched
            })

        except Exception as e:
            return self.handle_error(state, e)
