"""Tests for finance agents."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from src.core.state import AgentState
from src.agents.base_agent import BaseFinanceAgent, DISCLAIMER

# Mock LLM response
class MockLLMResponse:
    def __init__(self, content="Mock response"):
        self.content = content

class MockLLM:
    def invoke(self, messages):
        return MockLLMResponse("This is a mock educational response about finances.")

class ConcreteAgent(BaseFinanceAgent):
    """Concrete implementation for testing."""
    def process(self, state):
        return self.build_response(state, "Test response")

def make_state(**kwargs) -> AgentState:
    """Create a minimal test state."""
    defaults = {
        "messages": [],
        "user_query": "What is investing?",
        "agent_response": "",
        "agent_type": "finance_qa",
        "conversation_history": [],
        "user_profile": {"risk_tolerance": "moderate", "experience_level": "beginner", "investment_goals": [], "time_horizon": "long"},
        "portfolio_data": None,
        "error": None,
        "metadata": {},
        "needs_routing": False
    }
    defaults.update(kwargs)
    return defaults

class TestBaseAgent:
    def test_initialization(self):
        agent = ConcreteAgent(MockLLM())
        assert agent.llm is not None
        assert agent.agent_name == "Base Agent"

    def test_process_returns_state(self):
        agent = ConcreteAgent(MockLLM())
        state = make_state()
        result = agent.process(state)
        assert "agent_response" in result
        assert result["agent_response"] == "Test response" + DISCLAIMER

    def test_disclaimer_added(self):
        agent = ConcreteAgent(MockLLM())
        state = make_state()
        result = agent.process(state)
        assert DISCLAIMER in result["agent_response"]

    def test_handle_error(self):
        agent = ConcreteAgent(MockLLM())
        state = make_state()
        result = agent.handle_error(state, Exception("test error"))
        assert result["error"] == "test error"
        assert "encountered an issue" in result["agent_response"]

    def test_conversation_history_updated(self):
        agent = ConcreteAgent(MockLLM())
        state = make_state(user_query="test question")
        result = agent.process(state)
        assert len(result["conversation_history"]) == 1
        assert result["conversation_history"][0]["user"] == "test question"

    def test_user_profile_context(self):
        agent = ConcreteAgent(MockLLM())
        state = make_state(user_profile={
            "risk_tolerance": "aggressive",
            "experience_level": "advanced",
            "investment_goals": ["retirement"],
            "time_horizon": "long"
        })
        ctx = agent.get_user_profile_context(state)
        assert "aggressive" in ctx
        assert "advanced" in ctx

    def test_get_context_without_retriever(self):
        agent = ConcreteAgent(MockLLM(), retriever=None)
        ctx = agent.get_context("test query")
        assert ctx == ""

class TestFinanceQAAgent:
    def test_processes_query(self):
        from src.agents.finance_qa_agent import FinanceQAAgent
        agent = FinanceQAAgent(MockLLM())
        state = make_state(user_query="What is a stock?")
        result = agent.process(state)
        assert result["agent_response"] != ""
        assert result["error"] is None

    def test_agent_name(self):
        from src.agents.finance_qa_agent import FinanceQAAgent
        agent = FinanceQAAgent(MockLLM())
        assert agent.agent_name == "Finance Q&A Agent"

class TestPortfolioAgent:
    def test_no_portfolio_data(self):
        from src.agents.portfolio_agent import PortfolioAnalysisAgent
        agent = PortfolioAnalysisAgent(MockLLM())
        state = make_state(portfolio_data=None)
        result = agent.process(state)
        assert "Portfolio tab" in result["agent_response"] or "provide" in result["agent_response"].lower()

    def test_portfolio_metrics_calculation(self):
        from src.agents.portfolio_agent import calculate_portfolio_metrics
        portfolio_data = {
            "holdings": [
                {"symbol": "AAPL", "current_value": 1500, "sector": "Technology", "beta": 1.2, "dividend_yield": 0.005, "gain_loss_pct": 5.0},
                {"symbol": "BND", "current_value": 1000, "sector": "Fixed Income", "beta": 0.1, "dividend_yield": 0.03, "gain_loss_pct": -1.0}
            ],
            "total_value": 2500,
            "total_cost": 2400,
            "total_gain_loss": 100,
            "total_gain_loss_pct": 4.17
        }
        metrics = calculate_portfolio_metrics(portfolio_data)
        assert "diversification_score" in metrics
        assert "sector_allocation" in metrics
        assert "Technology" in metrics["sector_allocation"]
        assert metrics["num_holdings"] == 2

class TestGoalPlanningAgent:
    def test_future_value_calculation(self):
        from src.agents.goal_planning_agent import calculate_future_value
        # 10,000 at 7% for 10 years ~= 19,672
        fv = calculate_future_value(10000, 7.0, 10)
        assert abs(fv - 19671.51) < 10

    def test_required_monthly_savings(self):
        from src.agents.goal_planning_agent import calculate_required_monthly_savings
        # Should return 0 if already on track
        req = calculate_required_monthly_savings(100, 1000, 7.0, 1)
        assert req == 0

    def test_build_projections(self):
        from src.agents.goal_planning_agent import build_goal_projections
        projections = build_goal_projections(100000, 10000, 500, 10)
        assert "conservative (4%)" in projections
        assert "moderate (7%)" in projections
        assert "aggressive (10%)" in projections
        for scenario in projections.values():
            assert "projected_value" in scenario
            assert "on_track" in scenario

class TestMarketAgent:
    def test_agent_name(self):
        from src.agents.market_agent import MarketAnalysisAgent
        agent = MarketAnalysisAgent(MockLLM())
        assert agent.agent_name == "Market Analysis Agent"

class TestNewsAgent:
    def test_agent_name(self):
        from src.agents.news_agent import NewsSynthesizerAgent
        agent = NewsSynthesizerAgent(MockLLM())
        assert agent.agent_name == "News Synthesizer Agent"

class TestTaxAgent:
    def test_agent_name(self):
        from src.agents.tax_agent import TaxEducationAgent
        agent = TaxEducationAgent(MockLLM())
        assert agent.agent_name == "Tax Education Agent"

    def test_processes_tax_query(self):
        from src.agents.tax_agent import TaxEducationAgent
        agent = TaxEducationAgent(MockLLM())
        state = make_state(user_query="What is a Roth IRA?")
        result = agent.process(state)
        assert result["agent_response"] != ""
