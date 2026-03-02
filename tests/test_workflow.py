"""Tests for workflow orchestration and routing."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
from src.workflow.router import route_by_keywords, ROUTING_KEYWORDS
from src.core.state import AgentState

def make_state(query="", agent_type="router", **kwargs) -> AgentState:
    defaults = {
        "messages": [],
        "user_query": query,
        "agent_response": "",
        "agent_type": agent_type,
        "conversation_history": [],
        "user_profile": {"risk_tolerance": "moderate", "experience_level": "beginner", "investment_goals": [], "time_horizon": "long"},
        "portfolio_data": None,
        "error": None,
        "metadata": {},
        "needs_routing": True
    }
    defaults.update(kwargs)
    return defaults

class TestKeywordRouter:
    def test_portfolio_routing(self):
        assert route_by_keywords("analyze my portfolio") == "portfolio"
        assert route_by_keywords("review my holdings") == "portfolio"

    def test_market_routing(self):
        assert route_by_keywords("what is the stock price of AAPL") == "market"
        assert route_by_keywords("how is the S&P performing today") == "market"

    def test_goal_routing(self):
        assert route_by_keywords("how much should I save for retirement goal") == "goal_planning"
        assert route_by_keywords("I want to reach $500k in 10 years") == "goal_planning"

    def test_news_routing(self):
        assert route_by_keywords("latest market news") == "news"
        assert route_by_keywords("recent earnings report") == "news"

    def test_tax_routing(self):
        assert route_by_keywords("how does 401k work") == "tax"
        assert route_by_keywords("capital gains tax rate") == "tax"
        assert route_by_keywords("roth ira vs traditional") == "tax"

    def test_default_to_finance_qa(self):
        assert route_by_keywords("what is compound interest") == "finance_qa"
        assert route_by_keywords("hello") == "finance_qa"

class TestWorkflowState:
    def test_initial_state_creation(self):
        from src.workflow.graph import create_initial_state
        state = create_initial_state("What is a stock?")
        assert state["user_query"] == "What is a stock?"
        assert state["needs_routing"] == True
        assert state["user_profile"]["risk_tolerance"] == "moderate"

    def test_initial_state_with_profile(self):
        from src.workflow.graph import create_initial_state
        state = create_initial_state(
            "Test query",
            user_profile={"risk_tolerance": "aggressive", "experience_level": "advanced", "investment_goals": [], "time_horizon": "long"}
        )
        assert state["user_profile"]["risk_tolerance"] == "aggressive"
