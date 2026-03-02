"""LangGraph state TypedDict definitions for the AI Finance Assistant."""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator


class UserProfile(TypedDict, total=False):
    """User profile containing investment preferences and history."""
    risk_tolerance: str  # "conservative", "moderate", "aggressive"
    experience_level: str  # "beginner", "intermediate", "advanced"
    investment_goals: List[str]
    time_horizon: str  # "short", "medium", "long"


class AgentState(TypedDict, total=False):
    """State dict for LangGraph workflow managing multi-agent orchestration."""
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    agent_response: str
    agent_type: str  # "finance_qa", "portfolio", "market", "goal_planning", "news", "tax", "router"
    conversation_history: List[Dict[str, str]]
    user_profile: UserProfile
    portfolio_data: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]
    needs_routing: bool
