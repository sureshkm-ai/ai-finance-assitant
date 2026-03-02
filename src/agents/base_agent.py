"""Base agent class for all finance agents."""
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from langchain_core.messages import AIMessage
from ..core.state import AgentState

logger = logging.getLogger(__name__)

DISCLAIMER = "\n\n⚠️ *Disclaimer: This information is for educational purposes only and does not constitute financial advice. Please consult a qualified financial advisor before making investment decisions.*"

class BaseFinanceAgent(ABC):
    """Abstract base class for all finance agents."""

    def __init__(self, llm, retriever=None):
        self.llm = llm
        self.retriever = retriever
        self.agent_name: str = "Base Agent"
        self.agent_description: str = "Base finance agent"
        self.include_disclaimer: bool = True

    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """Process user query and return updated state."""
        pass

    def get_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context from knowledge base."""
        if self.retriever:
            try:
                docs = self.retriever.retrieve(query, k=k)
                return self.retriever.format_context(docs)
            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")
        return ""

    def get_conversation_context(self, state: AgentState, max_turns: int = 5) -> str:
        """Format recent conversation history as context."""
        history = state.get("conversation_history", [])[-max_turns:]
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"User: {turn.get('user', '')}")
            lines.append(f"Assistant: {turn.get('assistant', '')}")
        return "\n".join(lines)

    def build_response(self, state: AgentState, response_text: str, metadata: Dict = None) -> AgentState:
        """Build the updated state with agent response."""
        if self.include_disclaimer:
            response_text = response_text + DISCLAIMER

        updated_metadata = dict(state.get("metadata", {}))
        if metadata:
            updated_metadata.update(metadata)
        updated_metadata["responding_agent"] = self.agent_name

        # Update conversation history
        history = list(state.get("conversation_history", []))
        history.append({
            "user": state.get("user_query", ""),
            "assistant": response_text
        })

        return {
            **state,
            "agent_response": response_text,
            "messages": [AIMessage(content=response_text)],
            "conversation_history": history[-50:],  # keep last 50 turns
            "metadata": updated_metadata,
            "error": None
        }

    def handle_error(self, state: AgentState, error: Exception) -> AgentState:
        """Handle errors gracefully."""
        error_msg = f"I encountered an issue while processing your request: {str(error)}. Please try rephrasing your question or try again in a moment."
        logger.error(f"{self.agent_name} error: {error}")
        return {
            **state,
            "agent_response": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "error": str(error),
            "metadata": {**state.get("metadata", {}), "error_agent": self.agent_name}
        }

    def get_user_profile_context(self, state: AgentState) -> str:
        """Format user profile for prompt context."""
        profile = state.get("user_profile", {})
        if not any(profile.values()):
            return ""
        parts = []
        if profile.get("experience_level"):
            parts.append(f"Experience Level: {profile['experience_level']}")
        if profile.get("risk_tolerance"):
            parts.append(f"Risk Tolerance: {profile['risk_tolerance']}")
        if profile.get("time_horizon"):
            parts.append(f"Time Horizon: {profile['time_horizon']}")
        if profile.get("investment_goals"):
            goals = ", ".join(profile["investment_goals"]) if isinstance(profile["investment_goals"], list) else profile["investment_goals"]
            parts.append(f"Goals: {goals}")
        return "User Profile:\n" + "\n".join(parts) if parts else ""
