"""LangGraph StateGraph: orchestrates the multi-agent workflow."""
import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from ..core.state import AgentState, UserProfile
from ..core.llm_config import get_llm
from ..workflow.router import create_router
from ..agents.finance_qa_agent import FinanceQAAgent
from ..agents.portfolio_agent import PortfolioAnalysisAgent
from ..agents.market_agent import MarketAnalysisAgent
from ..agents.goal_planning_agent import GoalPlanningAgent
from ..agents.news_agent import NewsSynthesizerAgent
from ..agents.tax_agent import TaxEducationAgent

logger = logging.getLogger(__name__)

def create_initial_state(
    user_query: str,
    conversation_history: list = None,
    user_profile: dict = None,
    portfolio_data: dict = None
) -> AgentState:
    """Create an initial AgentState from user input."""
    default_profile: UserProfile = {
        "risk_tolerance": "moderate",
        "experience_level": "beginner",
        "investment_goals": [],
        "time_horizon": "long"
    }
    return {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "agent_response": "",
        "agent_type": "router",
        "conversation_history": conversation_history or [],
        "user_profile": {**default_profile, **(user_profile or {})},
        "portfolio_data": portfolio_data,
        "error": None,
        "metadata": {},
        "needs_routing": True
    }

class FinanceWorkflow:
    """Multi-agent finance assistant workflow using LangGraph."""

    def __init__(self, retriever=None):
        self.llm = get_llm()
        self.retriever = retriever

        # Initialize all agents
        self.agents = {
            "finance_qa": FinanceQAAgent(self.llm, retriever),
            "portfolio": PortfolioAnalysisAgent(self.llm, retriever),
            "market": MarketAnalysisAgent(self.llm, retriever),
            "goal_planning": GoalPlanningAgent(self.llm, retriever),
            "news": NewsSynthesizerAgent(self.llm, retriever),
            "tax": TaxEducationAgent(self.llm, retriever),
        }

        self.router = create_router(self.llm)
        self.graph = self._build_graph()
        logger.info("FinanceWorkflow initialized with all agents")

    def _build_graph(self) -> Any:
        """Build the LangGraph StateGraph."""
        workflow = StateGraph(AgentState)

        # Add router node
        workflow.add_node("router", self.router)

        # Add agent nodes
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, agent.process)

        # Set entry point
        workflow.set_entry_point("router")

        # Add conditional edges from router to agents
        def route_to_agent(state: AgentState) -> str:
            agent_type = state.get("agent_type", "finance_qa")
            if agent_type in self.agents:
                return agent_type
            return "finance_qa"

        workflow.add_conditional_edges(
            "router",
            route_to_agent,
            {agent: agent for agent in self.agents.keys()}
        )

        # All agents go to END
        for agent_name in self.agents:
            workflow.add_edge(agent_name, END)

        return workflow.compile()

    def process_query(
        self,
        user_query: str,
        conversation_history: list = None,
        user_profile: dict = None,
        portfolio_data: dict = None
    ) -> Dict[str, Any]:
        """Process a user query through the workflow."""
        initial_state = create_initial_state(
            user_query=user_query,
            conversation_history=conversation_history or [],
            user_profile=user_profile,
            portfolio_data=portfolio_data
        )

        try:
            result = self.graph.invoke(initial_state)
            return {
                "response": result.get("agent_response", "I could not generate a response."),
                "agent_type": result.get("agent_type", "unknown"),
                "conversation_history": result.get("conversation_history", []),
                "metadata": result.get("metadata", {}),
                "error": result.get("error"),
                "state": result
            }
        except Exception as e:
            logger.error(f"Workflow processing failed: {e}")
            return {
                "response": f"I encountered an error processing your request. Please try again. Error: {str(e)}",
                "agent_type": "error",
                "conversation_history": conversation_history or [],
                "metadata": {"error": str(e)},
                "error": str(e)
            }

    def get_agent_info(self) -> Dict[str, str]:
        """Return info about available agents."""
        return {name: agent.agent_description for name, agent in self.agents.items()}


# Singleton workflow instance
_workflow_instance = None

def get_workflow(retriever=None) -> FinanceWorkflow:
    """Get or create the workflow singleton."""
    global _workflow_instance
    if _workflow_instance is None:
        _workflow_instance = FinanceWorkflow(retriever=retriever)
    return _workflow_instance
