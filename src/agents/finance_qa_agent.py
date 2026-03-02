"""Finance Q&A Agent: Handles general financial education queries with RAG."""
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseFinanceAgent
from ..core.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Finnie, a friendly and knowledgeable financial education assistant.
Your role is to explain financial concepts clearly and accessibly, especially to beginners.

Guidelines:
- Use simple, jargon-free language unless explaining a specific term
- Provide concrete examples and analogies
- Always relate concepts to practical application
- Be encouraging and supportive
- Structure longer answers with clear sections
- When relevant, mention related concepts the user might want to explore
- ALWAYS clarify that this is educational information, not personalized financial advice

{user_profile_context}
{conversation_context}
"""

class FinanceQAAgent(BaseFinanceAgent):
    """Handles general financial education queries using RAG-enhanced responses."""

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_name = "Finance Q&A Agent"
        self.agent_description = "Handles general financial education queries"

    def process(self, state: AgentState) -> AgentState:
        try:
            query = state.get("user_query", "")

            # Get relevant context from knowledge base
            context = self.get_context(query, k=5)
            user_profile_ctx = self.get_user_profile_context(state)
            conversation_ctx = self.get_conversation_context(state)

            system_content = SYSTEM_PROMPT.format(
                user_profile_context=user_profile_ctx,
                conversation_context=f"Recent conversation:\n{conversation_ctx}" if conversation_ctx else ""
            )

            user_content = f"""Please answer this financial question:

{query}

Background knowledge from our financial education database:
{context}

Provide a clear, educational response. If the background knowledge is directly relevant, use it to inform your answer and cite the source."""

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content)
            ]

            response = self.llm.invoke(messages)
            return self.build_response(state, response.content, {"rag_used": bool(context)})

        except Exception as e:
            return self.handle_error(state, e)
