"""News Synthesizer Agent: Summarizes and contextualizes financial news."""
import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseFinanceAgent
from ..core.state import AgentState
from ..utils.market_data import market_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Finnie's News Synthesis specialist. You summarize and contextualize financial news to help users understand market developments.

Your role:
- Summarize news in accessible language
- Explain the potential implications for investors
- Put news in broader market context
- Help users understand why news matters (or doesn't)
- Avoid sensationalism and maintain educational tone
- Note that past performance doesn't guarantee future results

{user_profile_context}
"""

class NewsSynthesizerAgent(BaseFinanceAgent):
    """Summarizes and contextualizes financial news."""

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_name = "News Synthesizer Agent"
        self.agent_description = "Summarizes and contextualizes financial news"

    def process(self, state: AgentState) -> AgentState:
        try:
            query = state.get("user_query", "")
            user_profile_ctx = self.get_user_profile_context(state)

            symbols_in_query = re.findall(r'\b[A-Z]{2,5}\b', query)
            stop_words = {"A", "I", "IT", "IS", "IN", "AT", "TO", "OR", "AND", "THE", "FOR", "NEWS", "WHAT", "HOW"}
            symbols = [s for s in symbols_in_query if s not in stop_words][:3]

            news_items = market_client.search_news(query=query, symbols=symbols if symbols else None, limit=8)

            news_context = "Recent Financial News:\n"
            if news_items:
                for i, item in enumerate(news_items, 1):
                    news_context += f"\n{i}. **{item.get('title', 'No title')}**\n"
                    news_context += f"   Published: {item.get('published', 'Unknown')} | Source: {item.get('source', 'Unknown')}\n"
                    if item.get('summary') and item['summary'] != item.get('title'):
                        news_context += f"   Summary: {item.get('summary', '')[:200]}\n"
            else:
                news_context += "No recent news found for this query.\n"

            rag_context = self.get_context(query, k=2)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(user_profile_context=user_profile_ctx)),
                HumanMessage(content=f"""User request: {query}

{news_context}

Background financial context:
{rag_context}

Synthesize the news items and explain their significance. Help the user understand what this news means for the market and for individual investors. Keep the tone educational and balanced.""")
            ]

            response = self.llm.invoke(messages)
            return self.build_response(state, response.content, {
                "news_fetched": True,
                "num_news_items": len(news_items),
                "symbols_analyzed": symbols
            })

        except Exception as e:
            return self.handle_error(state, e)
