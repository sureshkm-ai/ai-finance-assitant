"""Tax Education Agent: Explains tax concepts and account types."""
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseFinanceAgent
from ..core.state import AgentState

logger = logging.getLogger(__name__)

TAX_KNOWLEDGE = """
Key Tax Concepts for Investors (2024):

ACCOUNT TYPES:
- Traditional 401(k): Pre-tax contributions, tax-deferred growth, taxed on withdrawal. 2024 limit: $23,000 ($30,500 if 50+)
- Roth 401(k): After-tax contributions, tax-free growth and withdrawals. Same limits as traditional.
- Traditional IRA: Pre-tax contributions (if eligible), tax-deferred growth. 2024 limit: $7,000 ($8,000 if 50+)
- Roth IRA: After-tax contributions, tax-free growth/withdrawals. Income limits apply (2024: phase-out $146k-$161k single, $230k-$240k married)
- HSA: Triple tax advantage for medical expenses. 2024 limit: $4,150 individual, $8,300 family
- 529 Plan: Tax-free growth for education expenses

CAPITAL GAINS TAXES (2024):
- Short-term (held <1 year): Taxed as ordinary income (10-37% depending on bracket)
- Long-term (held >1 year): 0%, 15%, or 20% depending on income
- Long-term 0% rate: Up to $47,025 (single) or $94,050 (married) in taxable income
- Long-term 15% rate: Up to $518,900 (single) or $583,750 (married)
- Long-term 20% rate: Above those thresholds
- Net Investment Income Tax (NIIT): 3.8% extra on investment income for high earners (above $200k single / $250k married)

LOSS HARVESTING RULES:
- Wash-sale rule: Cannot buy same/substantially identical security within 30 days before/after loss sale
- Capital losses first offset capital gains; excess up to $3,000/year deductible against ordinary income
- Unused losses carried forward indefinitely

DIVIDEND TAXES:
- Qualified dividends: Taxed at long-term capital gains rates (0/15/20%)
- Ordinary dividends: Taxed as ordinary income
- Dividends from REITs are mostly ordinary income
"""

SYSTEM_PROMPT = """You are Finnie's Tax Education specialist. You explain investment-related tax concepts clearly and help users understand how taxes affect their investment decisions.

IMPORTANT: You provide tax EDUCATION only, not tax ADVICE. Always recommend consulting a CPA or tax professional for personal tax situations.

Your role:
- Explain tax concepts related to investing in plain language
- Compare different account types and their tax implications
- Educate about capital gains, dividends, and tax-efficient investing
- Help users understand tax-loss harvesting basics
- Discuss general strategies but always defer to professionals for specifics

{user_profile_context}

Reference tax information:
{tax_knowledge}
"""

class TaxEducationAgent(BaseFinanceAgent):
    """Explains tax concepts related to investing."""

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_name = "Tax Education Agent"
        self.agent_description = "Explains tax concepts and account types"

    def process(self, state: AgentState) -> AgentState:
        try:
            query = state.get("user_query", "")
            user_profile_ctx = self.get_user_profile_context(state)
            rag_context = self.get_context(query + " tax accounts retirement", k=4)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(
                    user_profile_context=user_profile_ctx,
                    tax_knowledge=TAX_KNOWLEDGE
                )),
                HumanMessage(content=f"""Tax education question: {query}

Additional educational context from knowledge base:
{rag_context}

Explain the tax concepts clearly. Reference specific numbers and rules from the tax reference information. Always remind the user to consult a tax professional for their specific situation.""")
            ]

            response = self.llm.invoke(messages)
            return self.build_response(state, response.content, {"tax_education": True})

        except Exception as e:
            return self.handle_error(state, e)
