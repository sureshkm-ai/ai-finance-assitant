"""Goal Planning Agent: Financial goal setting with projection algorithms."""
import logging
import math
import re
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import BaseFinanceAgent
from ..core.state import AgentState

logger = logging.getLogger(__name__)

def calculate_future_value(pv: float, rate: float, years: int, pmt: float = 0) -> float:
    """Calculate future value with optional regular payments."""
    if rate == 0:
        return pv + pmt * years
    r = rate / 100
    fv_principal = pv * (1 + r) ** years
    fv_payments = pmt * (((1 + r) ** years - 1) / r) if pmt else 0
    return fv_principal + fv_payments

def calculate_required_monthly_savings(goal_amount: float, current_savings: float,
                                         annual_return: float, years: int) -> float:
    """Calculate monthly savings needed to reach a goal."""
    fv_current = calculate_future_value(current_savings, annual_return, years)
    remaining = goal_amount - fv_current
    if remaining <= 0:
        return 0
    monthly_rate = annual_return / 100 / 12
    months = years * 12
    if monthly_rate == 0:
        return remaining / months
    pmt = remaining / (((1 + monthly_rate) ** months - 1) / monthly_rate)
    return max(0, pmt)

def build_goal_projections(goal_amount: float, current_savings: float,
                            monthly_contribution: float, years: int) -> Dict[str, Any]:
    """Build projections for different return scenarios."""
    scenarios = {
        "conservative (4%)": 4.0,
        "moderate (7%)": 7.0,
        "aggressive (10%)": 10.0
    }
    projections = {}
    for scenario, annual_rate in scenarios.items():
        monthly_rate = annual_rate / 100 / 12
        months = years * 12
        fv = calculate_future_value(current_savings, annual_rate, years, monthly_contribution * 12)
        req_monthly = calculate_required_monthly_savings(goal_amount, current_savings, annual_rate, years)
        on_track = fv >= goal_amount
        projections[scenario] = {
            "projected_value": round(fv, 2),
            "goal_amount": goal_amount,
            "on_track": on_track,
            "shortfall": round(max(0, goal_amount - fv), 2),
            "required_monthly_savings": round(req_monthly, 2)
        }
    return projections

SYSTEM_PROMPT = """You are Finnie's Goal Planning specialist. You help users set and plan toward financial goals using projection algorithms.

Your approach:
- Help users clarify and quantify their goals (SMART goals)
- Explain the math behind projections in simple terms
- Consider their risk tolerance and time horizon
- Discuss realistic return expectations for different asset allocations
- Break down large goals into actionable monthly steps
- Educate about the importance of starting early (compounding)

{user_profile_context}
"""

class GoalPlanningAgent(BaseFinanceAgent):
    """Assists with financial goal setting and projections."""

    def __init__(self, llm, retriever=None):
        super().__init__(llm, retriever)
        self.agent_name = "Goal Planning Agent"
        self.agent_description = "Assists with financial goal setting and planning"

    def process(self, state: AgentState) -> AgentState:
        try:
            query = state.get("user_query", "").lower()
            user_profile_ctx = self.get_user_profile_context(state)
            rag_context = self.get_context("financial goal planning investment strategy retirement savings", k=4)

            # Extract numbers from query for projections (simple heuristic)
            amounts = re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*(?:k|thousand|million|m)?\b', query, re.IGNORECASE)
            years_match = re.findall(r'(\d+)\s*(?:year|yr)', query, re.IGNORECASE)

            projections_context = ""
            if amounts and years_match:
                try:
                    goal = float(amounts[0].replace(",", ""))
                    if "k" in query or "thousand" in query:
                        goal *= 1000
                    elif "million" in query or "m" in query:
                        goal *= 1000000
                    years = int(years_match[0])

                    current = 0
                    if len(amounts) > 1:
                        current = float(amounts[1].replace(",", ""))

                    monthly = 500  # default assumption
                    monthly_match = re.findall(r'\$?([\d,]+)\s*(?:per month|monthly|/month)', query, re.IGNORECASE)
                    if monthly_match:
                        monthly = float(monthly_match[0].replace(",", ""))

                    projections = build_goal_projections(goal, current, monthly, years)

                    projections_context = f"\n\nGoal Projections (Goal: ${goal:,.0f} in {years} years, Current savings: ${current:,.0f}, Monthly contribution: ${monthly:,.0f}/month):\n"
                    for scenario, data in projections.items():
                        status = "✅ ON TRACK" if data["on_track"] else f"⚠️ ${data['shortfall']:,.0f} short (need ${data['required_monthly_savings']:,.0f}/month)"
                        projections_context += f"  - {scenario}: Projected ${data['projected_value']:,.0f} | {status}\n"
                except Exception as calc_err:
                    logger.warning(f"Projection calculation failed: {calc_err}")

            messages = [
                SystemMessage(content=SYSTEM_PROMPT.format(user_profile_context=user_profile_ctx)),
                HumanMessage(content=f"""User goal planning question: {query}

{projections_context}

Educational context:
{rag_context}

Help the user plan toward their financial goals. Explain the projections if calculated, discuss realistic expectations, and provide a clear action plan. Use the compound interest and savings strategies from the educational context.""")
            ]

            response = self.llm.invoke(messages)
            return self.build_response(state, response.content, {"projections_calculated": bool(projections_context)})

        except Exception as e:
            return self.handle_error(state, e)
