"""
Finnie - AI Finance Assistant
Streamlit Multi-Tab Web Application
"""
import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Finnie - AI Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        background-color: #e8f4f8;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .positive { color: #28a745; font-weight: bold; }
    .negative { color: #dc3545; font-weight: bold; }
    .chat-message-user {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
    }
    .chat-message-assistant {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    defaults = {
        "conversation_history": [],
        "user_profile": {
            "risk_tolerance": "moderate",
            "experience_level": "beginner",
            "investment_goals": [],
            "time_horizon": "long"
        },
        "portfolio_holdings": [],
        "workflow": None,
        "retriever": None,
        "initialized": False,
        "market_data_cache": None,
        "market_data_timestamp": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# ============================================================
# LAZY INITIALIZATION (avoids rebuilding on every rerun)
# ============================================================
@st.cache_resource(show_spinner="Loading AI Finance Assistant (this takes ~30s on first run)...")
def initialize_system():
    """Initialize workflow and RAG — cached across sessions."""
    try:
        from src.rag.knowledge_base import get_vectorstore
        from src.rag.retriever import FinanceRAGRetriever
        from src.workflow.graph import get_workflow

        logger.info("Building FAISS knowledge base...")
        vectorstore = get_vectorstore()
        retriever = FinanceRAGRetriever(vectorstore)

        logger.info("Initializing FinanceWorkflow...")
        workflow = get_workflow(retriever=retriever)

        return workflow, retriever
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return None, None

# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 💰 Finnie")
        st.markdown("*AI Finance Assistant*")
        st.divider()

        st.markdown("### 👤 Your Profile")

        risk = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=["conservative", "moderate", "aggressive"].index(
                st.session_state.user_profile.get("risk_tolerance", "moderate")
            )
        )

        experience = st.selectbox(
            "Experience Level",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(
                st.session_state.user_profile.get("experience_level", "beginner")
            )
        )

        time_horizon = st.selectbox(
            "Time Horizon",
            ["short", "medium", "long"],
            index=["short", "medium", "long"].index(
                st.session_state.user_profile.get("time_horizon", "long")
            )
        )

        st.session_state.user_profile.update({
            "risk_tolerance": risk,
            "experience_level": experience,
            "time_horizon": time_horizon
        })

        st.divider()
        st.markdown("### Bot Agents")
        agents_info = {
            "Finance Q&A": "General financial education",
            "Portfolio": "Portfolio analysis & metrics",
            "Market": "Real-time market data",
            "Goal Planning": "Goal setting & projections",
            "News": "Financial news synthesis",
            "Tax Education": "Tax concepts & accounts"
        }
        for agent, desc in agents_info.items():
            st.markdown(f"**{agent}**  \n{desc}")

        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

        st.markdown("---")
        st.caption("For educational purposes only. Not financial advice.")

# ============================================================
# TAB 1: CHAT
# ============================================================
def render_chat_tab(workflow):
    st.markdown("## Chat with Finnie")
    st.markdown("Ask any financial question — I'll route it to the right specialist automatically.")

    # Suggested questions
    with st.expander("Suggested questions to try"):
        col1, col2 = st.columns(2)
        suggestions = [
            "What is dollar-cost averaging?",
            "Explain the difference between ETFs and mutual funds",
            "How does compound interest work?",
            "What is a good asset allocation for a beginner?",
            "What are the tax benefits of a Roth IRA?",
            "How do I evaluate if a stock is overvalued?",
        ]
        for i, q in enumerate(suggestions):
            col = col1 if i % 2 == 0 else col2
            if col.button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_query"] = q

    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.conversation_history:
            st.info("Hi! I'm Finnie, your AI Finance Assistant. Ask me anything about investing, portfolios, markets, taxes, or financial planning!")
        else:
            for turn in st.session_state.conversation_history[-20:]:
                with st.chat_message("user"):
                    st.write(turn.get("user", ""))
                with st.chat_message("assistant"):
                    agent = turn.get("agent_type", "")
                    if agent:
                        agent_labels = {
                            "finance_qa": "Finance Q&A",
                            "portfolio": "Portfolio Analysis",
                            "market": "Market Analysis",
                            "goal_planning": "Goal Planning",
                            "news": "News Synthesizer",
                            "tax": "Tax Education"
                        }
                        st.caption(f"Agent: {agent_labels.get(agent, agent)}")
                    st.markdown(turn.get("assistant", ""))

    # Chat input
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask a financial question...") or pending

    if user_input and workflow:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = workflow.process_query(
                    user_query=user_input,
                    conversation_history=st.session_state.conversation_history,
                    user_profile=st.session_state.user_profile,
                    portfolio_data={"holdings": st.session_state.portfolio_holdings} if st.session_state.portfolio_holdings else None
                )

            agent_type = result.get("agent_type", "")
            agent_labels = {
                "finance_qa": "Finance Q&A",
                "portfolio": "Portfolio Analysis",
                "market": "Market Analysis",
                "goal_planning": "Goal Planning",
                "news": "News Synthesizer",
                "tax": "Tax Education"
            }
            st.caption(f"Agent: {agent_labels.get(agent_type, agent_type)}")
            st.markdown(result["response"])

            # Update conversation history
            st.session_state.conversation_history = result.get("conversation_history", [])

    elif user_input and not workflow:
        st.error("System not initialized. Please set your GOOGLE_API_KEY in .env file.")

# ============================================================
# TAB 2: PORTFOLIO
# ============================================================
def render_portfolio_tab(workflow):
    st.markdown("## Portfolio Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Enter Your Holdings")
        st.caption("Enter your portfolio holdings below. Format: SYMBOL, shares, avg_cost_per_share")

        # Portfolio input
        default_input = "\n".join([
            f"{h.get('symbol', '')}, {h.get('shares', 0)}, {h.get('avg_cost', 0)}"
            for h in st.session_state.portfolio_holdings
        ]) if st.session_state.portfolio_holdings else "AAPL, 10, 150.00\nMSFT, 5, 310.00\nSPY, 20, 420.00\nBND, 30, 72.00"

        holdings_text = st.text_area(
            "Holdings (one per line: SYMBOL, shares, avg_cost)",
            value=default_input,
            height=200
        )

        col_load, col_sample = st.columns(2)

        if col_load.button("Analyze Portfolio", type="primary", use_container_width=True):
            holdings = []
            for line in holdings_text.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        holdings.append({
                            "symbol": parts[0].upper(),
                            "shares": float(parts[1]),
                            "avg_cost": float(parts[2])
                        })
                    except ValueError:
                        st.warning(f"Skipping invalid line: {line}")

            if holdings:
                st.session_state.portfolio_holdings = holdings
                st.rerun()

        if col_sample.button("Load Sample Portfolio", use_container_width=True):
            try:
                import json
                data_path = os.path.join(os.path.dirname(__file__), "../../data/sample_portfolios.json")
                with open(data_path) as f:
                    data = json.load(f)
                sample = data["sample_portfolios"][1]  # Growth portfolio
                st.session_state.portfolio_holdings = sample["holdings"]
                st.success(f"Loaded: {sample['name']}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load sample: {e}")

    with col2:
        st.markdown("### Quick Actions")
        portfolio_questions = [
            "How diversified is my portfolio?",
            "What are the risks in my portfolio?",
            "How can I improve my portfolio allocation?",
            "Analyze my portfolio performance",
        ]
        for q in portfolio_questions:
            if st.button(q, use_container_width=True, key=f"pq_{q[:20]}"):
                st.session_state["portfolio_question"] = q

    # Display portfolio if holdings exist
    if st.session_state.portfolio_holdings:
        from src.utils.market_data import market_client

        with st.spinner("Fetching live market data..."):
            enriched = market_client.get_portfolio_data(st.session_state.portfolio_holdings)

        holdings_df = enriched.get("holdings", [])

        if holdings_df:
            # Summary metrics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            total_val = enriched.get("total_value", 0)
            gain_loss = enriched.get("total_gain_loss", 0)
            gain_pct = enriched.get("total_gain_loss_pct", 0)

            col1.metric("Total Value", f"${total_val:,.2f}")
            col2.metric("Total Cost Basis", f"${enriched.get('total_cost', 0):,.2f}")
            col3.metric("Total Gain/Loss", f"${gain_loss:+,.2f}", f"{gain_pct:+.1f}%",
                        delta_color="normal" if gain_loss >= 0 else "inverse")
            col4.metric("Holdings", enriched.get("num_holdings", 0))

            # Charts row
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)

            # Pie chart: portfolio allocation
            with chart_col1:
                st.markdown("#### Portfolio Allocation")
                labels = [h.get("symbol") for h in holdings_df]
                values = [h.get("current_value", 0) for h in holdings_df]
                fig = px.pie(
                    values=values, names=labels,
                    title="Holdings by Value",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            # Bar chart: gain/loss per holding
            with chart_col2:
                st.markdown("#### Gain/Loss by Holding")
                symbols = [h.get("symbol") for h in holdings_df]
                gains = [h.get("gain_loss_pct", 0) for h in holdings_df]
                colors = ["green" if g >= 0 else "red" for g in gains]
                fig = go.Figure(go.Bar(
                    x=symbols, y=gains,
                    marker_color=colors,
                    text=[f"{g:+.1f}%" for g in gains],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Return % per Position",
                    xaxis_title="Symbol",
                    yaxis_title="Return (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Sector allocation chart
            sector_data = {}
            for h in holdings_df:
                sector = h.get("sector", "Unknown")
                sector_data[sector] = sector_data.get(sector, 0) + h.get("current_value", 0)

            if sector_data:
                st.markdown("#### Sector Allocation")
                fig = px.bar(
                    x=list(sector_data.keys()),
                    y=list(sector_data.values()),
                    title="Portfolio Value by Sector",
                    labels={"x": "Sector", "y": "Value ($)"},
                    color=list(sector_data.values()),
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Holdings table
            st.markdown("#### Holdings Detail")
            display_data = []
            for h in holdings_df:
                gain_pct_h = h.get("gain_loss_pct", 0)
                display_data.append({
                    "Symbol": h.get("symbol", ""),
                    "Name": h.get("name", "")[:25],
                    "Shares": h.get("shares", 0),
                    "Avg Cost": f"${h.get('avg_cost', 0):.2f}",
                    "Current Price": f"${h.get('current_price', 0):.2f}",
                    "Value": f"${h.get('current_value', 0):,.2f}",
                    "Gain/Loss": f"${h.get('gain_loss', 0):+,.2f}",
                    "Return %": f"{gain_pct_h:+.1f}%",
                    "Sector": h.get("sector", "N/A"),
                    "Today": f"{h.get('change_pct', 0):+.2f}%"
                })
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

            # AI Analysis
            st.markdown("---")
            portfolio_q = st.session_state.pop("portfolio_question", None)
            analyze_query = portfolio_q or "Please provide a comprehensive analysis of this portfolio."

            if portfolio_q or st.button("Get AI Portfolio Analysis", type="primary"):
                if workflow:
                    with st.spinner("Analyzing your portfolio..."):
                        result = workflow.process_query(
                            user_query=analyze_query,
                            conversation_history=st.session_state.conversation_history,
                            user_profile=st.session_state.user_profile,
                            portfolio_data={"holdings": st.session_state.portfolio_holdings}
                        )
                    st.markdown("### AI Analysis")
                    st.markdown(result["response"])
                    st.session_state.conversation_history = result.get("conversation_history", [])

# ============================================================
# TAB 3: MARKET
# ============================================================
def render_market_tab(workflow):
    st.markdown("## Market Overview")

    from src.utils.market_data import market_client

    # Auto-refresh button
    col1, col2 = st.columns([3, 1])
    col1.markdown("*Real-time market data*")
    if col2.button("Refresh Data"):
        st.rerun()

    with st.spinner("Loading market data..."):
        overview = market_client.get_market_overview()

    indices = overview.get("indices", {})
    market_status = overview.get("market_status", "unknown")

    # Market status badge
    status_color = "open" if market_status == "open" else "closed"
    st.markdown(f"**Market Status:** {status_color.upper()}")

    # Indices
    st.markdown("---")
    st.markdown("### Major Indices")
    idx_cols = st.columns(len(indices)) if indices else st.columns(1)
    for i, (name, data) in enumerate(indices.items()):
        if "error" not in data:
            change_pct = data.get("change_pct", 0)
            idx_cols[i].metric(
                name,
                f"{data.get('value', 0):,.2f}",
                f"{change_pct:+.2f}%",
                delta_color="normal" if change_pct >= 0 else "inverse"
            )

    # Stock lookup
    st.markdown("---")
    st.markdown("### Stock Lookup")

    col1, col2 = st.columns([2, 1])
    with col1:
        lookup_symbols = st.text_input("Enter stock symbols (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, NVDA")

    if st.button("Fetch Quotes", type="primary"):
        symbols = [s.strip().upper() for s in lookup_symbols.split(",") if s.strip()]
        if symbols:
            with st.spinner("Fetching quotes..."):
                quotes = market_client.get_multiple_quotes(symbols)

            quote_data = []
            for symbol, data in quotes.items():
                if "error" not in data:
                    change_pct = data.get("change_pct", 0)
                    direction = "up" if change_pct >= 0 else "down"
                    quote_data.append({
                        "Symbol": symbol,
                        "Name": data.get("name", symbol)[:30],
                        "Price": f"${data.get('current_price', 0):,.2f}",
                        "Change": f"{change_pct:+.2f}%",
                        "Trend": direction,
                        "52W High": f"${data.get('52_week_high', 0):,.2f}" if data.get('52_week_high') else "N/A",
                        "52W Low": f"${data.get('52_week_low', 0):,.2f}" if data.get('52_week_low') else "N/A",
                        "P/E": f"{data.get('pe_ratio', 'N/A')}",
                        "Beta": f"{data.get('beta', 'N/A')}",
                        "Sector": data.get("sector", "N/A")
                    })

            if quote_data:
                st.dataframe(pd.DataFrame(quote_data), use_container_width=True, hide_index=True)

                # Price chart for first symbol
                first_symbol = list(quotes.keys())[0]
                with st.spinner(f"Loading {first_symbol} price chart..."):
                    hist = market_client.get_historical_data(first_symbol, "6mo")
                if not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=hist["Close"],
                        mode="lines",
                        name=first_symbol,
                        line=dict(color="#1f77b4", width=2)
                    ))
                    fig.update_layout(
                        title=f"{first_symbol} - 6 Month Price History",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Market AI analysis
    st.markdown("---")
    st.markdown("### Ask About the Market")
    market_questions = [
        "What is the market doing today?",
        "Explain what VIX tells us about market sentiment",
        "How does interest rate changes affect stocks?",
        "What sectors are typically defensive investments?",
    ]

    selected_q = st.selectbox("Quick questions:", ["Custom question..."] + market_questions)
    if selected_q == "Custom question...":
        market_query = st.text_input("Enter your market question:")
    else:
        market_query = selected_q

    if market_query and st.button("Ask Market Agent", type="primary") and workflow:
        with st.spinner("Analyzing market..."):
            result = workflow.process_query(
                user_query=market_query,
                conversation_history=st.session_state.conversation_history,
                user_profile=st.session_state.user_profile
            )
        st.markdown("### Response")
        st.markdown(result["response"])
        st.session_state.conversation_history = result.get("conversation_history", [])

    # Financial News
    st.markdown("---")
    st.markdown("### Latest Financial News")
    if st.button("Load Latest News"):
        with st.spinner("Fetching news..."):
            news = market_client.search_news(limit=8)

        for item in news:
            with st.expander(f"{item.get('title', 'No title')[:80]}..."):
                st.caption(f"Source: {item.get('source', 'Unknown')} | Published: {item.get('published', 'Unknown')}")
                st.write(item.get('summary', item.get('title', '')))
                if item.get('url'):
                    st.markdown(f"[Read full article]({item['url']})")

# ============================================================
# TAB 4: GOALS
# ============================================================
def render_goals_tab(workflow):
    st.markdown("## Financial Goal Planning")
    st.markdown("Plan your path to financial goals with projections and personalized guidance.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Set Your Goal")

        goal_type = st.selectbox("Goal Type", [
            "Retirement",
            "Home Purchase",
            "Education Fund",
            "Emergency Fund",
            "Wealth Building",
            "Custom Goal"
        ])

        goal_name = st.text_input("Goal Name", goal_type)
        goal_amount = st.number_input("Target Amount ($)", min_value=1000, value=500000, step=5000)
        current_savings = st.number_input("Current Savings ($)", min_value=0, value=10000, step=1000)
        monthly_contribution = st.number_input("Monthly Contribution ($)", min_value=0, value=500, step=50)
        years_to_goal = st.slider("Years to Goal", min_value=1, max_value=40, value=20)

        risk_for_goal = st.selectbox("Risk Tolerance for this Goal", ["conservative", "moderate", "aggressive"])

    with col2:
        st.markdown("### Projections")

        # Calculate projections
        def calc_fv(pv, r, y, pmt):
            if r == 0:
                return pv + pmt * 12 * y
            monthly_r = r / 100 / 12
            months = y * 12
            return pv * (1 + r/100)**y + pmt * (((1 + monthly_r)**months - 1) / monthly_r)

        scenarios = {"Conservative (4%)": 4.0, "Moderate (7%)": 7.0, "Aggressive (10%)": 10.0}

        projection_data = []
        for scenario, rate in scenarios.items():
            fv = calc_fv(current_savings, rate, years_to_goal, monthly_contribution)
            req_monthly = 0
            if fv < goal_amount:
                monthly_r = rate / 100 / 12
                months = years_to_goal * 12
                remaining = goal_amount - current_savings * (1 + rate/100)**years_to_goal
                req_monthly = remaining * monthly_r / ((1 + monthly_r)**months - 1) if monthly_r > 0 else remaining / months

            on_track = fv >= goal_amount
            projection_data.append({
                "Scenario": scenario,
                "Projected": fv,
                "Required Monthly": max(0, req_monthly),
                "On Track": on_track
            })

        for p in projection_data:
            status = "On Track" if p["On Track"] else f"Need ${p['Required Monthly']:,.0f}/mo"
            st.metric(
                p["Scenario"],
                f"${p['Projected']:,.0f}",
                status,
                delta_color="normal" if p["On Track"] else "inverse"
            )

        # Projection chart
        years_range = list(range(0, years_to_goal + 1))
        fig = go.Figure()

        colors = {"Conservative (4%)": "#1f77b4", "Moderate (7%)": "#ff7f0e", "Aggressive (10%)": "#2ca02c"}
        for scenario, rate in scenarios.items():
            values = [calc_fv(current_savings, rate, y, monthly_contribution) for y in years_range]
            fig.add_trace(go.Scatter(
                x=years_range, y=values,
                mode="lines",
                name=scenario,
                line=dict(color=colors[scenario], width=2)
            ))

        # Goal line
        fig.add_hline(y=goal_amount, line_dash="dash", line_color="red",
                      annotation_text=f"Goal: ${goal_amount:,.0f}")

        fig.update_layout(
            title=f"Path to ${goal_amount:,.0f} Goal",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)

    # AI Goal Planning
    st.markdown("---")
    st.markdown("### Get Personalized Goal Plan")

    if st.button("Generate AI Goal Plan", type="primary") and workflow:
        goal_query = f"""Help me plan for my {goal_name} goal.
        I want to reach ${goal_amount:,.0f} in {years_to_goal} years.
        I currently have ${current_savings:,.0f} saved and can contribute ${monthly_contribution:,.0f} per month.
        My risk tolerance is {risk_for_goal}.
        Please provide a detailed action plan."""

        with st.spinner("Creating your personalized plan..."):
            result = workflow.process_query(
                user_query=goal_query,
                conversation_history=st.session_state.conversation_history,
                user_profile={**st.session_state.user_profile, "risk_tolerance": risk_for_goal}
            )

        st.markdown("### Your Personalized Plan")
        st.markdown(result["response"])
        st.session_state.conversation_history = result.get("conversation_history", [])

    # Common goal questions
    st.markdown("---")
    st.markdown("### Common Goal Planning Questions")
    common_qs = [
        "How much should I save for retirement?",
        "What's the 4% withdrawal rule?",
        "How does inflation affect long-term goals?",
        "When should I start saving for retirement?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(common_qs):
        col = cols[i % 2]
        if col.button(q, use_container_width=True, key=f"gq_{i}") and workflow:
            with st.spinner("Thinking..."):
                result = workflow.process_query(
                    user_query=q,
                    conversation_history=st.session_state.conversation_history,
                    user_profile=st.session_state.user_profile
                )
            st.markdown("### Answer")
            st.markdown(result["response"])
            st.session_state.conversation_history = result.get("conversation_history", [])

# ============================================================
# MAIN APP
# ============================================================
def main():
    init_session_state()
    render_sidebar()

    st.markdown('<div class="main-header">Finnie - AI Finance Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your personalized financial education companion powered by multi-agent AI</div>', unsafe_allow_html=True)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found. Please create a .env file with your API key.")
        st.code("GOOGLE_API_KEY=your_api_key_here", language="bash")
        st.info("Get your free API key at: https://aistudio.google.com/")
        st.stop()

    # Initialize system
    workflow, retriever = initialize_system()

    if workflow is None:
        st.error("Failed to initialize the AI system. Check your GOOGLE_API_KEY and try again.")
        st.stop()

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Portfolio", "Market", "Goals"])

    with tab1:
        render_chat_tab(workflow)
    with tab2:
        render_portfolio_tab(workflow)
    with tab3:
        render_market_tab(workflow)
    with tab4:
        render_goals_tab(workflow)

if __name__ == "__main__":
    main()
