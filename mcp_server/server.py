"""
MCP Server for Finnie AI Finance Assistant.
Exposes finance tools via Model Context Protocol for Claude Desktop integration.

Run with: python -m mcp_server.server
"""
import os
import sys
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We'll use the MCP SDK. Create server manually to avoid complex dependencies.
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp import types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not available. Install with: pip install mcp")

# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def get_stock_quote(symbol: str) -> Dict[str, Any]:
    """Get current stock quote for a symbol."""
    from src.utils.market_data import market_client
    return market_client.get_stock_info(symbol)

def get_market_overview() -> Dict[str, Any]:
    """Get current market overview with major indices."""
    from src.utils.market_data import market_client
    return market_client.get_market_overview()

def analyze_portfolio(holdings_json: str) -> Dict[str, Any]:
    """Analyze a portfolio given holdings as JSON string.

    holdings_json format: '[{"symbol": "AAPL", "shares": 10, "avg_cost": 150.0}]'
    """
    from src.utils.market_data import market_client
    from src.agents.portfolio_agent import calculate_portfolio_metrics

    holdings = json.loads(holdings_json)
    enriched = market_client.get_portfolio_data(holdings)
    metrics = calculate_portfolio_metrics(enriched)
    return {"enriched_portfolio": enriched, "metrics": metrics}

def ask_finance_question(question: str, user_experience: str = "beginner") -> str:
    """Ask a financial education question and get an AI-powered answer.

    user_experience: "beginner", "intermediate", or "advanced"
    """
    from src.rag.knowledge_base import get_vectorstore
    from src.rag.retriever import FinanceRAGRetriever
    from src.workflow.graph import get_workflow

    vectorstore = get_vectorstore()
    retriever = FinanceRAGRetriever(vectorstore)
    workflow = get_workflow(retriever=retriever)

    result = workflow.process_query(
        user_query=question,
        user_profile={"experience_level": user_experience, "risk_tolerance": "moderate", "investment_goals": [], "time_horizon": "long"}
    )
    return result["response"]

def calculate_goal_projection(
    goal_amount: float,
    current_savings: float,
    monthly_contribution: float,
    years: int,
    annual_return_pct: float = 7.0
) -> Dict[str, Any]:
    """Calculate financial goal projections."""
    from src.agents.goal_planning_agent import build_goal_projections

    projections = build_goal_projections(goal_amount, current_savings, monthly_contribution, years)
    return {
        "goal_amount": goal_amount,
        "current_savings": current_savings,
        "monthly_contribution": monthly_contribution,
        "years": years,
        "projections": projections
    }

def search_financial_knowledge(query: str, k: int = 5) -> List[Dict[str, str]]:
    """Search the financial knowledge base for relevant information."""
    from src.rag.knowledge_base import get_vectorstore
    from src.rag.retriever import FinanceRAGRetriever

    vectorstore = get_vectorstore()
    retriever = FinanceRAGRetriever(vectorstore)
    docs = retriever.retrieve(query, k=k)
    sources = retriever.get_sources(docs)

    results = []
    for doc, source in zip(docs, sources):
        results.append({
            "title": source.get("title", ""),
            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            "category": source.get("category", ""),
            "source": source.get("source", "")
        })
    return results

def get_financial_news(symbols: str = "", limit: int = 5) -> List[Dict[str, str]]:
    """Get latest financial news. symbols: comma-separated stock symbols (optional)."""
    from src.utils.market_data import market_client

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    return market_client.search_news(symbols=symbol_list, limit=limit)

# ============================================================
# MCP SERVER SETUP
# ============================================================

TOOLS = [
    {
        "name": "get_stock_quote",
        "description": "Get the current stock quote and key metrics for a stock symbol (e.g., AAPL, MSFT, SPY)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_market_overview",
        "description": "Get an overview of current market conditions including major indices (S&P 500, NASDAQ, Dow Jones, VIX)",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "analyze_portfolio",
        "description": "Analyze a stock portfolio given holdings. Provides metrics like diversification score, sector allocation, total value, and gain/loss.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "holdings_json": {
                    "type": "string",
                    "description": "JSON array of holdings: [{\"symbol\": \"AAPL\", \"shares\": 10, \"avg_cost\": 150.0}]"
                }
            },
            "required": ["holdings_json"]
        }
    },
    {
        "name": "ask_finance_question",
        "description": "Ask any financial education question. Uses RAG to provide accurate, educational answers about investing, markets, and personal finance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The financial question to ask"},
                "user_experience": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "User's experience level for tailored explanations",
                    "default": "beginner"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "calculate_goal_projection",
        "description": "Calculate financial goal projections across conservative/moderate/aggressive scenarios",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal_amount": {"type": "number", "description": "Target amount in dollars"},
                "current_savings": {"type": "number", "description": "Current savings in dollars"},
                "monthly_contribution": {"type": "number", "description": "Monthly contribution in dollars"},
                "years": {"type": "integer", "description": "Years to goal"},
                "annual_return_pct": {"type": "number", "description": "Expected annual return percentage (default 7)", "default": 7.0}
            },
            "required": ["goal_amount", "current_savings", "monthly_contribution", "years"]
        }
    },
    {
        "name": "search_financial_knowledge",
        "description": "Search Finnie's financial education knowledge base for information on specific topics",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "k": {"type": "integer", "description": "Number of results to return (default 5)", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_financial_news",
        "description": "Get the latest financial news, optionally filtered by stock symbols",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "string", "description": "Comma-separated stock symbols (optional, e.g., 'AAPL, MSFT')"},
                "limit": {"type": "integer", "description": "Number of news items (default 5)", "default": 5}
            }
        }
    }
]

def handle_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Handle a tool call and return result as JSON string."""
    try:
        if tool_name == "get_stock_quote":
            result = get_stock_quote(arguments["symbol"])
        elif tool_name == "get_market_overview":
            result = get_market_overview()
        elif tool_name == "analyze_portfolio":
            result = analyze_portfolio(arguments["holdings_json"])
        elif tool_name == "ask_finance_question":
            result = ask_finance_question(
                arguments["question"],
                arguments.get("user_experience", "beginner")
            )
            return result  # Already a string
        elif tool_name == "calculate_goal_projection":
            result = calculate_goal_projection(
                float(arguments["goal_amount"]),
                float(arguments["current_savings"]),
                float(arguments["monthly_contribution"]),
                int(arguments["years"]),
                float(arguments.get("annual_return_pct", 7.0))
            )
        elif tool_name == "search_financial_knowledge":
            result = search_financial_knowledge(
                arguments["query"],
                int(arguments.get("k", 5))
            )
        elif tool_name == "get_financial_news":
            result = get_financial_news(
                arguments.get("symbols", ""),
                int(arguments.get("limit", 5))
            )
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        return json.dumps({"error": str(e)})


async def run_mcp_server():
    """Run the MCP server using the MCP SDK."""
    if not MCP_AVAILABLE:
        logger.error("MCP SDK not available. Install with: pip install mcp")
        return

    server = Server("finnie-finance-assistant")

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"]
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        result = handle_tool_call(name, arguments)
        return [types.TextContent(type="text", text=result)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    if MCP_AVAILABLE:
        asyncio.run(run_mcp_server())
    else:
        print("MCP SDK not installed. To use the MCP server, install it with:")
        print("  pip install mcp")
        print("\nAvailable tools (for testing):")
        for tool in TOOLS:
            print(f"  - {tool['name']}: {tool['description']}")
