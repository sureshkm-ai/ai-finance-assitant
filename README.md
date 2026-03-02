# Finnie - AI Finance Assistant

Democratizing Financial Literacy Through Intelligent Conversational AI

A production-ready, multi-agent AI system for financial education and guidance. Built as a capstone project for the Applied Agentic AI for Software Engineers course (Interview Kickstart).

---

## Features

- **6 Specialized AI Agents** — each focused on a specific financial domain
- **RAG-Enhanced Responses** — grounded in a curated financial knowledge base (50+ articles + glossary)
- **Real-Time Market Data** — live stock quotes, indices, and news via yFinance + Alpha Vantage
- **Portfolio Analysis** — enriched holdings with metrics (diversification score, beta, sector allocation)
- **Goal Planning** — projection algorithms across conservative/moderate/aggressive scenarios
- **Streamlit Web UI** — 4-tab interface (Chat, Portfolio, Market, Goals)
- **LangGraph Orchestration** — intelligent query routing and state management
- **MCP Server** — Claude Desktop integration via Model Context Protocol (bonus)

---

## Architecture

```
User Query
    |
    v
┌─────────────────────────────────────────────────────────────┐
|                    LangGraph Workflow                        |
|                                                             |
|   ┌─────────┐   routes to   ┌─────────────────────────┐    |
|   | Router  | ───────────→  |   Specialized Agent      |    |
|   |  Node   |               |                         |    |
|   └─────────┘               |  ┌───────────────────┐  |    |
|                              |  |  Finance Q&A      |  |    |
|   Routing Rules:             |  |  Portfolio        |  |    |
|   - LLM classification       |  |  Market           |  |    |
|   - Keyword fallback         |  |  Goal Planning    |  |    |
|                              |  |  News Synthesizer |  |    |
|                              |  |  Tax Education    |  |    |
|                              |  └───────────────────┘  |    |
|                              |         |               |    |
|                              |         v               |    |
|                              |  ┌─────────────────┐   |    |
|                              |  |  RAG Retrieval  |   |    |
|                              |  |  (FAISS Index)  |   |    |
|                              |  └─────────────────┘   |    |
|                              └─────────────────────────┘    |
└─────────────────────────────────────────────────────────────┘
    |
    v
Streamlit UI / MCP Server
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Multi-Agent System | LangGraph StateGraph | Orchestrates 6 specialized agents |
| Language Model | Google Gemini 2.0 Flash | Powers NLU and generation |
| Vector Database | FAISS | Semantic search over knowledge base |
| Market Data | yFinance + Alpha Vantage | Real-time stock quotes and news |
| Web Interface | Streamlit + Plotly | Interactive 4-tab UI |
| State Management | LangGraph AgentState | Conversation context, user profiles |
| MCP Server | MCP SDK | Claude Desktop integration |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Google API Key (free at https://aistudio.google.com)
- Optional: Alpha Vantage API key (free at https://www.alphavantage.co)

### Installation

```bash
# 1. Navigate to project directory
cd ai_finance_assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Run the Streamlit app
streamlit run src/web_app/app.py
```

The app will open at http://localhost:8501

### First Run Notes
- On first launch, the app will build the FAISS knowledge base (~30 seconds)
- The index is cached to disk and reused on subsequent runs
- Make sure your GOOGLE_API_KEY is valid before starting

---

## Agents

### 1. Finance Q&A Agent
Handles general financial education queries using RAG-enhanced responses.
- **Triggers**: Questions about financial concepts, investment strategies, definitions
- **RAG**: Yes — retrieves from knowledge base of 65+ documents
- **Example**: "What is dollar-cost averaging?" → "How does compound interest work?"

### 2. Portfolio Analysis Agent
Analyzes investment portfolios with live market data enrichment.
- **Triggers**: "analyze my portfolio", "review my holdings", portfolio performance questions
- **Metrics**: Diversification score, sector allocation, weighted beta, gain/loss per position
- **Example**: "How diversified is my portfolio?" → "What are the risks in my holdings?"

### 3. Market Analysis Agent
Provides real-time market insights by fetching live data.
- **Triggers**: Stock price queries, market performance, index analysis
- **Data**: Live from yFinance — S&P 500, NASDAQ, Dow Jones, VIX, individual stocks
- **Example**: "How is AAPL doing today?" → "What are major indices doing?"

### 4. Goal Planning Agent
Assists with financial goal setting and projection algorithms.
- **Triggers**: Retirement planning, savings goals, "how much should I save", projection requests
- **Algorithms**: Future value, required monthly savings, 3-scenario projections (4/7/10%)
- **Example**: "How much do I need to save to reach $1M in 20 years?"

### 5. News Synthesizer Agent
Summarizes and contextualizes financial news.
- **Triggers**: "latest news", "what happened with", earnings inquiries
- **Data**: Fetches from Yahoo Finance via yFinance
- **Example**: "What's the latest news on tech stocks?"

### 6. Tax Education Agent
Explains tax concepts and account types with 2024 figures.
- **Triggers**: 401k, IRA, Roth, capital gains, tax-loss harvesting questions
- **Knowledge**: 2024 contribution limits, capital gains brackets, wash-sale rules
- **Example**: "Should I choose a Roth or Traditional IRA?"

---

## Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/            # 6 specialized agents + base class
│   ├── core/              # LLM config + LangGraph state
│   ├── data/
│   │   ├── articles/      # Financial education JSON articles
│   │   ├── glossary.json  # 50 financial term definitions
│   │   ├── sample_portfolios.json
│   │   └── faiss_index/   # Generated FAISS vector index (auto-created)
│   ├── rag/               # Knowledge base builder + RAG retriever
│   ├── utils/             # Market data client + TTL cache
│   ├── workflow/          # LangGraph router + StateGraph
│   └── web_app/           # Streamlit application
├── mcp_server/            # MCP server for Claude Desktop
├── tests/                 # pytest test suite
├── config.yaml            # Centralized configuration
├── requirements.txt
├── .env.example
└── README.md
```

---

## MCP Server (Claude Desktop Integration)

The MCP server exposes Finnie's capabilities directly in Claude Desktop.

### Setup

Add to Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "finnie": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/ai_finance_assistant",
      "env": {
        "GOOGLE_API_KEY": "your_key_here"
      }
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `get_stock_quote` | Current price, P/E ratio, beta, 52-week range |
| `get_market_overview` | Major indices (S&P 500, NASDAQ, Dow, VIX) |
| `analyze_portfolio` | Portfolio metrics with live data enrichment |
| `ask_finance_question` | RAG-powered financial education Q&A |
| `calculate_goal_projection` | Multi-scenario goal projections |
| `search_financial_knowledge` | Search the curated knowledge base |
| `get_financial_news` | Latest market news by symbol or general |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  model: "gemini-2.0-flash"
  temperature: 0.7
  rate_limit_rpm: 55

rag:
  chunk_size: 1000
  top_k: 5

market_data:
  cache_ttl_seconds: 1800  # 30-minute cache
```

---

## Troubleshooting

**"GOOGLE_API_KEY not found"**
→ Ensure `.env` file exists with `GOOGLE_API_KEY=your_key`

**"Failed to build FAISS index"**
→ Check that `src/data/articles/basics.json` exists and is valid JSON

**Slow first startup (~30s)**
→ Normal — the app is building the FAISS vector index from articles. Cached after first run.

**Market data shows "unavailable"**
→ yFinance may have rate limits during market hours. Data will retry with exponential backoff.

**Alpha Vantage errors**
→ Set `ALPHA_VANTAGE_API_KEY` in `.env` for the fallback to work.

---

## Disclaimer

This application is for educational purposes only and does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Google Gemini API](https://aistudio.google.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [yFinance API](https://github.com/ranaroussi/yfinance)
- [Investopedia](https://www.investopedia.com/) - Financial concepts reference

---

*Built with care as part of the Applied Agentic AI for SWEs capstone project — Interview Kickstart*
