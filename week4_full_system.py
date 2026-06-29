# ================================================================
# WEEK 4 — The Full Connected System
#
# Connects all 4 agents into ONE LangGraph:
#   Planner → RAG Agent → Data Agent → Analyst
#
# One question flows through all agents, each using REAL data,
# producing a complete investment analysis.
# ================================================================

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import yfinance as yf

# ── SHARED RESOURCES ─────────────────────────────────────────────
print("🔧 Initialising system...")

llm = ChatOllama(model="llama3.2", temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings,
    collection_name="microsoft_10k"
)
print("✅ LLM + ChromaDB ready")

# ── STATE — the shared notepad for all 4 agents ──────────────────
class ResearchState(TypedDict):
    question:    str        # user's original question
    ticker:      str        # stock ticker (e.g. MSFT)
    plan:        List[str]  # planner's task breakdown
    rag_results: str        # RAG agent's findings from 10-K
    live_data:   dict       # data agent's live market data
    final_report: str       # analyst's final investment brief

# ── AGENT 1: PLANNER ─────────────────────────────────────────────
def planner_agent(state: ResearchState) -> ResearchState:
    print("\n🧠 PLANNER AGENT")
    prompt = f"""You are a research planner.
Question: {state['question']}

Break this into 3 specific research tasks for analysing this stock.
Format as a numbered list."""

    response = llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    tasks = [l.strip() for l in lines if l.strip() and l[0].isdigit()]

    print(f"   ✅ Created {len(tasks)} research tasks")
    return {"plan": tasks}

# ── AGENT 2: RAG AGENT ───────────────────────────────────────────
def rag_agent(state: ResearchState) -> ResearchState:
    print("\n🔍 RAG AGENT — reading 10-K filing")

    # Retrieve relevant chunks for the question
    docs = vectorstore.similarity_search(state['question'], k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are a financial analyst. Use ONLY the context below.
Cite specific numbers when available.

CONTEXT FROM 10-K:
{context}

QUESTION: {state['question']}

Provide key findings from the filing:"""

    response = llm.invoke(prompt)
    print(f"   ✅ Retrieved insights from {len(docs)} passages")
    return {"rag_results": response.content}

# ── AGENT 3: DATA AGENT ──────────────────────────────────────────
def data_agent(state: ResearchState) -> ResearchState:
    print(f"\n📊 DATA AGENT — fetching live data for {state['ticker']}")

    stock = yf.Ticker(state['ticker'])
    info  = stock.info

    data = {
        "company":       info.get("longName", "N/A"),
        "current_price": info.get("currentPrice", "N/A"),
        "market_cap":    info.get("marketCap", "N/A"),
        "pe_ratio":      info.get("trailingPE", "N/A"),
        "forward_pe":    info.get("forwardPE", "N/A"),
        "52w_high":      info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low":       info.get("fiftyTwoWeekLow", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
    }

    print(f"   ✅ Live price: ${data['current_price']}")
    return {"live_data": data}

# ── AGENT 4: ANALYST ─────────────────────────────────────────────
def analyst_agent(state: ResearchState) -> ResearchState:
    print("\n📝 ANALYST AGENT — writing final report")

    d = state['live_data']
    live_summary = f"""
Company: {d['company']}
Current Price: ${d['current_price']}
Market Cap: {d['market_cap']}
P/E Ratio: {d['pe_ratio']}
Forward P/E: {d['forward_pe']}
52-Week Range: ${d['52w_low']} - ${d['52w_high']}
Analyst Rating: {d['recommendation']}"""

    prompt = f"""You are a senior investment analyst writing a brief.

Original question: {state['question']}

HISTORICAL DATA (from 10-K filing):
{state['rag_results']}

LIVE MARKET DATA (right now):
{live_summary}

Write a structured investment brief with:
1. Summary
2. Key Strengths (from filing)
3. Current Valuation (from live data)
4. Recommendation

Combine both the historical filing data and live market data."""

    response = llm.invoke(prompt)
    print("   ✅ Report complete")
    return {"final_report": response.content}

# ── BUILD THE GRAPH ──────────────────────────────────────────────
print("\n🔧 Building the 4-agent graph...")

graph = StateGraph(ResearchState)

graph.add_node("planner",  planner_agent)
graph.add_node("rag",      rag_agent)
graph.add_node("data",     data_agent)
graph.add_node("analyst",  analyst_agent)

# Connect: planner → rag → data → analyst → end
graph.set_entry_point("planner")
graph.add_edge("planner", "rag")
graph.add_edge("rag",     "data")
graph.add_edge("data",    "analyst")
graph.add_edge("analyst", END)

app = graph.compile()
print("✅ Graph compiled — 4 agents connected")

# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-AGENT FINANCIAL RESEARCH SYSTEM")
    print("="*60)

    question = "Should I invest in Microsoft? Analyse their financials and current valuation."
    ticker   = "MSFT"

    print(f"\n📌 Question: {question}")
    print(f"📌 Ticker: {ticker}")
    print("-"*60)

    result = app.invoke({
        "question":     question,
        "ticker":       ticker,
        "plan":         [],
        "rag_results":  "",
        "live_data":    {},
        "final_report": ""
    })

    print("\n" + "="*60)
    print("FINAL INVESTMENT BRIEF")
    print("="*60)
    print(result["final_report"])

    print("\n" + "="*60)
    print("✅ COMPLETE — 4 agents, real 10-K data + live prices")
    print("="*60)