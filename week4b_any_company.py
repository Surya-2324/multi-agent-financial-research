# ================================================================
# WEEK 4B — Full System for ANY Company (Level 1 + Level 2)
#
# - Type any ticker (AAPL, MSFT, GOOGL, NVDA, etc.)
# - System auto-downloads that company's 10-K from SEC
# - Builds a ChromaDB knowledge base for it
# - Type any question
# - 4 agents analyse and produce an investment brief
# ================================================================

import os
import requests
from typing import TypedDict, List
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# SEC requires identification (free, no signup)
HEADERS = {"User-Agent": "Surya Prakash Gautam sprkashgautam.2002@gmail.com"}

# ── SHARED RESOURCES ─────────────────────────────────────────────
print("🔧 Initialising system...")
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ LLM + embeddings ready")

# ── HELPER: Look up CIK from ticker ──────────────────────────────
def get_cik_from_ticker(ticker: str) -> str:
    """SEC provides a free ticker→CIK mapping file."""
    print(f"🔎 Looking up CIK for {ticker}...")
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()

    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)  # pad to 10 digits
            print(f"   ✅ Found CIK: {cik} ({entry['title']})")
            return cik

    print(f"   ❌ Ticker {ticker} not found")
    return None

# ── HELPER: Download 10-K for a CIK ──────────────────────────────
def download_10k(cik: str, ticker: str) -> str:
    """Download the latest 10-K and save as text."""
    path = f"data/{ticker}_10k.txt"

    # If already downloaded, reuse it
    if os.path.exists(path):
        print(f"   ✅ Using cached 10-K for {ticker}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    print(f"⬇  Downloading 10-K for {ticker}...")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()
    recent = data["filings"]["recent"]

    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc_name  = recent["primaryDocument"][i]
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                       f"{int(cik)}/{accession}/{doc_name}")

            response = requests.get(doc_url, headers=HEADERS)
            soup = BeautifulSoup(response.content, "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            clean = "\n".join(lines)

            os.makedirs("data", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean)

            print(f"   ✅ Downloaded {len(clean):,} characters")
            return clean

    return None

# ── HELPER: Build or load ChromaDB for a company ─────────────────
def get_vectorstore(text: str, ticker: str):
    """Build ChromaDB for this company (or load if exists)."""
    collection = f"{ticker.lower()}_10k"
    persist_dir = f"chroma_db_{ticker.lower()}"

    # If already built, just load it
    if os.path.exists(persist_dir):
        print(f"   ✅ Loading existing knowledge base for {ticker}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection
        )

    print(f"📚 Building knowledge base for {ticker}...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.split_documents([Document(page_content=text)])
    print(f"   Created {len(docs)} chunks")

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection
    )
    print(f"   ✅ Knowledge base ready")
    return vs

# ── STATE ────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    question:     str
    ticker:       str
    plan:         List[str]
    rag_results:  str
    live_data:    dict
    final_report: str

# These get set dynamically before running
current_vectorstore = None

# ── AGENT 1: PLANNER ─────────────────────────────────────────────
def planner_agent(state: ResearchState) -> ResearchState:
    print("\n🧠 PLANNER AGENT")
    prompt = f"""You are a research planner.
Question: {state['question']}
Break this into 3 specific research tasks. Format as a numbered list."""
    response = llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    tasks = [l.strip() for l in lines if l.strip() and l[0].isdigit()]
    print(f"   ✅ Created {len(tasks)} tasks")
    return {"plan": tasks}

# ── AGENT 2: RAG AGENT ───────────────────────────────────────────
def rag_agent(state: ResearchState) -> ResearchState:
    print(f"\n🔍 RAG AGENT — reading {state['ticker']} 10-K")
    docs = current_vectorstore.similarity_search(state['question'], k=4)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are a financial analyst. Use ONLY the context below.
Cite specific numbers when available.

CONTEXT:
{context}

QUESTION: {state['question']}

Key findings from the filing:"""
    response = llm.invoke(prompt)
    print(f"   ✅ Retrieved {len(docs)} passages")
    return {"rag_results": response.content}

# ── AGENT 3: DATA AGENT ──────────────────────────────────────────
def data_agent(state: ResearchState) -> ResearchState:
    print(f"\n📊 DATA AGENT — live data for {state['ticker']}")
    info = yf.Ticker(state['ticker']).info
    data = {
        "company":        info.get("longName", "N/A"),
        "current_price":  info.get("currentPrice", "N/A"),
        "market_cap":     info.get("marketCap", "N/A"),
        "pe_ratio":       info.get("trailingPE", "N/A"),
        "forward_pe":     info.get("forwardPE", "N/A"),
        "52w_high":       info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low":        info.get("fiftyTwoWeekLow", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
    }
    print(f"   ✅ Live price: ${data['current_price']}")
    return {"live_data": data}

# ── AGENT 4: ANALYST ─────────────────────────────────────────────
def analyst_agent(state: ResearchState) -> ResearchState:
    print("\n📝 ANALYST AGENT — writing report")
    d = state['live_data']
    live = f"""Company: {d['company']}
Current Price: ${d['current_price']}
Market Cap: {d['market_cap']}
P/E Ratio: {d['pe_ratio']}
Forward P/E: {d['forward_pe']}
52-Week Range: ${d['52w_low']} - ${d['52w_high']}
Analyst Rating: {d['recommendation']}"""

    prompt = f"""You are a senior investment analyst.

Question: {state['question']}

HISTORICAL DATA (from 10-K):
{state['rag_results']}

LIVE MARKET DATA:
{live}

Write a structured investment brief:
1. Summary
2. Key Strengths (from filing)
3. Current Valuation (from live data)
4. Recommendation"""
    response = llm.invoke(prompt)
    print("   ✅ Report complete")
    return {"final_report": response.content}

# ── BUILD GRAPH ──────────────────────────────────────────────────
graph = StateGraph(ResearchState)
graph.add_node("planner", planner_agent)
graph.add_node("rag",     rag_agent)
graph.add_node("data",    data_agent)
graph.add_node("analyst", analyst_agent)
graph.set_entry_point("planner")
graph.add_edge("planner", "rag")
graph.add_edge("rag",     "data")
graph.add_edge("data",    "analyst")
graph.add_edge("analyst", END)
app = graph.compile()

# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-AGENT FINANCIAL RESEARCH — ANY COMPANY")
    print("="*60)

    # LEVEL 2: User picks any company
    ticker = input("\n📈 Enter a stock ticker (e.g. AAPL, MSFT, GOOGL, NVDA): ").strip().upper()

    # Look up and download that company's 10-K
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print("Could not find that ticker. Exiting.")
        exit()

    text = download_10k(cik, ticker)
    if not text:
        print("Could not download 10-K. Exiting.")
        exit()

    # Build/load knowledge base for this company
    current_vectorstore = get_vectorstore(text, ticker)

   # LEVEL 1: Keep asking questions in a loop
    print(f"\n💬 Ask anything about {ticker}")
    print("   (type 'quit' to exit, or 'switch' to analyse a different company)")

    while True:
        print("\n" + "-"*60)
        question = input(f"\n❓ Your question about {ticker} (or 'quit'/'switch'): ").strip()

        # Exit the program
        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        # Switch to a different company
        if question.lower() == "switch":
            new_ticker = input("📈 Enter new ticker: ").strip().upper()
            cik = get_cik_from_ticker(new_ticker)
            if not cik:
                print("Ticker not found, keeping current company.")
                continue
            text = download_10k(cik, new_ticker)
            current_vectorstore = get_vectorstore(text, new_ticker)
            ticker = new_ticker
            print(f"✅ Switched to {ticker}")
            continue

        # Empty question — skip
        if not question:
            print("Please type a question.")
            continue

        # Run the 4-agent analysis
        print("\n" + "-"*60)
        result = app.invoke({
            "question":     question,
            "ticker":       ticker,
            "plan":         [],
            "rag_results":  "",
            "live_data":    {},
            "final_report": ""
        })

        print("\n" + "="*60)
        print("INVESTMENT BRIEF")
        print("="*60)
        print(result["final_report"])