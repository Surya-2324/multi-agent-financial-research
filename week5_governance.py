# ================================================================
# WEEK 5 — Governance Layer with MLflow
#
# Every agent decision is logged to MLflow for full audit trail.
# This is what makes the system enterprise-grade for Microsoft/JPM.
#
# For each question, we log:
#   - which agent ran, how long it took
#   - what the RAG agent retrieved
#   - the live data used
#   - the final report
#   - full timing breakdown
# ================================================================

import os
import time
import requests
from typing import TypedDict, List
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import yfinance as yf
import mlflow
import warnings
warnings.filterwarnings("ignore")

HEADERS = {"User-Agent": "Surya Prakash Gautam sprkashgautam.2002@gmail.com"}

# ── MLFLOW SETUP ─────────────────────────────────────────────────
load_dotenv()

# Set DagsHub credentials as environment variables for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("financial-research-agents")
print("✅ MLflow governance tracking enabled")

# ── SHARED RESOURCES ─────────────────────────────────────────────
print("🔧 Initialising system...")
llm = ChatOllama(model="llama3.2", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ LLM + embeddings ready")

# ── HELPERS (same as week 4) ─────────────────────────────────────
def get_cik_from_ticker(ticker: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    return None

def download_10k(cik: str, ticker: str) -> str:
    path = f"data/{ticker}_10k.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
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
            return clean
    return None

def get_vectorstore(text: str, ticker: str):
    collection = f"{ticker.lower()}_10k"
    persist_dir = f"chroma_db_{ticker.lower()}"
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir,
                      embedding_function=embeddings,
                      collection_name=collection)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.split_documents([Document(page_content=text)])
    return Chroma.from_documents(documents=docs, embedding=embeddings,
                                 persist_directory=persist_dir,
                                 collection_name=collection)

# ── STATE — now includes an audit log ────────────────────────────
class ResearchState(TypedDict):
    question:     str
    ticker:       str
    plan:         List[str]
    rag_results:  str
    live_data:    dict
    final_report: str
    audit_log:    list   # NEW: every agent logs here
    timings:      dict   # NEW: how long each agent took

current_vectorstore = None

# ── AGENT 1: PLANNER (with governance) ───────────────────────────
def planner_agent(state: ResearchState) -> ResearchState:
    print("\n🧠 PLANNER AGENT")
    start = time.time()

    prompt = f"""You are a research planner.
Question: {state['question']}
Break this into 3 specific research tasks. Format as a numbered list."""
    response = llm.invoke(prompt)
    lines = response.content.strip().split('\n')
    tasks = [l.strip() for l in lines if l.strip() and l[0].isdigit()]

    elapsed = time.time() - start
    print(f"   ✅ Created {len(tasks)} tasks ({elapsed:.1f}s)")

    # GOVERNANCE: log this agent's action
    log = state.get("audit_log", [])
    log.append({
        "agent":    "planner",
        "input":    state['question'],
        "output":   f"{len(tasks)} tasks created",
        "duration": round(elapsed, 2),
        "model":    "llama3.2",
    })
    timings = state.get("timings", {})
    timings["planner"] = round(elapsed, 2)

    return {"plan": tasks, "audit_log": log, "timings": timings}

# ── AGENT 2: RAG (with governance) ───────────────────────────────
def rag_agent(state: ResearchState) -> ResearchState:
    print(f"\n🔍 RAG AGENT — reading {state['ticker']} 10-K")
    start = time.time()

    docs = current_vectorstore.similarity_search(state['question'], k=4)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are a financial analyst. Use ONLY the context below.
Cite specific numbers when available.

CONTEXT:
{context}

QUESTION: {state['question']}

Key findings from the filing:"""
    response = llm.invoke(prompt)

    elapsed = time.time() - start
    print(f"   ✅ Retrieved {len(docs)} passages ({elapsed:.1f}s)")

    # GOVERNANCE
    log = state.get("audit_log", [])
    log.append({
        "agent":           "rag",
        "input":           state['question'],
        "output":          f"Retrieved {len(docs)} passages from {state['ticker']} 10-K",
        "passages_used":   len(docs),
        "duration":        round(elapsed, 2),
        "model":           "llama3.2 + all-MiniLM-L6-v2",
    })
    timings = state.get("timings", {})
    timings["rag"] = round(elapsed, 2)

    return {"rag_results": response.content, "audit_log": log, "timings": timings}

# ── AGENT 3: DATA (with governance) ──────────────────────────────
def data_agent(state: ResearchState) -> ResearchState:
    print(f"\n📊 DATA AGENT — live data for {state['ticker']}")
    start = time.time()

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

    elapsed = time.time() - start
    print(f"   ✅ Live price: ${data['current_price']} ({elapsed:.1f}s)")

    # GOVERNANCE
    log = state.get("audit_log", [])
    log.append({
        "agent":    "data",
        "input":    state['ticker'],
        "output":   f"Price ${data['current_price']}, P/E {data['pe_ratio']}",
        "source":   "yfinance (Yahoo Finance API)",
        "duration": round(elapsed, 2),
    })
    timings = state.get("timings", {})
    timings["data"] = round(elapsed, 2)

    return {"live_data": data, "audit_log": log, "timings": timings}

# ── AGENT 4: ANALYST (with governance) ───────────────────────────
def analyst_agent(state: ResearchState) -> ResearchState:
    print("\n📝 ANALYST AGENT — writing report")
    start = time.time()

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

    elapsed = time.time() - start
    print(f"   ✅ Report complete ({elapsed:.1f}s)")

    # GOVERNANCE
    log = state.get("audit_log", [])
    log.append({
        "agent":    "analyst",
        "input":    "RAG results + live data",
        "output":   f"{len(response.content)} char investment brief",
        "duration": round(elapsed, 2),
        "model":    "llama3.2",
    })
    timings = state.get("timings", {})
    timings["analyst"] = round(elapsed, 2)

    return {"final_report": response.content, "audit_log": log, "timings": timings}

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

# ── RUN WITH GOVERNANCE ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("MULTI-AGENT SYSTEM WITH GOVERNANCE LOGGING")
    print("="*60)

    ticker = input("\n📈 Enter a stock ticker: ").strip().upper()
    cik = get_cik_from_ticker(ticker)
    if not cik:
        print("Ticker not found. Exiting.")
        exit()
    text = download_10k(cik, ticker)
    current_vectorstore = get_vectorstore(text, ticker)

    while True:
        question = input(f"\n❓ Question about {ticker} (or 'quit'): ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue

        # ── START MLFLOW RUN (governance) ─────────────────────────
        total_start = time.time()
        with mlflow.start_run(run_name=f"{ticker}-research"):
            # Log the inputs
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("question", question)
            mlflow.log_param("model", "llama3.2")

            # Run all 4 agents
            result = app.invoke({
                "question":     question,
                "ticker":       ticker,
                "plan":         [],
                "rag_results":  "",
                "live_data":    {},
                "final_report": "",
                "audit_log":    [],
                "timings":      {},
            })

            total_time = time.time() - total_start

            # ── LOG GOVERNANCE METRICS ────────────────────────────
            timings = result["timings"]
            mlflow.log_metric("total_time_sec",    round(total_time, 2))
            mlflow.log_metric("planner_time_sec",  timings.get("planner", 0))
            mlflow.log_metric("rag_time_sec",      timings.get("rag", 0))
            mlflow.log_metric("data_time_sec",     timings.get("data", 0))
            mlflow.log_metric("analyst_time_sec",  timings.get("analyst", 0))
            mlflow.log_metric("agents_executed",   len(result["audit_log"]))

            # Log the live data fetched
            for key, val in result["live_data"].items():
                mlflow.log_param(f"data_{key}", val)

            # Save the full audit log as a file artifact
            import json
            os.makedirs("audit_logs", exist_ok=True)
            audit_path = f"audit_logs/{ticker}_{int(time.time())}.json"
            with open(audit_path, "w") as f:
                json.dump({
                    "question":   question,
                    "ticker":     ticker,
                    "audit_log":  result["audit_log"],
                    "timings":    result["timings"],
                    "report":     result["final_report"],
                }, f, indent=2)
            mlflow.log_artifact(audit_path)

            # Save the report as artifact
            report_path = f"audit_logs/{ticker}_report_{int(time.time())}.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(result["final_report"])
            mlflow.log_artifact(report_path)

        # ── PRINT RESULTS ─────────────────────────────────────────
        print("\n" + "="*60)
        print("INVESTMENT BRIEF")
        print("="*60)
        print(result["final_report"])

        print("\n" + "="*60)
        print("GOVERNANCE AUDIT TRAIL")
        print("="*60)
        for entry in result["audit_log"]:
            print(f"  [{entry['agent'].upper()}] {entry['output']} ({entry['duration']}s)")
        print(f"\n  Total time: {total_time:.1f}s")
        print(f"  ✅ Logged to MLflow → http://localhost:5000")