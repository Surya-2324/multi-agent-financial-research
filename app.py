# ================================================================
# ATLAS — Multi-Agent Financial Research Dashboard
# Cloud version: uses Groq (LLaMA) instead of Ollama
# ================================================================

import os
import time
import requests
import traceback
from typing import TypedDict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import yfinance as yf
from bs4 import BeautifulSoup

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace with Vercel URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from dotenv import load_dotenv
import mlflow
import warnings
warnings.filterwarnings("ignore")

# ── SETUP ────────────────────────────────────────────────────────
load_dotenv()
HEADERS = {"User-Agent": "Surya Prakash Gautam sprkashgautam.2002@gmail.com"}

# MLflow governance
MLFLOW_ENABLED = bool(os.getenv("MLFLOW_TRACKING_URI"))
if MLFLOW_ENABLED:
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    try:
        mlflow.set_experiment("financial-research-agents")
    except Exception:
        MLFLOW_ENABLED = False

# ── LLM — Groq (cloud, fast, free) ───────────────────────────────
print("🔧 Initialising Groq LLM...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=GROQ_API_KEY
)
print("✅ Groq LLM ready")

# ── Embeddings — HuggingFace (local, free) ────────────────────────
print("🔧 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("✅ Embeddings ready")

# ── FASTAPI + CORS ────────────────────────────────────────────────
app = FastAPI(title="Atlas — Financial Research Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_vectorstores = {}
_company_list = None

def load_company_list():
    global _company_list
    if _company_list is None:
        url = "https://www.sec.gov/files/company_tickers.json"
        data = requests.get(url, headers=HEADERS).json()
        _company_list = [{"ticker": e["ticker"], "name": e["title"]} for e in data.values()]
    return _company_list

# ════════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════════
def get_cik(ticker: str):
    companies = requests.get("https://www.sec.gov/files/company_tickers.json",
                             headers=HEADERS).json()
    for e in companies.values():
        if e["ticker"].upper() == ticker.upper():
            return str(e["cik_str"]).zfill(10)
    return None

def download_10k(cik: str, ticker: str):
    path = f"/tmp/{ticker}_10k.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            acc = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            durl = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
            r = requests.get(durl, headers=HEADERS)
            soup = BeautifulSoup(r.content, "lxml")
            for t in soup(["script", "style"]):
                t.decompose()
            text = soup.get_text(separator=" ")
            clean = "\n".join(l.strip() for l in text.splitlines() if l.strip())
            with open(path, "w", encoding="utf-8") as f:
                f.write(clean)
            return clean
    return None

def get_vectorstore(ticker: str):
    if ticker in _vectorstores:
        return _vectorstores[ticker]

    persist = f"/tmp/chroma_{ticker.lower()}"
    collection = f"{ticker.lower()}_10k"

    if os.path.exists(persist):
        vs = Chroma(persist_directory=persist, embedding_function=embeddings,
                    collection_name=collection)
        _vectorstores[ticker] = vs
        return vs

    cik = get_cik(ticker)
    if not cik:
        return None
    text = download_10k(cik, ticker)
    if not text:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""])
    docs = splitter.split_documents([Document(page_content=text)])
    vs = Chroma.from_documents(docs, embeddings,
                               persist_directory=persist,
                               collection_name=collection)
    _vectorstores[ticker] = vs
    return vs

# ════════════════════════════════════════════════════════════════
# 4-AGENT SYSTEM
# ════════════════════════════════════════════════════════════════
class State(TypedDict):
    question:     str
    ticker:       str
    plan:         List[str]
    rag_results:  str
    live_data:    dict
    final_report: str
    timings:      dict

class VSHolder:
    current = None
_holder = VSHolder()

def planner(state: State) -> State:
    t0 = time.time()
    p = f"Question: {state['question']}\nBreak into 3 research tasks. Numbered list only."
    r = llm.invoke(p)
    tasks = [l.strip() for l in r.content.split("\n") if l.strip() and l[0].isdigit()]
    state["timings"]["planner"] = round(time.time() - t0, 2)
    return {"plan": tasks, "timings": state["timings"]}

def rag(state: State) -> State:
    t0 = time.time()
    docs = _holder.current.similarity_search(state["question"], k=4)
    ctx = "\n\n".join(d.page_content for d in docs)
    p = f"""Use ONLY this context. Cite specific numbers.
CONTEXT:\n{ctx}\nQUESTION: {state['question']}\nKey findings:"""
    r = llm.invoke(p)
    state["timings"]["rag"] = round(time.time() - t0, 2)
    return {"rag_results": r.content, "timings": state["timings"]}

def data(state: State) -> State:
    t0 = time.time()
    try:
        info = yf.Ticker(state["ticker"]).info
    except Exception:
        info = {}
    d = {
        "company":        info.get("longName", state["ticker"]),
        "current_price":  info.get("currentPrice", "N/A"),
        "market_cap":     info.get("marketCap", "N/A"),
        "pe_ratio":       info.get("trailingPE", "N/A"),
        "forward_pe":     info.get("forwardPE", "N/A"),
        "52w_high":       info.get("fiftyTwoWeekHigh", "N/A"),
        "52w_low":        info.get("fiftyTwoWeekLow", "N/A"),
        "recommendation": info.get("recommendationKey", "N/A"),
    }
    state["timings"]["data"] = round(time.time() - t0, 2)
    return {"live_data": d, "timings": state["timings"]}

def analyst(state: State) -> State:
    t0 = time.time()
    d = state["live_data"]
    live = (f"Company: {d['company']}\nPrice: ${d['current_price']}\n"
            f"Market Cap: {d['market_cap']}\nP/E: {d['pe_ratio']}\n"
            f"52w Range: ${d['52w_low']} - ${d['52w_high']}\n"
            f"Rating: {d['recommendation']}")
    p = f"""Senior investment analyst. Question: {state['question']}
HISTORICAL (10-K): {state['rag_results']}
LIVE DATA: {live}
Write investment brief: 1. Summary 2. Key Strengths 3. Valuation 4. Recommendation"""
    r = llm.invoke(p)
    state["timings"]["analyst"] = round(time.time() - t0, 2)
    return {"final_report": r.content, "timings": state["timings"]}

_g = StateGraph(State)
_g.add_node("planner", planner)
_g.add_node("rag", rag)
_g.add_node("data", data)
_g.add_node("analyst", analyst)
_g.set_entry_point("planner")
_g.add_edge("planner", "rag")
_g.add_edge("rag", "data")
_g.add_edge("data", "analyst")
_g.add_edge("analyst", END)
_agent_app = _g.compile()

# ════════════════════════════════════════════════════════════════
# API ROUTES
# ════════════════════════════════════════════════════════════════
@app.get("/api/search")
def api_search(q: str = ""):
    if len(q) < 1:
        return {"results": []}
    companies = load_company_list()
    ql = q.lower()
    ticker_matches = [c for c in companies if c["ticker"].lower().startswith(ql)]
    name_matches   = [c for c in companies if ql in c["name"].lower()
                      and c not in ticker_matches]
    return {"results": (ticker_matches + name_matches)[:12]}

@app.get("/api/quote/{ticker}")
def api_quote(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker":         ticker.upper(),
            "company":        info.get("longName", ticker.upper()),
            "price":          info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "previous_close": info.get("previousClose", "N/A"),
            "currency":       info.get("currency", "USD"),
            "market_cap":     info.get("marketCap", "N/A"),
            "pe_ratio":       info.get("trailingPE", "N/A"),
            "forward_pe":     info.get("forwardPE", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52w_high":       info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low":        info.get("fiftyTwoWeekLow", "N/A"),
            "volume":         info.get("volume", "N/A"),
            "avg_volume":     info.get("averageVolume", "N/A"),
            "sector":         info.get("sector", "N/A"),
            "industry":       info.get("industry", "N/A"),
            "employees":      info.get("fullTimeEmployees", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
            "target_price":   info.get("targetMeanPrice", "N/A"),
            "summary":        info.get("longBusinessSummary", "No description available."),
            "website":        info.get("website", ""),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/history/{ticker}")
def api_history(ticker: str, period: str = "6mo"):
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return {
            "dates":   [d.strftime("%Y-%m-%d") for d in hist.index],
            "prices":  [round(float(p), 2) for p in hist["Close"]],
            "volumes": [int(v) for v in hist["Volume"]],
            "highs":   [round(float(h), 2) for h in hist["High"]],
            "lows":    [round(float(l), 2) for l in hist["Low"]],
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/news/{ticker}")
def api_news(ticker: str):
    try:
        news = yf.Ticker(ticker).news
        items = []
        for n in news[:8]:
            content = n.get("content", n)
            prov = content.get("provider", {})
            url  = content.get("canonicalUrl", {})
            items.append({
                "title":     content.get("title", "Untitled"),
                "publisher": prov.get("displayName", "") if isinstance(prov, dict) else "",
                "link":      url.get("url", "#") if isinstance(url, dict) else "#",
            })
        return {"news": items}
    except Exception as e:
        return {"news": [], "error": str(e)}

class AnalyseRequest(BaseModel):
    ticker: str
    question: str

@app.post("/api/analyse")
def api_analyse(req: AnalyseRequest):
    ticker = req.ticker.upper()
    try:
        vs = get_vectorstore(ticker)
        if vs is None:
            return {"error": f"Could not load 10-K for {ticker}. It may be a foreign company filing a 20-F instead of 10-K."}
        _holder.current = vs

        total_start = time.time()
        result = _agent_app.invoke({
            "question": req.question, "ticker": ticker,
            "plan": [], "rag_results": "", "live_data": {},
            "final_report": "", "timings": {},
        })
        total_time = round(time.time() - total_start, 2)

        if MLFLOW_ENABLED:
            try:
                with mlflow.start_run(run_name=f"{ticker}-dashboard"):
                    mlflow.log_param("ticker", ticker)
                    mlflow.log_param("question", req.question)
                    for k, v in result["timings"].items():
                        mlflow.log_metric(f"{k}_time", v)
                    mlflow.log_metric("total_time", total_time)
            except Exception:
                pass

        return {
            "report":     result["final_report"],
            "timings":    result["timings"],
            "total_time": total_time,
            "live_data":  result["live_data"],
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# ════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ════════════════════════════════════════════════════════════════
with open("dashboard.html", encoding="utf-8") as f:
    DASHBOARD_HTML = f.read()

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML