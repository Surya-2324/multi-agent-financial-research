Project Name: Atlas — Multi-Agent Financial Research System
Tagline: Ask any question about any public company. Four AI agents research the answer using real SEC filings and live market data.

Atlas is a production-grade multi-agent AI system where four specialist agents collaborate to answer investment research questions. A user types any company name or ticker, and the system automatically downloads that company's real annual report (10-K) from the US SEC government database, builds a searchable knowledge base, fetches live stock prices, and runs four AI agents in sequence — a Planner that breaks the question into tasks, a RAG Agent that reads the actual filing, a Data Agent that fetches live market prices, and an Analyst that synthesises everything into a structured investment brief. Every agent decision is logged to a cloud governance dashboard (MLflow on DagsHub) for full audit trail. The system covers 8,000+ US public companies.

Core Features:

1) Universal company search — type any company name or ticker from 8,000+ US public companies, with a live dropdown of suggestions as you type
2) Live stock data — current price, market cap, P/E ratio, 52-week range, analyst rating, target price, trading volume, all fetched in real time from Yahoo Finance
3) Interactive price chart — Chart.js line chart with 1M / 6M / 1Y / 5Y range toggles, colour-coded green (up) or red (down) vs period start
4) Key metrics panel — market cap, P/E, forward P/E, 52-week range, target price, analyst rating (shown as a colour-coded pill — green/buy, yellow/hold, red/sell)
5) Company about section — business description, industry, employee count, website
6) Live news feed — latest 8 news headlines from Yahoo Finance, clickable through to full articles
7) AI Investment Brief — user types any free-form question, four agents run in sequence and produce a structured brief combining real SEC filing data with live market prices. Shows per-agent timing (Planner, RAG·10-K, Live Data, Analyst) with an animated pipeline
8) Cloud governance logging — every agent run logged to DagsHub/MLflow with per-agent timing, parameters, and report artifacts

Tech stack (for the technical section of the website):

LangGraph        Multi-agent orchestration
LangChain        LLM tooling and RAG chains
Ollama / LLaMA   Local LLM inference (no API cost)
ChromaDB         Vector database for RAG
HuggingFace      Sentence embeddings (all-MiniLM-L6-v2)
yfinance         Live stock market data
SEC EDGAR API    Real US government company filings
FastAPI          REST API backend
Chart.js         Interactive price charts
MLflow / DagsHub Cloud governance and audit logging
Python 3.11      Core language
