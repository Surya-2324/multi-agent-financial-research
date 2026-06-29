# ================================================================
# WEEK 3 — The Data Agent
#
# What this does:
# - Fetches LIVE, real-time stock data using yfinance
# - Current price, P/E ratio, market cap, 52-week high/low
# - This is data from RIGHT NOW, not from a document
#
# Combined with the RAG agent (historical 10-K facts), your system
# will have both: what happened (filing) + what's happening (live).
# ================================================================

import yfinance as yf

# ── THE DATA AGENT ───────────────────────────────────────────────
def data_agent(ticker: str) -> dict:
    print(f"\n📊 DATA AGENT fetching live data for: {ticker}")

    # Fetch the stock object from Yahoo Finance
    stock = yf.Ticker(ticker)
    info  = stock.info

    # Get recent price history (last 5 days)
    hist = stock.history(period="5d")
    latest_close = hist["Close"].iloc[-1] if not hist.empty else None

    # Build a clean dictionary of key metrics
    data = {
        "ticker":          ticker,
        "company_name":    info.get("longName", "N/A"),
        "current_price":   info.get("currentPrice", latest_close),
        "currency":        info.get("currency", "USD"),
        "market_cap":      info.get("marketCap", "N/A"),
        "pe_ratio":        info.get("trailingPE", "N/A"),
        "forward_pe":      info.get("forwardPE", "N/A"),
        "52_week_high":    info.get("fiftyTwoWeekHigh", "N/A"),
        "52_week_low":     info.get("fiftyTwoWeekLow", "N/A"),
        "dividend_yield":  info.get("dividendYield", "N/A"),
        "sector":          info.get("sector", "N/A"),
        "recommendation":  info.get("recommendationKey", "N/A"),
    }

    print(f"   ✅ Retrieved live data for {data['company_name']}")
    return data

# ── HELPER: format big numbers nicely ────────────────────────────
def format_market_cap(value):
    if isinstance(value, (int, float)):
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        else:
            return f"${value:,.0f}"
    return value

# ── TEST IT ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("DATA AGENT — Live Stock Data")
    print("="*60)

    # Fetch Microsoft live data
    data = data_agent("MSFT")

    print("\n" + "="*60)
    print(f"LIVE DATA: {data['company_name']} ({data['ticker']})")
    print("="*60)
    print(f"  Current Price   : ${data['current_price']} {data['currency']}")
    print(f"  Market Cap      : {format_market_cap(data['market_cap'])}")
    print(f"  P/E Ratio       : {data['pe_ratio']}")
    print(f"  Forward P/E     : {data['forward_pe']}")
    print(f"  52-Week High    : ${data['52_week_high']}")
    print(f"  52-Week Low     : ${data['52_week_low']}")
    print(f"  Sector          : {data['sector']}")
    print(f"  Analyst Rating  : {data['recommendation']}")
    print("="*60)
    print("\n✅ This is LIVE data fetched right now from Yahoo Finance")
    