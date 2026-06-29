# ================================================================
# WEEK 2 — Step 1: Download Microsoft's real 10-K from SEC EDGAR
#
# What this does:
# - Connects to SEC's official EDGAR database
# - Finds Microsoft's latest 10-K annual report
# - Downloads it and saves as a text file
#
# SEC requires a User-Agent header with your name + email (free, no signup)
# ================================================================

import requests
import os
from bs4 import BeautifulSoup

# ── SEC requires you to identify yourself (free, no signup) ──────
# Replace with your real name and email
HEADERS = {
    "User-Agent": "Surya Prakash Gautam sprakashgautam.2002@gmail.com"
}

# Microsoft's CIK (Central Index Key) — SEC's ID for each company
MICROSOFT_CIK = "0000789019"

# ── STEP 1: Find the latest 10-K filing ──────────────────────────
def get_latest_10k_url():
    print("🔍 Searching SEC EDGAR for Microsoft's latest 10-K...")

    # SEC's API endpoint for company filings
    url = f"https://data.sec.gov/submissions/CIK{MICROSOFT_CIK}.json"
    response = requests.get(url, headers=HEADERS)
    data = response.json()

    # Get recent filings
    recent = data["filings"]["recent"]

    # Find the first 10-K
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc_name  = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]

            # Build the document URL
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/"
                       f"{int(MICROSOFT_CIK)}/{accession}/{doc_name}")

            print(f"✅ Found 10-K filed on {filing_date}")
            print(f"   URL: {doc_url}")
            return doc_url

    print("❌ No 10-K found")
    return None

# ── STEP 2: Download and clean the filing ────────────────────────
def download_and_clean(url):
    print("\n⬇  Downloading filing...")
    response = requests.get(url, headers=HEADERS)

    print("🧹 Cleaning HTML into plain text...")
    soup = BeautifulSoup(response.content, "lxml")

    # Remove script and style tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Get clean text
    text = soup.get_text(separator=" ")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    print(f"✅ Extracted {len(clean_text):,} characters")
    return clean_text

# ── STEP 3: Save to file ─────────────────────────────────────────
def save_text(text):
    os.makedirs("data", exist_ok=True)
    path = "data/microsoft_10k.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"💾 Saved to {path}")
    return path

# ── RUN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("SEC 10-K DOWNLOADER — Microsoft")
    print("="*60)

    url = get_latest_10k_url()

    if url:
        text = download_and_clean(url)
        save_text(text)

        # Show a preview
        print("\n" + "="*60)
        print("PREVIEW (first 500 characters):")
        print("="*60)
        print(text[:500])
        print("\n✅ Download complete! Ready for chunking and embedding.")