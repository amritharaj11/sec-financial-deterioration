import requests 
import pandas as pd
import time
import re
from pathlib import Path
from bs4 import BeautifulSoup

# Configuration

your_email = 'genaiupdates24@gmail.com'
input_path = Path('data/raw/filings_metadata.csv')
output_dir = Path('data/raw')

headers = {
    'User-Agent': f'Student research project ({your_email})',
    'Accept-Encoding': 'gzip, deflate'
    # 'Host': 'data.sec.gov'
}

def get_document_url(cik, accession_number):
    """
    Every filing on EDGAR has an index page that lists all documents
    in that filing. We fetch that index and find the main 10-K document.
 
    The index URL format is:
    https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{accession_no_dashes}-index.htm
    """
    cik_int       = int(cik)
    clean_acc     = accession_number.replace("-", "")
    index_url     = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{clean_acc}/{accession_number}-index.htm"
    )
 
    try:
        r = requests.get(index_url, headers=headers, timeout=15)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"      ✗ Could not fetch filing index: {e}")
        return None
 
    soup  = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"class": "tableFile"})
    if not table:
        return None
 
    # Loop through filing documents and find the main 10-K htm file
    # We skip exhibits, amendments, and other supplementary documents
    for row in table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
 
        doc_type    = cols[3].text.strip()
        description = cols[1].text.strip().lower()
        link        = cols[2].find("a")
 
        # We want the primary 10-K document, not exhibits or amendments
        if doc_type == "10-K" and link:
            href = link.get("href", "")
            if href.endswith(".htm") or href.endswith(".html"):
                if "ix?doc=" in href:
                    href = href.split("ix?doc=")[1]
                return f"https://www.sec.gov{href}"
 
    return None
 
 
# ─────────────────────────────────────────────
# STEP 2: Download and clean the 10-K HTML
# ─────────────────────────────────────────────
def download_and_clean(url):
    """
    Downloads the 10-K HTML document and strips it down to plain text.
    Removes all HTML tags, extra whitespace, and page artifacts.
    """
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"      ✗ Could not download document: {e}")
        return None
 
    soup = BeautifulSoup(r.text, "lxml")
 
    # Remove script and style tags — they add noise
    for tag in soup(["script", "style"]):
        tag.decompose()
 
    # Get plain text and normalize whitespace
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
 
    return text
 
 
# ─────────────────────────────────────────────
# STEP 3: Extract a specific section from text
# ─────────────────────────────────────────────
def extract_section(text, start_patterns, end_patterns, max_chars=50000):
    """
    Extracts a section of text between a start pattern and end pattern.
 
    10-K documents follow a standard structure but the exact formatting
    varies by company and year. We try multiple patterns to find each
    section boundary and take the first one that matches.
 
    max_chars limits extraction to avoid pulling in the entire document
    if the end pattern isn't found.
    """
    text_upper = text.upper()
 
    # Find the start of the section
    start_idx = None
    for pattern in start_patterns:
        match = re.search(pattern, text_upper)
        if match:
            start_idx = match.start()
            break
 
    if start_idx is None:
        return None
 
    # Find the end of the section
    end_idx = None
    for pattern in end_patterns:
        match = re.search(pattern, text_upper[start_idx + 100:])
        if match:
            # Add offset back since we searched from start_idx + 100
            end_idx = start_idx + 100 + match.start()
            break
 
    if end_idx is None:
        # If no end pattern found, take max_chars from start
        end_idx = start_idx + max_chars
 
    extracted = text[start_idx:end_idx].strip()
 
    # Return None if extracted text is suspiciously short
    # (likely a false match on a table of contents entry)
    if len(extracted) < 500:
        return None
 
    # Truncate to max_chars to keep file sizes manageable
    return extracted[:max_chars]
 
 
# Section boundary patterns
# These are the most common ways companies label these sections
RISK_FACTORS_START = [
    r"ITEM\s+1A[\.\s]+RISK\s+FACTORS",
    r"ITEM\s+1A\s*[\.\-–]\s*RISK\s+FACTORS",
]
 
RISK_FACTORS_END = [
    r"ITEM\s+1B[\.\s]+UNRESOLVED\s+STAFF\s+COMMENTS",
    r"ITEM\s+2[\.\s]+PROPERTIES",
]
 
MDA_START = [
    r"ITEM\s+7[\.\s]+MANAGEMENT.S\s+DISCUSSION",
    r"ITEM\s+7\s*[\.\-–]\s*MANAGEMENT.S\s+DISCUSSION",
]
 
MDA_END = [
    r"ITEM\s+7A[\.\s]+QUANTITATIVE",
    r"ITEM\s+8[\.\s]+FINANCIAL\s+STATEMENTS",
]
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
print("=" * 65)
print("  SEC EDGAR 10-K Text Extractor")
print("=" * 65)
 
if not input_path.exists():
    raise SystemExit(f"✗ Could not find {input_path}. Run sec_pull_local.py first.")
 
meta_df = pd.read_csv(input_path)
print(f"\n✓ Loaded {len(meta_df)} filings from {input_path}")
print(f"  Extracting Risk Factors (Item 1A) and MD&A (Item 7)\n")
 
all_texts = []
total     = len(meta_df)
 
for idx, row in meta_df.iterrows():
    ticker  = row["ticker"]
    company = row["company"]
    year    = row["year"]
    cik     = str(row["cik"]).zfill(10)
    acc     = row["accession_number"]
 
    print(f"  [{idx+1}/{total}] {ticker} {year} — {company}")
 
    # Step 1: Find the document URL from the filing index
    doc_url = get_document_url(cik, acc)
    if not doc_url:
        print(f"      ✗ Could not find main document URL")
        all_texts.append({
            "ticker":       ticker,
            "company":      company,
            "year":         year,
            "risk_factors": None,
            "mda":          None,
            "doc_url":      None,
        })
        time.sleep(0.3)
        continue
 
    print(f"      → {doc_url}")
 
    # Step 2: Download and clean the document
    text = download_and_clean(doc_url)
    if not text:
        print(f"      ✗ Could not download document")
        all_texts.append({
            "ticker":       ticker,
            "company":      company,
            "year":         year,
            "risk_factors": None,
            "mda":          None,
            "doc_url":      doc_url,
        })
        time.sleep(0.3)
        continue
 
    # Step 3: Extract Risk Factors section
    risk_factors = extract_section(
        text,
        start_patterns = RISK_FACTORS_START,
        end_patterns   = RISK_FACTORS_END,
    )
 
    # Step 4: Extract MD&A section
    mda = extract_section(
        text,
        start_patterns = MDA_START,
        end_patterns   = MDA_END,
    )
 
    # Report what was found
    rf_status  = f"{len(risk_factors):,} chars" if risk_factors else "✗ NOT FOUND"
    mda_status = f"{len(mda):,} chars"          if mda          else "✗ NOT FOUND"
    print(f"      Risk Factors : {rf_status}")
    print(f"      MD&A         : {mda_status}")
 
    all_texts.append({
        "ticker":       ticker,
        "company":      company,
        "year":         year,
        "risk_factors": risk_factors,
        "mda":          mda,
        "doc_url":      doc_url,
    })
 
    # Be polite to SEC servers — larger delay since we are downloading full documents
    time.sleep(1.0)
 
# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
df = pd.DataFrame(all_texts)
 
print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")
print(f"  Total filings processed : {len(df)}")
rf_found  = df["risk_factors"].notna().sum()
mda_found = df["mda"].notna().sum()
print(f"  Risk Factors extracted  : {rf_found}/{len(df)}")
print(f"  MD&A extracted          : {mda_found}/{len(df)}")
print(f"  Both sections found     : {(df['risk_factors'].notna() & df['mda'].notna()).sum()}/{len(df)}")
print(f"{'=' * 65}\n")
 
output_path = output_dir / "filing_texts.csv"
df.to_csv(output_path, index=False)
print(f"✓ Saved to {output_path}")
print(f"\nNext step: Run build_features.py to combine financial ratios + NLP features.")