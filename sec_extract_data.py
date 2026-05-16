import requests
import pandas as pd
import time
import json
from pathlib import Path

# Configuration

your_email = 'genaiupdates24@gmail.com'
input_path = Path('data/raw/filings_metadata.csv')
output_dir = Path('data/raw')

headers = {
    'User-Agent': f'Student research project ({your_email})',
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

# ─────────────────────────────────────────────
# XBRL CONCEPT (Extensible Business Reporting Language) MAPPINGS. Before XBRL, we had to manually get the figures to compare companies. But with XBRL, 
# companies tag every single number that they report with a standardized label. THese numbers are made machine readable and hence XBRL is used to standardize the financial data.
# Different companies use different tag names
# for the same financial figure. We try each
# one in order and use the first one that works.
# ─────────────────────────────────────────────
#The following is a list of XBRL concepts and their possible tag names and have been mapped to a common name.This is done because companies use different XRBL labels to represent the same financial figures.
concepts = {
    'revenue': [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],

    'gross_profit': [
        "GrossProfit",
        'GrossProfitLoss',
    ],

    'operating_income': [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],

    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],

    "total_assets": [
        "Assets",
    ],

    "current_assets": [
        "AssetsCurrent",
    ],

    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    
    "total_debt": [
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
        "ShortTermBorrowing",
    ],
    
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
    ],
}


# Step 1: Fetch all XBRL Facts for a company
def fetch_company_facts(cik):
    '''Downloads all XBRL financial facts for a company from Edgar.
    Returns a dict of all reported financial concepts and their values for the company.
    This is one API call per company -- not per filing'''

    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'

    try:
        r = requests.get(url, headers=headers, timeout=20) #Timeout is 20 because these data can be of several MB and might crash for large requests and shorter timeframes
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f'Failed to fetch XBRL facts: {e}')
        return None
    

# Step 2: Extract a single concept value for a year
def extract_concept_value(facts, concept_name, year):
    """
    Pulls the annual value for a specific financial concept and year.
 
    XBRL data is structured as:
    facts → us-gaap → ConceptName → units → USD → [list of reported values]
 
    Each reported value has:
        - val:   the number
        - end:   the period end date e.g. "2023-12-31"
        - form:  the form it came from e.g. "10-K"
        - fp:    fiscal period e.g. "FY" for full year
 
    We filter for annual (FY) 10-K values ending in the target year.
    """

    us_gaap = facts.get('facts', {}).get('us-gaap', {}) # The XBRL JSON is nested. we are fetching the us-gaap section. If any key is missing, we default to an empty dict.

    for tag in concepts[concept_name]: # Try each possible tag for the concept. It is going to use the values for the various possible tags.
        concept_data = us_gaap.get(tag) # We use get instead of us_gaap[tag] to avoid KeyError if the key is missing. If a key is missing, we get None.
        if not concept_data:
            continue

        usd_data = concept_data.get('units', {}).get('USD', [])
        if not usd_data:
            continue

        annual_values = [
            entry for entry in usd_data
            if entry.get('form') == '10-K'
            and entry.get('fp') == 'FY'
            and entry.get('end', '').startswith(year)
        ]

        # There are times when there are multiple values for the same concept in the same year.
        # We want the most recent one, so we sort by the 'filed' date in descending order and get the most recent value.
        if annual_values:
            annual_values.sort(key=lambda x: x.get('filed', ''), reverse=True)
            return annual_values[0]['val']
        
        return None
    

# Step 3: Calculates ratios from raw figures
def calculate_ratios(raw, ticker, company, sector, year):
    """
    Takes raw financial figures and calculates 8 ratios.
    Returns None for any ratio where the required figures are missing.
    Division by zero is handled safely.
    """
    def safe_divide(numerator, denominator):
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            return None
        return round(numerator/denominator, 4)
    
    revenue = raw.get('revenue')
    gross_profit = raw.get('gross_profit')
    operating_income = raw.get('operating_income')
    net_income = raw.get('net_income')
    total_assets = raw.get('total_assets')
    current_assets = raw.get('current_assets')
    current_liab = raw.get('current_liabilities')
    total_debt = raw.get('total_debt')
    equity = raw.get('stockholders_equity')
    interest_expense = raw.get('interest_expense')

    return {
        'ticker': ticker,
        'company': company,
        'sector': sector,
        'year': year,
        'revenue': revenue,
        'net_income': net_income,
        'total_assets': total_assets,
        'gross_margin': safe_divide(gross_profit, revenue),
        'operating_margin': safe_divide(operating_income, revenue),
        'net_margin': safe_divide(net_income, revenue),
        'current_ratio': safe_divide(current_assets, current_liab),
        'debt_to_equity': safe_divide(total_debt, equity),
        'return_on_assets': safe_divide(net_income, total_assets),
        'interest_coverage': safe_divide(operating_income, interest_expense)
    }



# Main function to orchestrate the data extraction and processing
print("=" * 65)
print("  SEC EDGAR Financial Ratios Extraction")
print("=" * 65)
 
# Load filings metadata from previous script
if not input_path.exists():
    raise SystemExit(f"✗ Could not find {input_path}. Run sec_pull_local.py first.")
 
meta_df = pd.read_csv(input_path)
print(f"\n✓ Loaded {len(meta_df)} filings from {input_path}")
 
# Get unique companies — one XBRL fetch per company not per filing
companies = meta_df[["ticker", "company", "cik", "sector"]].drop_duplicates("ticker")
print(f"✓ Processing {len(companies)} companies\n")
 
all_ratios = []
 
for _, row in companies.iterrows():
    ticker  = row["ticker"]
    company = row["company"]
    cik     = str(row["cik"]).zfill(10)
    sector  = row["sector"]
 
    print(f"  [{sector}] {ticker} — {company}")
 
    # Fetch all XBRL facts for this company (one request covers all years)
    facts = fetch_company_facts(cik)
    if facts is None:
        print(f"    ✗ Skipping {ticker} — could not fetch XBRL data")
        continue
 
    # Get the years available for this company from the metadata
    company_years = meta_df[meta_df["ticker"] == ticker]["year"].astype(str).tolist()
 
    for year in company_years:
        # Extract each raw financial figure for this year
        raw = {}
        for concept in concepts:
            raw[concept] = extract_concept_value(facts, concept, year)
 
        # Calculate ratios from raw figures
        ratios = calculate_ratios(raw, ticker, company, sector, year)
        all_ratios.append(ratios)
 
        # Show which ratios were found vs missing
        found   = sum(1 for k, v in ratios.items() if v is not None and k not in ["ticker", "company", "sector", "year"])
        missing = sum(1 for k, v in ratios.items() if v is None and k not in ["ticker", "company", "sector", "year"])
        print(f"    ✓ {year} — {found} values extracted, {missing} missing")
 
    time.sleep(0.5)  # SEC rate limit
 
# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
df = pd.DataFrame(all_ratios)
 
print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")
print(f"  Total rows     : {len(df)}")
print(f"  Companies      : {df['ticker'].nunique()}")
print(f"  Years          : {sorted(df['year'].unique())}")
print(f"  Missing values : {df.isnull().sum().sum()} total NaNs across all columns")
print(f"{'=' * 65}\n")
 
# Show missing value breakdown per column
print("  Missing values per ratio:")
ratio_cols = ["gross_margin", "operating_margin", "net_margin",
              "current_ratio", "debt_to_equity", "return_on_assets", "interest_coverage"]
for col in ratio_cols:
    missing = df[col].isnull().sum()
    pct     = round(missing / len(df) * 100, 1)
    status  = "⚠" if pct > 20 else "✓"
    print(f"    {status} {col:25s} {missing} missing ({pct}%)")
 
output_path = output_dir / "financial_ratios.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")
print(f"\nNext step: Run sec_extract_text.py to pull Risk Factors text from each filing.")


