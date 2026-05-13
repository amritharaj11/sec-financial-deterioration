import requests # Handles all http calls to SEC
import pandas as pd
import time
import json
from pathlib import Path

# Configurations
email = 'genaiupdates24@gmail.com'
output_dir = Path('data/raw')
output_dir.mkdir(parents=True, exist_ok=True)

#Every request to the SEC should include these headers
headers = {
    'User-Agent': f"Student research ({email})", # Identify who is making the requests
    'Accept-Encoding': 'gzip, deflate' #Compress the response to make the download faster
}

#Companies list
companies = {
    'Apple': {'ticker': 'AAPL', 'sector': 'Technology'},
    'Microsoft': {'ticker': 'MSFT', 'sector': 'Technology'},
    'Google': {'ticker': 'GOOGL', 'sector': 'Technology'},
    'Amazon': {'ticker': 'AMZN', 'sector': 'Consumer Discretionary'},
    'Meta': {'ticker': 'META', 'sector': 'Technology'},
    'Tesla': {'ticker': 'TSLA', 'sector': 'Automotive'},
    'NVIDIA': {'ticker': 'NVDA', 'sector': 'Technology'},
    'Johnson & Johnson': {'ticker': 'JNJ', 'sector': 'Healthcare'},
    'Procter & Gamble': {'ticker': 'PG', 'sector': 'Consumer Staples'},
    'Pfizer': {'ticker': 'PFE', 'sector': 'Healthcare'},
    'JPMorgan Chase': {'ticker': 'JPM', 'sector': 'Financial Services'},
    'Goldman Sachs': {'ticker': 'GS', 'sector': 'Financial Services'},
    'Walmart': {'ticker': 'WMT', 'sector': 'Consumer Discretionary'},
    'Target': {'ticker': 'TGT', 'sector': 'Consumer Discretionary'},
    'Exxon Mobil': {'ticker': 'XOM', 'sector': 'Energy'},
    'Chevron': {'ticker': 'CVX', 'sector': 'Energy'},
    # 'Shell': {'ticker': 'SHEL', 'sector': 'Energy'},
    'General Motors': {'ticker': 'GM', 'sector': 'Automotive'},
    'AT&T': {'ticker': 'T', 'sector': 'Communication Services'},
    'Verizon': {'ticker': 'VZ', 'sector': 'Communication Services'}
}

years = ['2020', '2021', '2022', '2023']



# Build ticker to cik lookup table
def build_ticker_cik_lookup():
    'Download the ticker to cik mapping file'
    cache_path = output_dir/'ticker_cik.json'
    if cache_path.exists():
        print('Using cached ticker to cik mapping')
        with open(cache_path) as f:
            return json.load(f)
        
    print('Downloading ticker to cik mapping')
    url = 'https://www.sec.gov/files/company_tickers.json'

    try:
        r=requests.get(url, headers=headers, timeout=15) # Make the request with the headers. Timeout refers to the wait time for a response. If time exceeds, then an error will be raised
        r.raise_for_status() # Raise an error if the request fails into a python exception so that we can see the error message
    except requests.exceptions.RequestException as e:
        raise SystemExit(f'Failed to download ticker map: {e}')
    
    data = r.json() # r is the response object, .json() converts the response to a Python dictionary
    ticker_map = {}
    for entry in data.values():
        ticker = entry['ticker'].upper()
        cik = str(entry['cik_str']).zfill(10) # Pad the CIK with leading zeros to make it 10 characters long
        ticker_map[ticker] = cik

    with open(cache_path, 'w') as f:
        json.dump(ticker_map, f)
    
    print(f'Loaded {len(ticker_map):,} tickers from SEC EDGAR')
    return ticker_map


def resolve_companies(companies, ticker_map):
    '''Resolve each company's ticker to a cik number'''
    resolved={}
    for company_name, info in companies.items(): # items() returns key-value pairs
        ticker = info['ticker'].upper()
        sector = info['sector']

        if ticker in ticker_map:
            resolved[ticker] = {
                'cik': ticker_map[ticker],
                'sector': sector,
                'name': company_name
            }
            print(f'{ticker:6s} -> CIK {ticker_map[ticker]} | {company_name} ({sector})')
        else:
            print(f'{ticker:6s} -> Not found in SEC | {company_name}')
    
    return resolved


# Function to fetch SEC data for a given company and year
def get_10k_filings(ticker, cik, sector,company_name):
    '''Fetch 10-K filings metadata for a company from the SEC's EDGAR database.'''
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'

    # Each company will have its own JSON file at the url containing all thier filing history
    try:
        r = requests.get(url, headers={**headers, 'Host': 'data.sec.gov'}, timeout=10)
        r.raise_for_status()
    
    except requests.exceptions.RequestException as e:
        print(f'Error fetching {company_name}-{ticker}: {e}')
        return []
    
    data = r.json()

    #Combine recent filings and archived older filings
    all_forms = []
    all_dates = []
    all_accNums = []

    recent = data.get('filings', {}).get('recent', {}) # This will first extract the filing section and from that it will return the recent filings. If none present, then it will return an empty dict
    all_forms+= recent.get('form', [])
    all_dates+= recent.get('filingDate', [])
    all_accNums+= recent.get('accessionNumber', [])

    #Pull older filings from the archived section
    for archive in data.get('filings', {}).get('files', []):
        archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
        try:
            ar = requests.get(archive_url, headers = {**headers, 'Host': 'data.sec.gov'}, timeout=10)
            ar.raise_for_status()
            archive_data = ar.json()
            all_forms.extend(archive_data.get('form', []))
            all_dates.extend(archive_data.get('filingDate', []))
            all_accNums.extend(archive_data.get('accessionNumber', []))
        except requests.exceptions.RequestException as e:
            print(f'Error fetching archive {archive["name"]}: {e}')

    # filings = data.get('filings', {})
    # forms = filings.get('form', [])
    # dates = filings.get('filingDate', [])
    # accNums = filings.get('accessionNumber', [])

    results = []

    for form, date, acc in zip(all_forms, all_dates, all_accNums):
        if form == '10-K' and any(date.startswith(y) for y in years):
            clean_acc = acc.replace('-', '')
            cik_int = int(cik)
            results.append({
                "ticker":            ticker,
                "company":           company_name,
                "cik":               cik,
                "sector":            sector,
                "filing_date":       date,
                "year":              date[:4],
                "accession_number":  acc,
                # Open this URL in a browser to inspect the raw filing documents
                "filing_folder_url": (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{clean_acc}/"
                ),
            })
    return results


#Main
print('='*65)
print('SEC 10-K Filings Data Pull')
print('='*65)

#Resolve tickers to ciks
print('\n[1/3] Resolving tickers to cik numbers')
ticker_map = build_ticker_cik_lookup()
resolved = resolve_companies(companies, ticker_map)

if not resolved:
    raise SystemExit('No valid tickers resolved. Check your company dict and email')

#Pull 10-k filings for each company
print(f'\n[2/3] Pulling 10-k filings for years {years}')
all_filings = []
for ticker, info in resolved.items():
    sector = info['sector']
    name = info['name']
    print(f"[{sector}] {ticker} -- {name}")
    filings = get_10k_filings(ticker, info['cik'], info['sector'], info['name'])
    all_filings.extend(filings) # Add the filings to the list of all filings

    if filings:
        for f in filings:
            print(f"{f['year']} -- filed {f['filing_date']}")
    
    else:
        print(f'No 10-k filings found for years{years}')
    
    time.sleep(0.6)

#Save results
print(f'\n[3/3] Saving results')
df = pd.DataFrame(all_filings)

print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")
print(f"  Total filings : {len(df)}")
print(f"  Companies     : {df['ticker'].nunique()}")
print(f"  Sectors       : {df['sector'].nunique()}")
print(f"  Years covered : {sorted(df['year'].unique())}")
print(f"{'=' * 65}\n")
print(df[["ticker", "company", "sector", "year", "filing_date"]].to_string(index=False))
 
output_path = output_dir / "filings_metadata.csv"
df.to_csv(output_path, index=False)
 
print(f"\n✓ Saved to {output_path}")
print(f"\nNext step: Run sec_extract_financials.py to pull financial ratios from each filing.")