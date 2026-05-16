# SEC 10-K Financial Deterioration Classifier

A machine learning pipeline that predicts financial deterioration 
in large cap US companies using structured financial ratios extracted 
from SEC 10-K filings.


## Problem Statement
Traditional financial analysis requires manually reading hundreds 
of pages of SEC filings to assess company health. This project 
automates that process by extracting key financial ratios from 
SEC EDGAR and training a classifier to detect year-over-year 
financial deterioration across 47 large cap US companies.


## Dataset
- Source: SEC EDGAR XBRL API (public, no authentication required)
- Companies: 47 large cap US companies across 8 sectors
- Sectors: Technology, Healthcare, Financial Services, Energy, 
  Consumer Discretionary, Consumer Staples, Automotive, 
  Communication Services, Industrials, Real Estate
- Years: 2020-2023
- Total observations: 131


## Approach
1. Pulled 10-K filing metadata for 47 companies using SEC EDGAR submissions API
2. Extracted structured financial ratios via XBRL API (one call per company covers all years)
3. Extracted Risk Factors and MD&A text sections using BeautifulSoup
4. Engineered target variable based on year-over-year deterioration across 4 financial ratios
5. Compared Logistic Regression and Random Forest using 5-fold stratified cross validation


## Key Finding
Structured financial ratios alone achieved ROC-AUC of 0.748. 
Adding TF-IDF text features extracted from Risk Factors and MD&A 
sections reduced performance to 0.606. This suggests that boilerplate 
legal language in large cap 10-K filings introduces noise rather than 
signal, and that structured ratios are stronger predictors of financial 
deterioration for this company profile.


## Results

| Model               | ROC-AUC | F1 Score |
|---------------------|---------|----------|
| Logistic Regression | 0.748   | best     |
| Random Forest       | 0.751   | —        |

Validation: 5-fold stratified cross validation
Class handling: class_weight='balanced' (25% positive / 75% negative)


## Tech Stack
- Python 3.x
- pandas, numpy — data manipulation
- scikit-learn — modeling and evaluation
- requests, BeautifulSoup — data collection and HTML parsing
- SEC EDGAR XBRL API — structured financial data
- matplotlib, seaborn — visualization


## Project Structure

sec-financial-deterioration/
├── sec_pull_local.py          # pulls 10-K filing metadata from SEC EDGAR
├── sec_extract_financials.py  # extracts financial ratios via XBRL API
├── sec_extract_text.py        # extracts Risk Factors and MD&A text
├── build_features.py          # cleans data and engineers features
├── train_model.py             # trains and evaluates models
├── data/
│   ├── raw/                   # raw extracted data (gitignored)
│   └── processed/             # final feature matrix (gitignored)
└── data/results/              # model comparison plots and metrics


## Future Work
- Expand to 200+ companies to improve model robustness
- Replace TF-IDF with sentiment analysis or forward-looking 
  statement extraction to better capture text signal
- Add macroeconomic features such as interest rates and GDP growth
- Experiment with time-series cross validation to better respect 
  temporal ordering of financial data

  