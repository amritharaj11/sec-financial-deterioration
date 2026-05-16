"""
Feature Engineering Pipeline
==============================
Run this locally with: python build_features.py

Requirements:
    pip install pandas numpy scikit-learn

What this script does:
    Step 1 — Loads and cleans financial ratios
    Step 2 — Creates target variable based on YoY financial deterioration
    Step 3 — Engineers NLP features from Risk Factors and MD&A text
    Step 4 — Merges structured + text features into one final feature matrix
    Step 5 — Saves to data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RATIOS_PATH = Path("data/raw/financial_ratios.csv")
TEXTS_PATH  = Path("data/raw/filing_texts.csv")
OUTPUT_DIR  = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# These are the 4 clean ratio features we decided to keep
# based on missing value analysis from the extraction step
RATIO_FEATURES = [
    "return_on_assets",
    "current_ratio",
    "interest_coverage",
    "net_margin",
]

# ─────────────────────────────────────────────
# STEP 1: Load and clean financial ratios
# ─────────────────────────────────────────────
print("=" * 65)
print("  Feature Engineering Pipeline")
print("=" * 65)

print("\n[1/5] Loading and cleaning financial ratios...")

ratios_df = pd.read_csv(RATIOS_PATH)
ratios_df["year"] = ratios_df["year"].astype(str)

print(f"  Loaded {len(ratios_df)} rows")

# Drop JNJ 2020 — completely empty as identified during extraction
before = len(ratios_df)
ratios_df = ratios_df[
    ~((ratios_df["ticker"] == "JNJ") & (ratios_df["year"] == "2020"))
]
print(f"  Dropped {before - len(ratios_df)} rows with known complete data loss (JNJ 2020)")

# Cap extreme outliers at 1st and 99th percentile
# This prevents one unusual year from distorting the whole model
# For example Tesla's interest coverage was extremely negative in 2020
for col in RATIO_FEATURES:
    if col in ratios_df.columns:
        p1  = ratios_df[col].quantile(0.01)
        p99 = ratios_df[col].quantile(0.99)
        before_nulls = ratios_df[col].isnull().sum()
        ratios_df[col] = ratios_df[col].clip(lower=p1, upper=p99)
        print(f"  Clipped {col} to [{p1:.3f}, {p99:.3f}]")

# Fill remaining missing values with sector median
# This is better than global median because financial ratios
# vary significantly by sector — a bank's current_ratio is
# structurally different from a tech company's
print("\n  Filling missing values with sector medians...")
for col in RATIO_FEATURES:
    missing = ratios_df[col].isnull().sum()
    if missing > 0:
        sector_medians = ratios_df.groupby("sector")[col].transform("median")
        ratios_df[col] = ratios_df[col].fillna(sector_medians)
        # If still missing (entire sector has no data), fill with global median
        ratios_df[col] = ratios_df[col].fillna(ratios_df[col].median())
        print(f"  Filled {missing} missing values in {col}")

print(f"\n  Clean ratio shape: {ratios_df.shape}")
print(f"  Remaining NaNs:    {ratios_df[RATIO_FEATURES].isnull().sum().sum()}")


# ─────────────────────────────────────────────
# STEP 2: Create target variable
# ─────────────────────────────────────────────
print("\n[2/5] Creating target variable...")

"""
Target variable: financial_deterioration

A company-year is labeled 1 (deteriorating) if TWO OR MORE
of the following conditions are true compared to the prior year:

    - return_on_assets  declined by more than 20%
    - net_margin        declined by more than 20%
    - current_ratio     declined by more than 15%
    - interest_coverage declined by more than 20%

Requiring 2+ conditions avoids labeling a company as distressed
just because one ratio had a bad year due to a one-time event.
This is a conservative, defensible labeling strategy.
"""

# Sort so prior year rows come before current year rows
ratios_df = ratios_df.sort_values(["ticker", "year"]).reset_index(drop=True)

def pct_change(current, prior):
    """Calculate percentage change, handling None and zero safely."""
    if prior is None or current is None:
        return None
    if prior == 0:
        return None
    return (current - prior) / abs(prior)

labels = []

for _, row in ratios_df.iterrows():
    ticker = row["ticker"]
    year   = row["year"]

    # Find the prior year row for this company
    prior_rows = ratios_df[
        (ratios_df["ticker"] == ticker) &
        (ratios_df["year"]   == str(int(year) - 1))
    ]

    # No prior year available — can't calculate deterioration
    if prior_rows.empty:
        labels.append(None)
        continue

    prior = prior_rows.iloc[0]

    # Count how many ratios deteriorated significantly
    deterioration_count = 0

    roa_change = pct_change(row["return_on_assets"], prior["return_on_assets"])
    if roa_change is not None and roa_change < -0.20:
        deterioration_count += 1

    nm_change = pct_change(row["net_margin"], prior["net_margin"])
    if nm_change is not None and nm_change < -0.20:
        deterioration_count += 1

    cr_change = pct_change(row["current_ratio"], prior["current_ratio"])
    if cr_change is not None and cr_change < -0.15:
        deterioration_count += 1

    ic_change = pct_change(row["interest_coverage"], prior["interest_coverage"])
    if ic_change is not None and ic_change < -0.20:
        deterioration_count += 1

    # Label as deteriorating if 2 or more ratios declined significantly
    labels.append(1 if deterioration_count >= 2 else 0)

ratios_df["financial_deterioration"] = labels

# Drop rows where we couldn't calculate a label (earliest year per company)
ratios_df = ratios_df.dropna(subset=["financial_deterioration"])
ratios_df["financial_deterioration"] = ratios_df["financial_deterioration"].astype(int)

# Report class distribution
total       = len(ratios_df)
n_positive  = ratios_df["financial_deterioration"].sum()
n_negative  = total - n_positive
print(f"  Total labeled rows : {total}")
print(f"  Deteriorating  (1) : {n_positive} ({round(n_positive/total*100, 1)}%)")
print(f"  Stable         (0) : {n_negative} ({round(n_negative/total*100, 1)}%)")

# Warn if severely imbalanced
if n_positive / total < 0.15 or n_positive / total > 0.85:
    print("  ⚠ Class imbalance detected — consider using class_weight='balanced' in your model")


# ─────────────────────────────────────────────
# STEP 3: Engineer NLP features
# ─────────────────────────────────────────────
print("\n[3/5] Engineering NLP features from text...")

texts_df = pd.read_csv(TEXTS_PATH)
texts_df["year"] = texts_df["year"].astype(str)

# Combine Risk Factors and MD&A into one text field per filing
# This gives TF-IDF more signal to work with
texts_df["combined_text"] = (
    texts_df["risk_factors"].fillna("") +
    " " +
    texts_df["mda"].fillna("")
).str.strip()

# Drop rows with no text at all
texts_df = texts_df[texts_df["combined_text"].str.len() > 100]
print(f"  Loaded {len(texts_df)} filings with text")

"""
TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) converts text
into numerical features. Words that appear frequently in one document
but rarely across all documents get high scores — these are the
distinctive, meaningful words.

Parameters chosen:
    max_features=100  — top 100 most informative words only
                        keeps the feature matrix manageable
    ngram_range=(1,2) — single words AND two-word phrases
                        "liquidity risk" is more meaningful than
                        "liquidity" and "risk" separately
    min_df=3          — word must appear in at least 3 documents
                        removes very rare terms that add noise
    max_df=0.85       — ignore words appearing in 85%+ of documents
                        removes boilerplate legal language that
                        appears in every 10-K regardless of health
    stop_words        — removes common English words like "the", "and"
"""
# vectorizer = TfidfVectorizer(
#     max_features  = 30,
#     ngram_range   = (1, 2),
#     min_df        = 3,
#     max_df        = 0.85,
#     stop_words    = "english",
# )

# tfidf_matrix = vectorizer.fit_transform(texts_df["combined_text"])
# tfidf_df     = pd.DataFrame(
#     tfidf_matrix.toarray(),
#     columns = [f"tfidf_{w.replace(' ', '_')}" for w in vectorizer.get_feature_names_out()]
# )

# # Add ticker and year back so we can merge with ratios
# tfidf_df["ticker"] = texts_df["ticker"].values
# tfidf_df["year"]   = texts_df["year"].values


# print(f"  TF-IDF matrix shape : {tfidf_df.shape}")
# print(f"\n  Top 20 most informative terms:")
# feature_names = vectorizer.get_feature_names_out()
# mean_scores   = tfidf_matrix.toarray().mean(axis=0)
# top_indices   = mean_scores.argsort()[::-1][:20]
# for i, idx in enumerate(top_indices):
#     print(f"    {i+1:2d}. {feature_names[idx]:30s} (avg score: {mean_scores[idx]:.4f})")


# ─────────────────────────────────────────────
# STEP 4: Merge structured + NLP features
# ─────────────────────────────────────────────
print("\n[4/5] Merging financial ratios and NLP features...")

# Keep only the columns we need from ratios
ratio_cols  = ["ticker", "company", "sector", "year", "financial_deterioration"] + RATIO_FEATURES
ratios_slim = ratios_df[ratio_cols]

# # Merge on ticker + year
# features_df = ratios_slim.merge(tfidf_df, on=["ticker", "year"], how="left")

# # Report merge quality
# merged_with_text = features_df["tfidf_" + tfidf_df.columns[0].replace("tfidf_", "")].notna().sum()
# print(f"  Rows after merge          : {len(features_df)}")
# print(f"  Rows with NLP features    : {features_df.filter(like='tfidf_').notna().any(axis=1).sum()}")
# print(f"  Rows without NLP features : {features_df.filter(like='tfidf_').isna().any(axis=1).sum()}")

# # Fill any missing TF-IDF values with 0 (word simply didn't appear)
# tfidf_cols = [c for c in features_df.columns if c.startswith("tfidf_")]
# features_df[tfidf_cols] = features_df[tfidf_cols].fillna(0)

# Step 4: No NLP features for this run — structured only
print("\n[4/5] Skipping NLP merge — testing structured features only...")
features_df = ratios_slim.copy()

# ─────────────────────────────────────────────
# STEP 5: Save final feature matrix
# ─────────────────────────────────────────────
print("\n[5/5] Saving final feature matrix...")

output_path = OUTPUT_DIR / "features.csv"
features_df.to_csv(output_path, index=False)

print(f"\n{'=' * 65}")
print(f"  SUMMARY")
print(f"{'=' * 65}")
print(f"  Final dataset shape   : {features_df.shape}")
print(f"  Structured features   : {len(RATIO_FEATURES)}")
# print(f"  NLP features          : {len(tfidf_cols)}")
# print(f"  Total features        : {len(RATIO_FEATURES) + len(tfidf_cols)}")
print(f"  NLP features          : 0 (disabled)")
print(f"  Total features        : {len(RATIO_FEATURES)}")
print(f"  Target variable       : financial_deterioration")
print(f"  Class balance         : {n_positive} positive / {n_negative} negative")
print(f"{'=' * 65}")
print(f"\n✓ Saved to {output_path}")
print(f"\nNext step: Run train_model.py to train and evaluate your model.")