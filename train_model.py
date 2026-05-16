"""
Model Training and Evaluation
===============================
Run this locally with: python train_model.py

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn

What this script does:
    Step 1 — Loads the feature matrix from build_features.py
    Step 2 — Trains 3 models: Logistic Regression, Random Forest, XGBoost
    Step 3 — Evaluates each model using stratified cross validation
    Step 4 — Compares models and selects the best one
    Step 5 — Analyzes feature importance
    Step 6 — Saves results and plots to data/results/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    make_scorer,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
FEATURES_PATH = Path("data/processed/features.csv")
OUTPUT_DIR    = Path("data/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: Load data
# ─────────────────────────────────────────────
print("=" * 65)
print("  Model Training and Evaluation")
print("=" * 65)

print("\n[1/6] Loading feature matrix...")
df = pd.read_csv(FEATURES_PATH)

# Separate features from target and metadata
META_COLS   = ["ticker", "company", "sector", "year"]
TARGET_COL  = "financial_deterioration"
FEATURE_COLS = [c for c in df.columns if c not in META_COLS + [TARGET_COL]]

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"  Features : {X.shape[1]}")
print(f"  Samples  : {X.shape[0]}")
print(f"  Positive : {y.sum()} ({round(y.mean()*100, 1)}%)")
print(f"  Negative : {(y==0).sum()} ({round((1-y.mean())*100, 1)}%)")


# ─────────────────────────────────────────────
# STEP 2: Define models
# ─────────────────────────────────────────────
print("\n[2/6] Defining models...")

"""
Why these three models:

Logistic Regression — the baseline. Simple, interpretable, works well
on high dimensional data. With class_weight='balanced' it handles
the 25/75 imbalance. L2 regularization (default) prevents overfitting
on our 56-row dataset.

Random Forest — handles non-linear relationships between features.
Good at ignoring irrelevant features which matters since we have
100 TF-IDF features many of which may not be predictive.

We are NOT using XGBoost or deep learning — with only 56 rows
these models will massively overfit and produce misleading results.
Simple models with proper cross validation are more honest here.
"""

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            class_weight = "balanced",
            max_iter     = 1000,
            C            = 0.1,        # strong regularization for small dataset
            random_state = 42,
        ))
    ]),

    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators  = 100,
            max_depth     = 3,         # shallow trees prevent overfitting
            class_weight  = "balanced",
            random_state  = 42,
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingClassifier(
            n_estimators  = 100,
            max_depth     = 2,      # very shallow to prevent overfitting
            learning_rate = 0.05,   # slow learning rate for small dataset
            random_state  = 42,
        ))
    ]),
}

for name in models:
    print(f"  ✓ {name}")


# ─────────────────────────────────────────────
# STEP 3: Cross validation
# ─────────────────────────────────────────────
print("\n[3/6] Running stratified cross validation...")

"""
Why stratified k-fold cross validation:

With only 56 rows a simple train/test split would give us maybe
11 test samples — far too few to trust any metric. Cross validation
uses all the data for both training and testing across multiple folds.

Stratified means each fold preserves the 25/75 class ratio so we
never accidentally get a fold with no positive examples.

We use 5 folds — standard choice that balances bias and variance.
Each fold trains on 45 rows and tests on 11 rows.
"""

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "roc_auc": "roc_auc",
    "f1":      make_scorer(f1_score, zero_division=0),
    "precision": "precision",
    "recall":    "recall",
}

cv_results = {}

for name, pipeline in models.items():
    print(f"\n  {name}...")
    scores = cross_validate(
        pipeline, X, y,
        cv            = cv,
        scoring       = scoring,
        return_train_score = True,
    )
    cv_results[name] = scores

    # Mean and std across folds
    print(f"    ROC-AUC  : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")
    print(f"    F1 Score : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"    Precision: {scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}")
    print(f"    Recall   : {scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}")

    # Check for overfitting — large gap between train and test score
    train_auc = scores['train_roc_auc'].mean()
    test_auc  = scores['test_roc_auc'].mean()
    gap       = train_auc - test_auc
    if gap > 0.15:
        print(f"    ⚠ Possible overfitting: train AUC {train_auc:.3f} vs test AUC {test_auc:.3f}")
    else:
        print(f"    ✓ No significant overfitting detected (gap: {gap:.3f})")


# ─────────────────────────────────────────────
# STEP 4: Select best model and final evaluation
# ─────────────────────────────────────────────
print("\n[4/6] Selecting best model...")

# Rank by ROC-AUC — best metric for imbalanced binary classification
best_name  = max(cv_results, key=lambda n: cv_results[n]["test_roc_auc"].mean())
best_score = cv_results[best_name]["test_roc_auc"].mean()
print(f"  Best model: {best_name} (ROC-AUC: {best_score:.3f})")

# Print comparison table
print(f"\n  {'Model':<25} {'ROC-AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print(f"  {'-'*65}")
for name, scores in cv_results.items():
    marker = " ←" if name == best_name else ""
    print(
        f"  {name:<25}"
        f" {scores['test_roc_auc'].mean():>10.3f}"
        f" {scores['test_f1'].mean():>10.3f}"
        f" {scores['test_precision'].mean():>10.3f}"
        f" {scores['test_recall'].mean():>10.3f}"
        f"{marker}"
    )

# Fit best model on full dataset for feature importance analysis
best_pipeline = models[best_name]
best_pipeline.fit(X, y)


# ─────────────────────────────────────────────
# STEP 5: Feature importance
# ─────────────────────────────────────────────
print("\n[5/6] Analyzing feature importance...")

"""
Feature importance tells us which features the model
actually relied on to make predictions. This is one of
the most valuable parts of the project — you can say
'the model found that net_margin and the term liquidity_risk
were the strongest predictors of financial deterioration.'
"""

if best_name == "Logistic Regression":
    # For logistic regression, coefficients show feature importance
    # Positive coefficient = increases probability of deterioration
    # Negative coefficient = decreases probability
    model      = best_pipeline.named_steps["model"]
    coefs      = model.coef_[0]
    importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": np.abs(coefs),
        "direction":  ["increases risk" if c > 0 else "decreases risk" for c in coefs],
    }).sort_values("importance", ascending=False)

elif best_name == "Random Forest":
    model      = best_pipeline.named_steps["model"]
    importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
        "direction":  ["—"] * len(FEATURE_COLS),
    }).sort_values("importance", ascending=False)

# Show top 15 features
print(f"\n  Top 15 most important features ({best_name}):")
print(f"  {'Feature':<40} {'Importance':>12} {'Direction':>20}")
print(f"  {'-'*75}")
for _, row in importance.head(15).iterrows():
    print(f"  {row['feature']:<40} {row['importance']:>12.4f} {row['direction']:>20}")

# Separate financial vs NLP features in top 15
top15        = importance.head(15)
top_financial = top15[~top15["feature"].str.startswith("tfidf_")]
top_nlp       = top15[top15["feature"].str.startswith("tfidf_")]
print(f"\n  Financial features in top 15 : {len(top_financial)}")
print(f"  NLP features in top 15      : {len(top_nlp)}")


# ─────────────────────────────────────────────
# STEP 6: Save results and plots
# ─────────────────────────────────────────────
print("\n[6/6] Saving results and plots...")

# Save CV results summary
results_rows = []
for name, scores in cv_results.items():
    results_rows.append({
        "model":          name,
        "roc_auc_mean":   round(scores["test_roc_auc"].mean(),   3),
        "roc_auc_std":    round(scores["test_roc_auc"].std(),    3),
        "f1_mean":        round(scores["test_f1"].mean(),        3),
        "f1_std":         round(scores["test_f1"].std(),         3),
        "precision_mean": round(scores["test_precision"].mean(), 3),
        "recall_mean":    round(scores["test_recall"].mean(),    3),
    })

results_df = pd.DataFrame(results_rows)
results_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)

# Save feature importance
importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

# Plot 1: Model comparison bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x     = np.arange(len(results_df))
width = 0.35
ax.bar(x - width/2, results_df["roc_auc_mean"], width, label="ROC-AUC", color="#2196F3", alpha=0.8)
ax.bar(x + width/2, results_df["f1_mean"],      width, label="F1 Score", color="#4CAF50", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(results_df["model"], fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Cross Validation Scores")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150)
plt.close()
print(f"  ✓ Saved model_comparison.png")

# Plot 2: Top 20 feature importance
fig, ax = plt.subplots(figsize=(10, 8))
top20   = importance.head(20)
colors  = ["#2196F3" if not f.startswith("tfidf_") else "#FF9800" for f in top20["feature"]]
ax.barh(range(len(top20)), top20["importance"], color=colors, alpha=0.8)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([f.replace("tfidf_", "") for f in top20["feature"]], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Importance Score")
ax.set_title(f"Top 20 Feature Importances ({best_name})\nBlue = Financial Ratio  |  Orange = NLP Term")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
plt.close()
print(f"  ✓ Saved feature_importance.png")

print(f"\n{'=' * 65}")
print(f"  FINAL SUMMARY")
print(f"{'=' * 65}")
print(f"  Best model       : {best_name}")
print(f"  ROC-AUC          : {best_score:.3f}")
print(f"  Dataset size     : {len(df)} rows")
print(f"  Features used    : {len(FEATURE_COLS)} (4 financial + 100 NLP)")
print(f"  Validation       : 5-fold stratified cross validation")
print(f"  Class handling   : class_weight='balanced'")
print(f"{'=' * 65}")
print(f"\n✓ All results saved to data/results/")
print(f"\nYour resume bullet:")
print(f"  'Built a financial deterioration classifier using SEC 10-K filings")
print(f"   combining 4 financial ratios with 100 TF-IDF NLP features across")
print(f"   19 companies. Compared Logistic Regression and Random Forest using")
print(f"   5-fold stratified cross validation, achieving ROC-AUC of {best_score:.2f}.'")