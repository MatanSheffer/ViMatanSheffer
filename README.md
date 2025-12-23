# WellCo Member Churn Prediction

A machine learning solution to identify members at risk of churning for targeted outreach intervention.

## Executive Summary

WellCo is experiencing increased member churn. This solution provides:
1. **A ranked list of members** prioritized by churn risk for outreach
2. **Recommended outreach size (n=2,000)** based on cost-efficiency analysis
3. **1.77x lift over random** — contacting the top 20% captures ~35% of churners

| Metric | Random Baseline | Our Model | Improvement |
|--------|-----------------|-----------|-------------|
| **ROC-AUC** | 0.489 | **0.666** | +36% |
| **AUC-PRC** | 0.200 | **0.329** | +65% |
| **Churn Recall @ optimal threshold** | 48% | **68%** | +20 pts |

---

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd ViMatanSheffer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model and generate predictions
python -m src.train_eval

# View evaluation notebook
jupyter notebook notebooks/evaluation.ipynb
```

---

## Deliverables

| Deliverable | Location |
|-------------|----------|
| **Ranked member list** | `artifacts/outreach_ranking.csv` |
| **Recommended top-n list** | `artifacts/outreach_top_n.csv` (n=2,000) |
| **Trained model** | `artifacts/baseline_model.joblib` |
| **Model metadata** | `artifacts/baseline_model_meta.json` |
| **Evaluation notebook** | `notebooks/evaluation.ipynb` |
| **Executive presentation** | `presentation/` |

---

## Approach

### Data Overview

Four data sources covering a 14-day observation window (July 1-14, 2025):

| Source | Description | Key Fields |
|--------|-------------|------------|
| `app_usage.csv` | Mobile app sessions | member_id, timestamp |
| `web_visits.csv` | Website visits | member_id, url, title, description, timestamp |
| `claims.csv` | Medical claims | member_id, icd_code, diagnosis_date |
| `churn_labels.csv` | Target & metadata | member_id, churn, signup_date, outreach |

**Client focus** (per `wellco_client_brief.txt`): Cardiometabolic conditions — E11.9 (Type 2 Diabetes), I10 (Hypertension), Z71.3 (Dietary Counseling).

### Feature Engineering

23 features engineered across four categories:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Tenure** | `tenure_days_at_obs_end` | Longer tenure → established relationship |
| **Engagement Intensity** | `app_sessions_per_day`, `total_engagement_per_tenure` | Activity level relative to membership length |
| **Recency Signals** | `days_since_last_app`, `days_since_last_web`, `min_days_since_activity` | Recent disengagement predicts churn |
| **Engagement Trends** | `app_engagement_trend`, `web_engagement_trend`, `combined_trend` | Declining engagement (early vs late window) |
| **Content Patterns** | `web_health_content_ratio`, `web_topic_engagement_ratio` | Health-focused browsing indicates commitment |
| **Clinical Signals** | `n_focus_code_claims` | Claims for focus ICD codes (E11.9, I10, Z71.3) |
| **Cross-channel** | `digital_to_claims_ratio`, `app_preference_ratio` | Channel engagement balance |

**Feature selection rationale**: Features were selected based on (1) domain relevance to healthcare engagement, (2) predictive power measured by feature importance (Gain > 0.01), and (3) avoiding data leakage (no post-observation features).

### Model

- **Algorithm**: XGBoost with cross-validation hyperparameter tuning
- **Class imbalance handling**: `scale_pos_weight` parameter (~4:1 ratio)
- **Best hyperparameters**: `max_depth=3`, `learning_rate=0.05`, `n_estimators=100`
- **Optimal threshold**: 0.458 (maximizes F1 score)

### Model Evaluation

**Why these metrics?**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.666 | Measures ranking quality — how well the model separates churners from non-churners |
| **AUC-PRC** | 0.329 | Critical for imbalanced data (20% churn rate) — 1.65x above random baseline |
| **Precision@K** | 47% @ top 250 | Of the top-ranked members, what fraction are actual churners |
| **Lift** | 2.24x @ top 5% | How much better than random targeting |

**Precision-Recall tradeoff**: At optimal threshold (0.458), the model achieves 68% recall (captures 2/3 of churners) with 28% precision.

---

## Outreach Data: Treatment Effect Consideration

### The Challenge

The dataset includes an `outreach` flag indicating members who received intervention between observation and churn measurement. This creates a methodological challenge:

```
Observation Window → [Outreach Event] → Churn Measurement
    (features)         (treatment)         (outcome)
```

### Our Approach

1. **Excluded from prediction features**: The outreach flag cannot be used for scoring new members (it's not known at prediction time)

2. **Observed correlation**: Members who received outreach show lower churn rates, but this could be:
   - **Treatment effect**: Outreach actually reduces churn
   - **Selection bias**: High-risk members (identified by other signals) were targeted for outreach

3. **Implication**: The lift analysis in `notebooks/evaluation.ipynb` shows that in top risk deciles, outreach recipients have higher churn than non-recipients — suggesting selection bias (they were already high-risk when selected).

4. **Recommendation**: A/B testing is required to isolate the true treatment effect before scaling outreach.

---

## Selecting Outreach Size (n)

### Methodology

We use lift analysis to determine the optimal outreach size:

| Outreach % | Members (n) | Churners Captured | Lift vs Random |
|------------|-------------|-------------------|----------------|
| 5% | 500 | 224 (11%) | **2.24x** |
| 10% | 1,000 | 395 (20%) | **1.97x** |
| 15% | 1,500 | 505 (25%) | **1.68x** |
| 20% | 2,000 | 709 (35%) | **1.77x** |
| 30% | 3,000 | 957 (48%) | **1.59x** |

### Recommendation: n = 2,000 (Top 20%)

**Rationale:**

1. **Balanced approach**: 20% sits in the middle of the efficient range (10-30%), balancing coverage and precision
2. **Strong lift**: 1.77x lift means we're still nearly twice as effective as random targeting
3. **Meaningful coverage**: Captures 35% of all churners — over one-third of at-risk members
4. **Diminishing returns threshold**: Beyond 30%, lift drops to <1.6x, making additional outreach less cost-effective

**Adjustments based on cost**:
- If outreach is expensive → reduce to n=1,000 (top 10%, 2x lift)
- If outreach is cheap → expand to n=3,000 (top 30%, 1.59x lift)

---

## Data Limitations

The available data has inherent constraints on achievable accuracy:

| Data Source | Limitation | Impact |
|-------------|------------|--------|
| App usage | Only session timestamps, no in-app behavior | Cannot measure engagement depth |
| Web visits | URL/title only, no time-on-page or scroll depth | Limited engagement quality signal |
| Claims | ICD codes and dates only | Minimal clinical context |

**Maximum feature correlations with churn**: ~0.15 (weak individual signals)

**Estimated AUC ceiling**: 0.65-0.70 with current data. Further improvement would require richer data sources (customer support interactions, billing issues, NPS scores, detailed app behavior).

---

## Project Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/
│   ├── train/                          # Training data
│   ├── test/                           # Test data (for final evaluation)
│   ├── wellco_client_brief.txt         # Client context
│   ├── auc_baseline_test.txt           # Baseline metrics
│   └── schema_*.md                     # Data schemas
├── notebooks/
│   ├── eda.ipynb                       # Exploratory data analysis
│   └── evaluation.ipynb                # Model evaluation & business analysis
├── src/
│   ├── config.py                       # Path configuration
│   ├── data_io.py                      # Data loading utilities
│   ├── features.py                     # Feature engineering
│   └── train_eval.py                   # Model training & evaluation
├── artifacts/
│   ├── baseline_model.joblib           # Trained XGBoost model
│   ├── baseline_model_meta.json        # Model metadata & hyperparameters
│   ├── test_predictions.csv            # Full test set predictions
│   ├── outreach_ranking.csv            # Complete ranked member list
│   └── outreach_top_n.csv              # Recommended top 2,000 members
└── presentation/                       # Executive slides
```

---

## Reproducing Results

```bash
# 1. Train model (generates artifacts/)
python -m src.train_eval

# 2. Run evaluation notebook
jupyter nbconvert --execute --inplace notebooks/evaluation.ipynb

# 3. Verify metrics
cat artifacts/baseline_model_meta.json | grep -E "auc|threshold"
```

Expected output:
```
"test_auc_roc": 0.666
"test_auc_prc": 0.329
"optimal_threshold": 0.458
```

---

## Author

Matan Sheffer
