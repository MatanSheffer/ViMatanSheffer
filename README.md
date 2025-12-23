# WellCo Member Churn Prediction

Predictive model to identify members at risk of churning for targeted outreach.

## Problem Statement

WellCo, an employer-sponsored healthcare platform, is experiencing increased member churn. The goal is to:
1. Build a model to predict which members are likely to churn
2. Provide a ranked list of members for prioritized outreach
3. Determine the optimal number of members (n) to contact

## Data

Four data sources covering a 14-day observation window (July 1-14, 2025):
- **app_usage.csv**: Mobile app session events
- **web_visits.csv**: Website page visits with content metadata
- **claims.csv**: Medical claims with ICD-10 diagnosis codes
- **churn_labels.csv**: Target variable (churn), signup date, and outreach flag

Key ICD codes of interest (per client brief): E11.9 (Type 2 Diabetes), I10 (Hypertension), Z71.3 (Dietary Counseling).

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Training & Evaluation
```bash
python -m src.train_eval
```

This will:
- Load train/test data
- Build features from all data sources
- Train a GradientBoosting classifier
- Evaluate on test set and print metrics
- Save model and predictions to `artifacts/`

### Exploratory Data Analysis
Open `notebooks/01_eda.ipynb` in Jupyter:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Approach

### Feature Engineering

Features are built at the member level across three categories:

1. **Engagement Metrics**: Session counts, visit counts, unique days active, sessions per day
2. **Recency Signals**: Days since last activity (app/web/claims), engagement trends (early vs late window)
3. **Content Patterns**: Health content ratio, topic diversity, visits to specific health topics
4. **Clinical Signals**: Claim counts, chronic/acute ratio, presence of focus ICD codes

### Modeling

- **Algorithm**: Gradient Boosting Classifier (sklearn)
- **Handling Imbalance**: ~80/20 class split addressed through model selection and threshold tuning
- **Outreach Variable**: Excluded from prediction features (occurs after observation window)

### Important Note on Outreach

The dataset includes an `outreach` flag indicating members who received intervention between the observation period and churn measurement. This variable:
- Cannot be used as a feature (would be data leakage for new predictions)
- Shows strong correlation with reduced churn (confounded - could be treatment effect or selection bias)
- Is available for stratified analysis of model performance

## Results

| Metric | Baseline (Random) | Our Model |
|--------|-------------------|-----------|
| ROC-AUC | 0.489 | **0.653** |
| AUC-PRC | ~0.20 | **0.313** |

The model significantly outperforms random baseline, enabling prioritized outreach to high-risk members.

## Project Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── train/           # Training data
│   ├── test/            # Test data
│   └── *.md             # Schema documentation
├── notebooks/
│   └── 01_eda.ipynb     # Exploratory analysis
├── src/
│   ├── config.py        # Path configuration
│   ├── data_io.py       # Data loading utilities
│   ├── features.py      # Feature engineering
│   └── train_eval.py    # Model training & evaluation
└── artifacts/
    ├── baseline_model.joblib      # Trained model
    ├── baseline_model_meta.json   # Model metadata
    ├── test_predictions.csv       # Test set predictions
    └── outreach_ranking.csv       # Final ranked member list
```

## Selecting Outreach Size (n)

The optimal outreach size depends on:
1. **Cost per outreach**: Higher costs → target fewer members
2. **Expected churn reduction**: Estimated treatment effect from historical outreach
3. **Precision at various thresholds**: What fraction of contacted members are true churners?

A practical approach: rank members by churn probability, then select the top n where expected benefit (churn reduction × member value) exceeds outreach cost.

## Author

Matan Sheffer

