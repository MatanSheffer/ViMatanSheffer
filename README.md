# WellCo Churn Prediction & Outreach Optimization

A machine learning solution to predict member churn and prioritize outreach for WellCo, an employer-sponsored healthcare ecosystem. This project implements a **two-stage model approach** that not only predicts who will churn, but also identifies which at-risk members are most likely to be saved by intervention.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Model Performance](#model-performance)
- [Key Deliverables](#key-deliverables)
- [Feature Engineering](#feature-engineering)
- [Outreach Size Selection](#outreach-size-selection)

---

## Problem Statement

WellCo is experiencing increased member churn and seeks to reduce it through targeted outreach. The challenge involves:

1. **Predicting churn risk** — Identify members likely to leave
2. **Prioritizing outreach** — Rank members for contact, maximizing impact per outreach dollar
3. **Determining optimal N** — Find the right number of members to contact given marginal outreach costs

---

## Solution Approach

### Two-Stage Model Architecture

Rather than simply ranking by churn probability, this solution implements a **two-stage approach** that accounts for outreach effectiveness:

| Stage | Model | Purpose |
|-------|-------|---------|
| **Stage 1** | Churn Risk Model | Predicts P(churn) for all members using behavioral and claims features |
| **Stage 2** | Lost-Cause Model | Predicts P(churn \| outreach=1) — trained only on members who received outreach |

**Final Score Formula:**
```
final_score = proba_churn × (1 - penalty_weight × proba_lost_cause)
```

This prioritizes members who are:
- **High churn risk** (need intervention)
- **Low lost-cause probability** (likely to respond to outreach)

### Why This Matters

A naive approach would contact the highest-risk members first. But some high-risk members will churn regardless of intervention ("lost causes"). By modeling who responds to outreach, we can **focus resources on saveable members** and maximize ROI.

---

## Project Structure

```
ViMatanSheffer/
├── artifacts/                    # Model outputs and predictions
│   ├── baseline_model.joblib     # Trained churn model pipeline
│   ├── lost_cause_model.joblib   # Trained lost-cause model
│   ├── baseline_model_meta.json  # Model metadata and parameters
│   ├── outreach_ranking.csv      # Full member ranking
│   ├── outreach_top_n.csv        # Top N members for outreach
│   └── test_predictions.csv      # Complete test set predictions
├── data/
│   ├── train/                    # Training data
│   ├── test/                     # Test data (for evaluation)
│   └── *.md, *.txt               # Schemas and client brief
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis
│   └── evaluation.ipynb          # Model evaluation and lift analysis
├── src/
│   ├── config.py                 # Path configuration
│   ├── data_io.py                # Data loading utilities
│   ├── features.py               # Feature engineering
│   └── train_eval.py             # Training pipeline
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+ recommended
- pip package manager

### Installation Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd ViMatanSheffer

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `pandas>=2.0` — Data manipulation
- `numpy>=1.24` — Numerical computing
- `scikit-learn>=1.3` — ML preprocessing and evaluation
- `xgboost>=2.0` — Gradient boosting model
- `matplotlib>=3.7` & `seaborn>=0.12` — Visualization
- `joblib>=1.3` — Model serialization

---

## Running the Pipeline

### Training & Evaluation

Run the full training pipeline from the project root:

```bash
python -m src.train_eval
```

This will:
1. Load training and test data
2. Engineer features from web visits, app usage, and claims
3. Train Stage 1 (churn) and Stage 2 (lost-cause) models
4. Perform hyperparameter search via cross-validation
5. Generate predictions and rankings
6. Save all artifacts to `artifacts/`

### Exploring the Notebooks

Launch Jupyter to explore the analysis:

```bash
jupyter notebook notebooks/
```

- **`eda.ipynb`** — Exploratory data analysis and feature insights
- **`evaluation.ipynb`** — Model performance, lift curves, and outreach recommendations

---

## Model Performance

### Comparison to Baseline

| Metric | Baseline (Random) | Our Model |
|--------|-------------------|-----------|
| ROC-AUC | 0.489 | **0.666** |
| AUC-PRC | ~0.20 | **0.329** |

### Outreach Effectiveness

| Metric | Value |
|--------|-------|
| Optimal N | 3,500 members |
| Lift at N | **1.51×** |
| Churners captured | ~52% of all churners |

At the recommended outreach size (N=3,500), we capture 1.5× more churners per contact than a random baseline.

---

## Key Deliverables

### 1. Outreach List (`artifacts/outreach_top_n.csv`)

A prioritized list of top N members for outreach:

| Column | Description |
|--------|-------------|
| `member_id` | Unique member identifier |
| `churn_probability` | Combined priority score (0-1) |
| `rank` | Outreach priority (1 = highest) |

### 2. Full Ranking (`artifacts/outreach_ranking.csv`)

Complete ranking of all members for flexible outreach sizing.

### 3. Trained Models

- `baseline_model.joblib` — Full preprocessing + churn model pipeline
- `lost_cause_model.joblib` — Stage 2 lost-cause classifier

---

## Feature Engineering

Features are engineered from three data sources, focusing on **behavioral engagement patterns** that signal churn intent:

### Engagement Recency & Frequency
- `days_since_last_app` / `days_since_last_web` — Recent disengagement signals
- `app_sessions_per_day` — Usage intensity
- `min_days_since_activity` — Most recent touchpoint across all channels

### Temporal Patterns
- `app_engagement_trend` / `web_engagement_trend` — Early vs. late window comparison
- `app_mean_gap_hours` / `web_mean_gap_hours` — Session regularity
- `session_span_days` — Total active duration

### Content Engagement (Health Topics)
- `web_health_content_ratio` — Proportion of health-related page visits
- `web_topic_engagement_ratio` — Engagement with specific wellness topics (nutrition, movement, sleep, stress, diabetes, blood pressure)
- `web_health_ratio_delta` — Change in health content engagement over time

### Clinical Indicators
- `n_focus_code_claims` — Claims for priority conditions (E11.9, I10, Z71.3)
- `tenure_days_at_obs_end` — Member tenure

### Cross-Source Interactions
- `digital_to_claims_ratio` — Digital engagement relative to healthcare utilization
- `app_preference_ratio` — Mobile vs. web channel preference
- `combined_trend` — Aggregated engagement trajectory

---

## Outreach Size Selection

### Method: Elbow Detection

We determine optimal N by finding the **elbow point** in the lift curve — where diminishing returns set in:

1. Compute cumulative lift (churn rate in top K / baseline rate) for varying K
2. Find the point of maximum curvature (greatest deviation from a straight line)
3. This balances **capturing more churners** vs. **diminishing marginal value**

### Result

The elbow detection algorithm recommends **N = 3,500** members, where:
- Lift remains strong (1.51×)
- Additional contacts yield progressively less incremental value
- ~35% of the member base is targeted for intervention

### Incorporating Outreach Data

The training data includes an `outreach` flag indicating members who were previously contacted. Rather than ignoring this signal:

1. **Stage 2 model** is trained exclusively on outreach=1 members to learn who churns despite intervention
2. This enables us to **deprioritize lost causes** and focus on responsive members
3. The combined score reflects both churn risk and outreach responsiveness

---

## Author

Matan Sheffer

