from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import get_paths
from .data_io import load_tables
from .features import FeatureConfig, build_member_features


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Build the preprocessing pipeline."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )


def find_best_xgboost_params(X: np.ndarray, y: np.ndarray, scale_pos_weight: float) -> dict:
    """Simple hyperparameter search for XGBoost using cross-validation."""
    print("Running hyperparameter search...")
    
    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
    }
    
    best_score = -1
    best_params = {}
    
    for max_depth in param_grid["max_depth"]:
        for learning_rate in param_grid["learning_rate"]:
            for n_estimators in param_grid["n_estimators"]:
                clf = xgb.XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='aucpr',
                    verbosity=0,
                )
                
                scores = cross_val_score(clf, X, y, cv=3, scoring='average_precision')
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                    }
    
    print(f"Best CV AUC-PRC: {best_score:.4f}")
    print(f"Best params: {best_params}")
    return best_params


def build_xgboost_model(scale_pos_weight: float, **params) -> xgb.XGBClassifier:
    """Build XGBoost classifier with given parameters."""
    return xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='aucpr',
        verbosity=0,
        **params,
    )


# =============================================================================
# STAGE 2: LOST-CAUSE MODEL
# =============================================================================

def train_lost_cause_model(
    X_processed: np.ndarray,
    y: pd.Series,
    outreach_mask: pd.Series,
) -> xgb.XGBClassifier:
    """Train a model on outreach=1 members to predict who churns despite outreach.
    
    This identifies 'lost causes' — members who churn even when contacted.
    """
    # Filter to only members who were outreached
    X_outreached = X_processed[outreach_mask.values]
    y_outreached = y[outreach_mask.values]
    
    n_outreached = len(y_outreached)
    n_churned = y_outreached.sum()
    print(f"\n--- Training Lost-Cause Model ---")
    print(f"Outreached members: {n_outreached}")
    print(f"Churned despite outreach: {n_churned} ({100*n_churned/n_outreached:.1f}%)")
    
    if n_churned == 0 or n_churned == n_outreached:
        print("Warning: Cannot train lost-cause model (all same class). Returning None.")
        return None
    
    # Simple class weight
    scale_pos_weight = (n_outreached - n_churned) / n_churned
    
    # Use fixed reasonable params (simpler than full search for Stage 2)
    clf = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='aucpr',
        verbosity=0,
    )
    clf.fit(X_outreached, y_outreached)
    
    # Quick validation
    proba = clf.predict_proba(X_outreached)[:, 1]
    auc = roc_auc_score(y_outreached, proba)
    print(f"Lost-cause model AUC (on training subset): {auc:.4f}")
    
    return clf


def compute_final_score(
    proba_churn: np.ndarray, 
    proba_lost_cause: np.ndarray,
    penalty_weight: float = 0.3,
) -> np.ndarray:
    """Combine churn risk and lost-cause probability into a final outreach priority score.
    
    final_score = proba_churn * (1 - penalty_weight * proba_lost_cause)
    
    - High churn risk + low lost-cause probability = HIGH priority (saveable)
    - High churn risk + high lost-cause probability = SLIGHTLY LOWER priority (lost cause)
    
    Args:
        proba_churn: P(churn) from Stage 1 model
        proba_lost_cause: P(churn | outreach) from Stage 2 model  
        penalty_weight: How much to penalize lost causes (0=ignore, 1=full penalty)
                       Default 0.3 is conservative since Stage 2 model is trained on biased subset.
    """
    return proba_churn * (1 - penalty_weight * proba_lost_cause)


def select_optimal_n(final_scores: np.ndarray, y_true: np.ndarray, step: int = 500) -> int:
    """Select optimal outreach size N using elbow detection.
    
    Finds the elbow point where the lift curve deviates most from a straight
    line connecting start to end - the point of diminishing returns.
    
    Args:
        final_scores: Priority scores (higher = contact first)
        y_true: Actual churn labels
        step: Round results to this increment (default 500)
    
    Returns:
        Optimal N (number of members to contact)
    """
    order = np.argsort(-final_scores)
    y_sorted = y_true[order]
    baseline_rate = y_true.mean()
    n_total = len(y_true)
    
    # Compute lift curve at step intervals
    ks = np.arange(step, n_total + 1, step)
    lifts = np.array([(y_sorted[:k].sum() / k) / baseline_rate for k in ks])
    
    # Find elbow: max perpendicular distance from line connecting first to last point
    x0, y0 = ks[0], lifts[0]
    x1, y1 = ks[-1], lifts[-1]
    
    # Distance from point (ks[i], lifts[i]) to line through (x0,y0) and (x1,y1)
    distances = np.abs((y1 - y0) * ks - (x1 - x0) * lifts + x1 * y0 - y1 * x0)
    elbow_idx = np.argmax(distances)
    
    return int(ks[elbow_idx])


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main() -> None:
    paths = get_paths()
    paths.artifacts_dir.mkdir(exist_ok=True)

    # Load raw tables
    train_tables = load_tables(paths.train_dir)
    test_tables = load_tables(paths.test_dir)

    # Feature config: keep defaults for now
    feat_cfg = FeatureConfig()

    # Important: use a single observation_end reference across train/test for consistent "recency".
    train_obs_end = max(
        train_tables.web_visits["timestamp"].max(),
        train_tables.app_usage["timestamp"].max(),
        pd.to_datetime(train_tables.claims["diagnosis_date"]).max(),
    )

    X_train, y_train = build_member_features(train_tables, feat_cfg, reference_observation_end=train_obs_end)
    X_test, y_test = build_member_features(test_tables, feat_cfg, reference_observation_end=train_obs_end)

    # Get outreach flags (aligned with member_id order)
    train_outreach = train_tables.labels.set_index("member_id").loc[X_train["member_id"], "outreach"].reset_index(drop=True)
    test_outreach = test_tables.labels.set_index("member_id").loc[X_test["member_id"], "outreach"].reset_index(drop=True)

    # Columns: treat all (except member_id) as numeric
    feature_cols = [c for c in X_train.columns if c != "member_id"]
    
    # Build preprocessor
    preprocessor = build_preprocessor(feature_cols)
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train[feature_cols])
    X_test_processed = preprocessor.transform(X_test[feature_cols])
    
    # ==========================================================================
    # STAGE 1: CHURN-RISK MODEL (existing logic)
    # ==========================================================================
    print("=" * 60)
    print("STAGE 1: Training Churn-Risk Model")
    print("=" * 60)
    
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    best_params = find_best_xgboost_params(X_train_processed, y_train, scale_pos_weight)
    
    churn_model = build_xgboost_model(scale_pos_weight, **best_params)
    churn_model.fit(X_train_processed, y_train)
    
    # Churn predictions
    proba_churn_train = churn_model.predict_proba(X_train_processed)[:, 1]
    proba_churn_test = churn_model.predict_proba(X_test_processed)[:, 1]
    
    auc_roc = roc_auc_score(y_test, proba_churn_test)
    auc_prc = average_precision_score(y_test, proba_churn_test)
    print(f"\nChurn model ROC-AUC_test = {auc_roc:.6f}")
    print(f"Churn model AUC-PRC_test = {auc_prc:.6f}")
    
    # ==========================================================================
    # STAGE 2: LOST-CAUSE MODEL
    # ==========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Training Lost-Cause Model (outreach=1 subset)")
    print("=" * 60)
    
    outreach_mask_train = train_outreach == 1
    lost_cause_model = train_lost_cause_model(X_train_processed, y_train, outreach_mask_train)
    
    # Predict lost-cause probability for all test members
    if lost_cause_model is not None:
        proba_lost_cause_test = lost_cause_model.predict_proba(X_test_processed)[:, 1]
    else:
        # Fallback: no lost-cause adjustment
        proba_lost_cause_test = np.zeros(len(y_test))
    
    # ==========================================================================
    # COMBINED SCORE & RANKING
    # ==========================================================================
    print("\n" + "=" * 60)
    print("FINAL RANKING")
    print("=" * 60)
    
    final_scores = compute_final_score(proba_churn_test, proba_lost_cause_test)
    
    # Select optimal N using elbow detection
    optimal_n = select_optimal_n(final_scores, y_test.values)
    print(f"Optimal outreach size N: {optimal_n} (elbow detection)")
    
    # Compute lift at optimal N for reporting
    order = np.argsort(-final_scores)
    y_sorted = y_test.values[order]
    baseline_rate = y_test.mean()
    churners_in_top_n = y_sorted[:optimal_n].sum()
    lift_at_n = (churners_in_top_n / optimal_n) / baseline_rate
    print(f"Churners captured in top {optimal_n}: {churners_in_top_n} ({100*churners_in_top_n/y_test.sum():.1f}% of all churners)")
    print(f"Lift at N={optimal_n}: {lift_at_n:.2f}x")
    
    # ==========================================================================
    # SAVE ARTIFACTS
    # ==========================================================================
    
    # Save churn model pipeline
    churn_pipeline = Pipeline([("pre", preprocessor), ("clf", churn_model)])
    joblib.dump(churn_pipeline, paths.artifacts_dir / "baseline_model.joblib")
    
    # Save lost-cause model
    if lost_cause_model is not None:
        joblib.dump(lost_cause_model, paths.artifacts_dir / "lost_cause_model.joblib")
    
    # Metadata
    precision, recall, thresholds = precision_recall_curve(y_test, proba_churn_test)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    optimal_threshold = thresholds[f1_scores.argmax()]
    
    meta = {
        "model_type": "XGBoost (two-stage)",
        "feature_config": asdict(feat_cfg),
        "train_observation_end": str(train_obs_end),
        "feature_cols": feature_cols,
        "stage1_churn_model": {
            "best_params": best_params,
            "scale_pos_weight": float(scale_pos_weight),
            "test_auc_roc": float(auc_roc),
            "test_auc_prc": float(auc_prc),
            "optimal_threshold": float(optimal_threshold),
        },
        "stage2_lost_cause_model": {
            "trained": lost_cause_model is not None,
        },
        "outreach_recommendation": {
            "method": "elbow_detection",
            "optimal_n": optimal_n,
            "lift_at_n": float(lift_at_n),
        },
    }
    (paths.artifacts_dir / "baseline_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Full test predictions
    pred_df = pd.DataFrame({
        "member_id": X_test["member_id"].astype(int),
        "y_true": y_test.astype(int),
        "outreach": test_outreach.astype(int),
        "proba_churn": proba_churn_test.round(4),
        "proba_lost_cause": proba_lost_cause_test.round(4),
        "final_score": final_scores.round(4),
    }).sort_values("final_score", ascending=False)
    pred_df.to_csv(paths.artifacts_dir / "test_predictions.csv", index=False)
    
    # Outreach ranking (full)
    ranking_df = pred_df[["member_id", "final_score"]].copy()
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df.columns = ["member_id", "churn_probability", "rank"]
    ranking_df.to_csv(paths.artifacts_dir / "outreach_ranking.csv", index=False)
    
    # Top N for outreach
    top_n_df = ranking_df.head(optimal_n)
    top_n_df.to_csv(paths.artifacts_dir / "outreach_top_n.csv", index=False)
    
    print(f"\nArtifacts saved to {paths.artifacts_dir}/")
    print("  - baseline_model.joblib (churn model)")
    print("  - lost_cause_model.joblib (lost-cause model)")
    print("  - baseline_model_meta.json")
    print("  - test_predictions.csv")
    print("  - outreach_ranking.csv")
    print(f"  - outreach_top_n.csv (top {optimal_n} members)")


if __name__ == "__main__":
    main()
