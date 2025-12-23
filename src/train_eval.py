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


def main() -> None:
    paths = get_paths()
    paths.artifacts_dir.mkdir(exist_ok=True)

    # Load raw tables
    train_tables = load_tables(paths.train_dir)
    test_tables = load_tables(paths.test_dir)

    # Feature config: keep defaults for now
    feat_cfg = FeatureConfig()

    # Important: use a single observation_end reference across train/test for consistent "recency".
    # We'll take it from the train observation window.
    train_obs_end = max(
        train_tables.web_visits["timestamp"].max(),
        train_tables.app_usage["timestamp"].max(),
        pd.to_datetime(train_tables.claims["diagnosis_date"]).max(),
    )

    X_train, y_train = build_member_features(train_tables, feat_cfg, reference_observation_end=train_obs_end)
    X_test, y_test = build_member_features(test_tables, feat_cfg, reference_observation_end=train_obs_end)

    # Columns: we already numeric-encoded bools in features.py, so treat all (except member_id) as numeric
    feature_cols = [c for c in X_train.columns if c != "member_id"]
    
    # Build preprocessor
    preprocessor = build_preprocessor(feature_cols)
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train[feature_cols])
    X_test_processed = preprocessor.transform(X_test[feature_cols])
    
    # Calculate scale_pos_weight for class imbalance (ratio of negatives to positives)
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
    
    # Find best hyperparameters via cross-validation
    best_params = find_best_xgboost_params(X_train_processed, y_train, scale_pos_weight)
    
    # Build and train final model with best params
    clf = build_xgboost_model(scale_pos_weight, **best_params)
    clf.fit(X_train_processed, y_train)
    
    # Create pipeline for saving (preprocessor + classifier)
    model = Pipeline([("pre", preprocessor), ("clf", clf)])

    proba_test = clf.predict_proba(X_test_processed)[:, 1]
    auc_roc = roc_auc_score(y_test, proba_test)
    auc_prc = average_precision_score(y_test, proba_test)

    # Find optimal threshold that maximizes F1 score
    precision, recall, thresholds = precision_recall_curve(y_test, proba_test)
    # Avoid division by zero
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]

    print(f"ROC-AUC_test = {auc_roc:.6f}")
    print(f"AUC-PRC_test = {auc_prc:.6f}")
    print(f"Optimal threshold (max F1) = {optimal_threshold:.4f}")

    # Report at default 0.5 threshold
    pred_test_default = (proba_test >= 0.5).astype(int)
    report_default = classification_report(y_test, pred_test_default, target_names=["no_churn", "churn"])
    print("\n--- Classification Report @ threshold=0.5 ---")
    print(report_default)

    # Report at optimal threshold
    pred_test_optimal = (proba_test >= optimal_threshold).astype(int)
    report_optimal = classification_report(y_test, pred_test_optimal, target_names=["no_churn", "churn"])
    print(f"\n--- Classification Report @ threshold={optimal_threshold:.4f} (optimal F1) ---")
    print(report_optimal)

    # Persist artifacts for later ranking work
    joblib.dump(model, paths.artifacts_dir / "baseline_model.joblib")

    meta = {
        "model_type": "XGBoost",
        "feature_config": asdict(feat_cfg),
        "train_observation_end": str(train_obs_end),
        "feature_cols": feature_cols,
        "best_params": best_params,
        "scale_pos_weight": float(scale_pos_weight),
        "test_auc_roc": float(auc_roc),
        "test_auc_prc": float(auc_prc),
        "optimal_threshold": float(optimal_threshold),
    }
    (paths.artifacts_dir / "baseline_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "member_id": X_test["member_id"].astype(int),
            "y_true": y_test.astype(int),
            "proba_churn": proba_test.astype(float),
        }
    ).sort_values("proba_churn", ascending=False)
    pred_df.to_csv(paths.artifacts_dir / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()


