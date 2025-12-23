from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import get_paths
from .data_io import load_tables
from .features import FeatureConfig, build_member_features


def build_model(numeric_cols: list[str], use_gradient_boosting: bool = True) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
        ],
        remainder="drop",
    )

    if use_gradient_boosting:
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
    else:
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=None,
        )

    return Pipeline([("pre", pre), ("clf", clf)])


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
    model = build_model(feature_cols)

    model.fit(X_train[feature_cols], y_train)

    proba_test = model.predict_proba(X_test[feature_cols])[:, 1]
    auc_roc = roc_auc_score(y_test, proba_test)
    auc_prc = average_precision_score(y_test, proba_test)

    # Simple thresholding for report (0.5). We'll revisit threshold later.
    pred_test = (proba_test >= 0.5).astype(int)
    report = classification_report(y_test, pred_test, target_names=["no_churn", "churn"])

    print(f"ROC-AUC_test = {auc_roc:.6f}")
    print(f"AUC-PRC_test = {auc_prc:.6f}")
    print(report)

    # Persist artifacts for later ranking work
    joblib.dump(model, paths.artifacts_dir / "baseline_model.joblib")

    meta = {
        "feature_config": asdict(feat_cfg),
        "train_observation_end": str(train_obs_end),
        "feature_cols": feature_cols,
        "test_auc_roc": float(auc_roc),
        "test_auc_prc": float(auc_prc),
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


