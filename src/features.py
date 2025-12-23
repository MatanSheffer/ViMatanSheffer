from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .data_io import RawTables


FOCUS_CODES = ("E11.9", "I10", "Z71.3")

# Health vs non-health domains for content categorization
HEALTH_DOMAINS = ("health.wellco", "care.portal", "guide.wellness", "living.better")

# Topic patterns for engagement ratio calculation
TOPIC_PATTERNS: dict[str, str] = {
    "nutrition": r"nutrition|diet|healthy eating|mediterranean|fiber|weight|bmi|cholesterol|lipid",
    "movement": r"movement|exercise|aerobic|strength|cardio|fitness|physical activity",
    "sleep": r"sleep|sleep hygiene|sleep apnea|restorative",
    "stress_mindfulness": r"stress|mindfulness|meditation|resilience|wellbeing|well-being|mental health",
    "diabetes": r"diabetes|glucose|insulin|hba1c|glycemic",
    "blood_pressure": r"hypertension|blood pressure",
}


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering knobs."""

    dedupe_claims_full_row: bool = True
    dedupe_web_visits_full_row: bool = True
    dedupe_app_usage_full_row: bool = True
    topic_patterns: dict[str, str] = None  # type: ignore[assignment]
    early_window_days: int = 7
    late_window_days: int = 7

    def __post_init__(self) -> None:
        if self.topic_patterns is None:
            object.__setattr__(self, "topic_patterns", dict(TOPIC_PATTERNS))


def _compute_temporal_stats(timestamps: pd.Series) -> dict[str, float]:
    """Compute temporal statistics from a series of timestamps."""
    if len(timestamps) < 2:
        return {
            "mean_gap_hours": np.nan,
            "max_gap_hours": np.nan,
            "session_span_days": 0.0,
        }
    
    sorted_ts = timestamps.sort_values()
    gaps = sorted_ts.diff().dropna().dt.total_seconds() / 3600.0  # hours
    
    return {
        "mean_gap_hours": gaps.mean(),
        "max_gap_hours": gaps.max(),
        "session_span_days": (sorted_ts.max() - sorted_ts.min()).total_seconds() / 86400.0,
    }


def _compute_engagement_trend(timestamps: pd.Series, obs_end: pd.Timestamp, early_days: int = 7, late_days: int = 7) -> float:
    """Compare early vs late window engagement to detect trends."""
    if len(timestamps) == 0:
        return 0.0
    
    obs_start = timestamps.min()
    early_cutoff = obs_start + pd.Timedelta(days=early_days)
    late_cutoff = obs_end - pd.Timedelta(days=late_days)
    
    early_count = (timestamps < early_cutoff).sum()
    late_count = (timestamps >= late_cutoff).sum()
    
    total = early_count + late_count + 1
    return (late_count - early_count) / total


def _window_bounds(
    obs_end: pd.Timestamp, window_days: int, offset_days: int = 0
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start/end bounds for a fixed-length window preceding obs_end."""
    window_end = obs_end - pd.Timedelta(days=offset_days)
    window_start = window_end - pd.Timedelta(days=window_days)
    return window_start, window_end


def _window_slice(
    df: pd.DataFrame,
    time_col: str,
    obs_end: pd.Timestamp,
    window_days: int,
    offset_days: int = 0,
) -> pd.DataFrame:
    """Return rows that fall inside a sliding window."""
    window_start, window_end = _window_bounds(obs_end, window_days, offset_days)
    mask = (df[time_col] >= window_start) & (df[time_col] < window_end)
    return df.loc[mask]


def build_member_features(
    tables: RawTables,
    cfg: FeatureConfig | None = None,
    *,
    reference_observation_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build per-member features and return (X, y).
    
    Only includes features with importance >= 0.01 Gain.
    """

    cfg = cfg or FeatureConfig()

    web = tables.web_visits.copy()
    app = tables.app_usage.copy()
    claims = tables.claims.copy()
    labels = tables.labels.copy()

    if cfg.dedupe_web_visits_full_row:
        web = web.drop_duplicates()
    if cfg.dedupe_app_usage_full_row:
        app = app.drop_duplicates()
    if cfg.dedupe_claims_full_row:
        claims = claims.drop_duplicates()

    # Observation reference
    if reference_observation_end is None:
        obs_end = max(
            web["timestamp"].max(),
            app["timestamp"].max(),
            pd.to_datetime(claims["diagnosis_date"]).max(),
        )
    else:
        obs_end = pd.to_datetime(reference_observation_end)

    # ========================================================================
    # APP FEATURES
    # ========================================================================
    
    app_by_member = (
        app.groupby("member_id")
        .agg(app_sessions=("timestamp", "size"), last_app_ts=("timestamp", "max"))
        .reset_index()
    )
    app_by_member["days_since_last_app"] = (obs_end - app_by_member["last_app_ts"]).dt.total_seconds() / 86400.0
    
    # Unique days for sessions_per_day calculation
    app["date"] = app["timestamp"].dt.date
    app_days = app.groupby("member_id")["date"].nunique().reset_index()
    app_days.columns = ["member_id", "app_unique_days"]
    app_by_member = app_by_member.merge(app_days, on="member_id", how="left")
    app_by_member["app_sessions_per_day"] = app_by_member["app_sessions"] / app_by_member["app_unique_days"].clip(lower=1)
    
    # Temporal stats and engagement trend
    app_extra_features = []
    for member_id, group in app.groupby("member_id"):
        ts = group["timestamp"]
        temporal = _compute_temporal_stats(ts)
        trend = _compute_engagement_trend(ts, obs_end, cfg.early_window_days, cfg.late_window_days)
        row = {
            "member_id": member_id,
            "app_mean_gap_hours": temporal["mean_gap_hours"],
            "app_max_gap_hours": temporal["max_gap_hours"],
            "app_session_span_days": temporal["session_span_days"],
            "app_engagement_trend": trend,
        }
        app_extra_features.append(row)
    
    app_extra_df = pd.DataFrame(app_extra_features)
    app_by_member = app_by_member.merge(app_extra_df, on="member_id", how="left")

    # ========================================================================
    # WEB FEATURES
    # ========================================================================
    
    web_by_member = (
        web.groupby("member_id")
        .agg(
            web_visits=("timestamp", "size"),
            last_web_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    web_by_member["days_since_last_web"] = (obs_end - web_by_member["last_web_ts"]).dt.total_seconds() / 86400.0
    
    # Health content ratio
    web["is_health_domain"] = web["url"].str.contains("|".join(HEALTH_DOMAINS), regex=True, na=False)
    health_ratio = web.groupby("member_id")["is_health_domain"].mean().reset_index()
    health_ratio.columns = ["member_id", "web_health_content_ratio"]
    web_by_member = web_by_member.merge(health_ratio, on="member_id", how="left")
    
    # Temporal stats and engagement trend
    web_extra_features = []
    for member_id, group in web.groupby("member_id"):
        ts = group["timestamp"]
        temporal = _compute_temporal_stats(ts)
        trend = _compute_engagement_trend(ts, obs_end, cfg.early_window_days, cfg.late_window_days)
        row = {
            "member_id": member_id,
            "web_mean_gap_hours": temporal["mean_gap_hours"],
            "web_max_gap_hours": temporal["max_gap_hours"],
            "web_session_span_days": temporal["session_span_days"],
            "web_engagement_trend": trend,
        }
        web_extra_features.append(row)
    
    web_extra_df = pd.DataFrame(web_extra_features)
    web_by_member = web_by_member.merge(web_extra_df, on="member_id", how="left")

    # Topic engagement ratio (ratio of visits with any health topic)
    text = (
        web[["title", "url", "description"]]
        .fillna("")
        .astype("string")
        .agg(" ".join, axis=1)
        .str.lower()
    )
    for topic, pattern in cfg.topic_patterns.items():
        web[f"is_{topic}"] = text.str.contains(pattern, regex=True)

    topic_cols = [c for c in web.columns if c.startswith("is_") and c != "is_health_domain"]
    web["any_topic"] = web[topic_cols].any(axis=1)
    
    topic_ratio = web.groupby("member_id")["any_topic"].mean().reset_index()
    topic_ratio.columns = ["member_id", "web_topic_engagement_ratio"]
    web_by_member = web_by_member.merge(topic_ratio, on="member_id", how="left")

    # ========================================================================
    # WEB WINDOW FEATURES (recent vs previous health ratio)
    # ========================================================================
    
    window_days = cfg.early_window_days
    web_recent = _window_slice(web, "timestamp", obs_end, window_days)
    web_prev = _window_slice(web, "timestamp", obs_end, window_days, offset_days=window_days)
    
    web_recent_topic_ratio = (
        web_recent.groupby("member_id")["any_topic"].mean().reset_index(name="web_recent_topic_ratio")
    )
    web_prev_health_ratio = (
        web_prev.groupby("member_id")["is_health_domain"]
        .mean()
        .reset_index(name="web_prev_health_ratio")
    )
    web_recent_health_ratio = (
        web_recent.groupby("member_id")["is_health_domain"]
        .mean()
        .reset_index(name="web_recent_health_ratio")
    )
    
    web_window = (
        web_recent_topic_ratio
        .merge(web_prev_health_ratio, on="member_id", how="outer")
        .merge(web_recent_health_ratio, on="member_id", how="outer")
    ).fillna(0)
    web_window["web_health_ratio_delta"] = (
        web_window["web_recent_health_ratio"] - web_window["web_prev_health_ratio"]
    )
    # Drop intermediate column
    web_window = web_window.drop(columns=["web_recent_health_ratio"])

    # ========================================================================
    # CLAIMS FEATURES
    # ========================================================================
    
    claims["diagnosis_date"] = pd.to_datetime(claims["diagnosis_date"])
    
    claims_by_member = (
        claims.groupby("member_id")
        .agg(
            n_claims=("icd_code", "size"),
            last_claim_date=("diagnosis_date", "max"),
        )
        .reset_index()
    )
    # Keep days_since_last_claim as intermediate for min_days_since_activity
    claims_by_member["days_since_last_claim"] = (
        obs_end - claims_by_member["last_claim_date"]
    ).dt.total_seconds() / 86400.0
    
    # Count of focus code claims
    focus_counts = claims[claims["icd_code"].isin(FOCUS_CODES)].groupby("member_id").size().reset_index()
    focus_counts.columns = ["member_id", "n_focus_code_claims"]
    claims_by_member = claims_by_member.merge(focus_counts, on="member_id", how="left")
    claims_by_member["n_focus_code_claims"] = claims_by_member["n_focus_code_claims"].fillna(0)

    # ========================================================================
    # LABELS / STATIC FEATURES
    # ========================================================================
    
    labels_feat = labels[["member_id", "signup_date", "churn"]].copy()
    labels_feat["tenure_days_at_obs_end"] = (obs_end - labels_feat["signup_date"]).dt.days.astype(float)

    # ========================================================================
    # JOIN ALL FEATURES
    # ========================================================================
    
    df = labels_feat.merge(
        app_by_member.drop(columns=["last_app_ts"]), 
        on="member_id", how="left"
    )
    df = df.merge(
        web_by_member.drop(columns=["last_web_ts"]), 
        on="member_id", how="left"
    )
    df = df.merge(web_window, on="member_id", how="left")
    df = df.merge(
        claims_by_member.drop(columns=["last_claim_date"]), 
        on="member_id", how="left"
    )

    # ========================================================================
    # CROSS-SOURCE INTERACTION FEATURES
    # ========================================================================
    
    # Digital engagement (intermediate for ratios)
    digital_engagement = df["app_sessions"].fillna(0) + df["web_visits"].fillna(0)
    
    # Digital to claims ratio
    df["digital_to_claims_ratio"] = digital_engagement / (df["n_claims"].fillna(0) + 1)
    
    # App preference ratio
    df["app_preference_ratio"] = df["app_sessions"].fillna(0) / (digital_engagement + 1)
    
    # Engagement density
    df["total_engagement_per_tenure"] = digital_engagement / df["tenure_days_at_obs_end"].clip(lower=1)
    
    # Combined recency score
    df["days_since_last_app"] = df["days_since_last_app"].fillna(999)
    df["days_since_last_web"] = df["days_since_last_web"].fillna(999)
    df["days_since_last_claim"] = df["days_since_last_claim"].fillna(999)
    df["min_days_since_activity"] = df[["days_since_last_app", "days_since_last_web", "days_since_last_claim"]].min(axis=1)
    
    # Combined engagement trend
    df["combined_trend"] = (
        df["app_engagement_trend"].fillna(0) + df["web_engagement_trend"].fillna(0)
    ) / 2

    # ========================================================================
    # DROP INTERMEDIATE COLUMNS & FILL MISSING
    # ========================================================================
    
    # Remove intermediate/low-importance columns
    drop_cols = [
        "app_sessions", "app_unique_days", "web_visits",
        "n_claims", "days_since_last_claim",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Fill missing values
    num_cols = [c for c in df.columns if c not in {"member_id", "signup_date", "churn"}]
    for c in num_cols:
        df[c] = df[c].fillna(0.0)

    # ========================================================================
    # FINAL MATRICES
    # ========================================================================
    
    y = df["churn"].astype(int)
    X = df.drop(columns=["churn", "signup_date"])

    return X, y
