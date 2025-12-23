from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .data_io import RawTables


FOCUS_CODES = ("E11.9", "I10", "Z71.3")

# Chronic vs acute ICD code categorization (first letter/digits indicate category)
CHRONIC_ICD_PREFIXES = ("E11", "I10", "I25", "J44", "J45", "K21", "M54", "G43")  # diabetes, hypertension, heart disease, COPD, asthma, GERD, back pain, migraine
ACUTE_ICD_PREFIXES = ("J00", "A09", "R51", "S", "T")  # cold, gastroenteritis, headache, injuries

TOPIC_PATTERNS: dict[str, str] = {
    "nutrition": r"nutrition|diet|healthy eating|mediterranean|fiber|weight|bmi|cholesterol|lipid",
    "movement": r"movement|exercise|aerobic|strength|cardio|fitness|physical activity",
    "sleep": r"sleep|sleep hygiene|sleep apnea|restorative",
    "stress_mindfulness": r"stress|mindfulness|meditation|resilience|wellbeing|well-being|mental health",
    "diabetes": r"diabetes|glucose|insulin|hba1c|glycemic",
    "blood_pressure": r"hypertension|blood pressure",
}

# Health vs non-health domains for content categorization
HEALTH_DOMAINS = ("health.wellco", "care.portal", "guide.wellness", "living.better")

CLAIMS_RECENT_WINDOW_DAYS = 30


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering knobs.

    Keeping this tiny and explicit is useful later when you want to iterate.
    """

    dedupe_claims_full_row: bool = True
    dedupe_web_visits_full_row: bool = True
    dedupe_app_usage_full_row: bool = True
    max_one_hot_icd: int = 10
    topic_patterns: dict[str, str] = None  # type: ignore[assignment]
    # Time window for early/late engagement comparison (in days)
    early_window_days: int = 7
    late_window_days: int = 7

    def __post_init__(self) -> None:
        if self.topic_patterns is None:
            object.__setattr__(self, "topic_patterns", dict(TOPIC_PATTERNS))


def _compute_temporal_stats(timestamps: pd.Series) -> dict[str, float]:
    """Compute temporal statistics from a series of timestamps.
    
    Simplified: removed cv_gap (too noisy for short windows).
    """
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


def _compute_engagement_trend(timestamps: pd.Series, obs_end: pd.Timestamp, early_days: int = 7, late_days: int = 7) -> dict[str, float]:
    """Compare early vs late window engagement to detect trends.
    
    Simplified: returns only the trend, not raw counts (redundant with other features).
    """
    if len(timestamps) == 0:
        return {"engagement_trend": 0.0}
    
    obs_start = timestamps.min()
    early_cutoff = obs_start + pd.Timedelta(days=early_days)
    late_cutoff = obs_end - pd.Timedelta(days=late_days)
    
    early_count = (timestamps < early_cutoff).sum()
    late_count = (timestamps >= late_cutoff).sum()
    
    # Trend: normalized difference (late - early) / (late + early + 1)
    total = early_count + late_count + 1
    trend = (late_count - early_count) / total
    
    return {"engagement_trend": trend}


def _entropy(counts: pd.Series) -> float:
    """Compute Shannon entropy for diversity measurement."""
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(stats.entropy(probs))


def _window_bounds(
    obs_end: pd.Timestamp, window_days: int, offset_days: int = 0
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return start/end bounds for a fixed-length window preceding obs_end."""
    window_end = obs_end - pd.Timedelta(days=offset_days)
    window_start = window_end - pd.Timedelta(days=window_days)
    return window_start, window_end


def _count_events_in_window(
    df: pd.DataFrame,
    time_col: str,
    obs_end: pd.Timestamp,
    window_days: int,
    member_col: str = "member_id",
    offset_days: int = 0,
) -> pd.Series:
    """Count events per member in a sliding window."""
    window_start, window_end = _window_bounds(obs_end, window_days, offset_days)
    mask = (df[time_col] >= window_start) & (df[time_col] < window_end)
    return df.loc[mask].groupby(member_col).size()


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


def _one_hot_top_k(values: pd.Series, k: int) -> Iterable[str]:
    top = values.value_counts().head(k).index.tolist()
    return [str(x) for x in top]


def build_member_features(
    tables: RawTables,
    cfg: FeatureConfig | None = None,
    *,
    reference_observation_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build per-member features and return (X, y).

    Notes on outreach:
    - The assignment says outreach occurred after the observation window.
    - We include `outreach` as a feature so later we can model / discuss it explicitly.
    - When generating an outreach ranking, we can score with `outreach=0` to represent
      baseline risk before any new intervention.
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

    # Observation reference: used to define "recency" and tenure at end-of-window.
    if reference_observation_end is None:
        obs_end = max(
            web["timestamp"].max(),
            app["timestamp"].max(),
            pd.to_datetime(claims["diagnosis_date"]).max(),
        )
    else:
        obs_end = pd.to_datetime(reference_observation_end)

    # ========================================================================
    # APP FEATURES - Enhanced with temporal patterns
    # ========================================================================
    
    # Basic aggregates
    app_by_member = (
        app.groupby("member_id")
        .agg(app_sessions=("timestamp", "size"), last_app_ts=("timestamp", "max"))
        .reset_index()
    )
    app_by_member["days_since_last_app"] = (obs_end - app_by_member["last_app_ts"]).dt.total_seconds() / 86400.0
    
    # Unique days with app activity (consistency)
    app["date"] = app["timestamp"].dt.date
    app_days = app.groupby("member_id")["date"].nunique().reset_index()
    app_days.columns = ["member_id", "app_unique_days"]
    app_by_member = app_by_member.merge(app_days, on="member_id", how="left")
    app_by_member["app_sessions_per_day"] = app_by_member["app_sessions"] / app_by_member["app_unique_days"].clip(lower=1)
    
    # Temporal stats and engagement trend - computed per member
    app_extra_features = []
    for member_id, group in app.groupby("member_id"):
        ts = group["timestamp"]
        temporal = _compute_temporal_stats(ts)
        trend = _compute_engagement_trend(ts, obs_end, cfg.early_window_days, cfg.late_window_days)
        row = {"member_id": member_id}
        row.update({f"app_{k}": v for k, v in temporal.items()})
        row.update({f"app_{k}": v for k, v in trend.items()})
        app_extra_features.append(row)
    
    app_extra_df = pd.DataFrame(app_extra_features)
    app_by_member = app_by_member.merge(app_extra_df, on="member_id", how="left")

    # ========================================================================
    # WEB FEATURES - Enhanced with content patterns and diversity
    # ========================================================================
    
    # Basic aggregates
    web_by_member = (
        web.groupby("member_id")
        .agg(
            web_visits=("timestamp", "size"),
            last_web_ts=("timestamp", "max"),
            web_unique_urls=("url", pd.Series.nunique),
            web_unique_titles=("title", pd.Series.nunique),
        )
        .reset_index()
    )
    web_by_member["days_since_last_web"] = (obs_end - web_by_member["last_web_ts"]).dt.total_seconds() / 86400.0
    
    # Visits per unique URL (engagement depth)
    web_by_member["visits_per_url"] = web_by_member["web_visits"] / web_by_member["web_unique_urls"].clip(lower=1)
    
    # Health vs non-health content ratio
    web["is_health_domain"] = web["url"].str.contains("|".join(HEALTH_DOMAINS), regex=True, na=False)
    health_ratio = web.groupby("member_id")["is_health_domain"].mean().reset_index()
    health_ratio.columns = ["member_id", "web_health_content_ratio"]
    web_by_member = web_by_member.merge(health_ratio, on="member_id", how="left")
    
    # Unique days with web activity
    web["date"] = web["timestamp"].dt.date
    web_days = web.groupby("member_id")["date"].nunique().reset_index()
    web_days.columns = ["member_id", "web_unique_days"]
    web_by_member = web_by_member.merge(web_days, on="member_id", how="left")
    
    # URL diversity entropy
    url_counts = web.groupby("member_id")["url"].value_counts()
    url_entropy = url_counts.groupby(level=0, group_keys=False).apply(_entropy).reset_index()
    url_entropy.columns = ["member_id", "web_url_entropy"]
    web_by_member = web_by_member.merge(url_entropy, on="member_id", how="left")
    
    # Temporal stats and engagement trend - computed per member
    web_extra_features = []
    for member_id, group in web.groupby("member_id"):
        ts = group["timestamp"]
        temporal = _compute_temporal_stats(ts)
        trend = _compute_engagement_trend(ts, obs_end, cfg.early_window_days, cfg.late_window_days)
        row = {"member_id": member_id}
        row.update({f"web_{k}": v for k, v in temporal.items()})
        row.update({f"web_{k}": v for k, v in trend.items()})
        web_extra_features.append(row)
    
    web_extra_df = pd.DataFrame(web_extra_features)
    web_by_member = web_by_member.merge(web_extra_df, on="member_id", how="left")

    # Topic features (enhanced with diversity)
    text = (
        web[["title", "url", "description"]]
        .fillna("")
        .astype("string")
        .agg(" ".join, axis=1)
        .str.lower()
    )
    for topic, pattern in cfg.topic_patterns.items():
        web[f"is_{topic}"] = text.str.contains(pattern, regex=True)

    topic_cols = [c for c in web.columns if c.startswith("is_") and c not in ["is_health_domain"]]
    web_topics_by_member = (
        web.groupby("member_id")
        .agg(**{f"web_topic_{c}": (c, "sum") for c in topic_cols})
        .reset_index()
    )
    
    # Topic diversity: number of unique topics visited
    topic_matrix = web.groupby("member_id")[topic_cols].any()
    web_topics_by_member["web_topic_diversity"] = topic_matrix.sum(axis=1).values
    
    # Topic engagement ratio (topic visits / total visits)
    web["any_topic"] = web[topic_cols].any(axis=1)
    topic_ratio = web.groupby("member_id")["any_topic"].mean().reset_index()
    topic_ratio.columns = ["member_id", "web_topic_engagement_ratio"]
    web_topics_by_member = web_topics_by_member.merge(topic_ratio, on="member_id", how="left")

    # ========================================================================
    # CLAIMS FEATURES - Enhanced with temporal and clinical patterns
    # ========================================================================
    
    claims["diagnosis_date"] = pd.to_datetime(claims["diagnosis_date"])
    
    claims_by_member = (
        claims.groupby("member_id")
        .agg(
            n_claims=("icd_code", "size"),
            n_unique_icd=("icd_code", pd.Series.nunique),
            last_claim_date=("diagnosis_date", "max"),
            first_claim_date=("diagnosis_date", "min"),
        )
        .reset_index()
    )
    claims_by_member["days_since_last_claim"] = (
        obs_end - claims_by_member["last_claim_date"]
    ).dt.total_seconds() / 86400.0
    claims_by_member["claim_span_days"] = (
        claims_by_member["last_claim_date"] - claims_by_member["first_claim_date"]
    ).dt.total_seconds() / 86400.0
    
    # Chronic vs acute code ratio
    claims["is_chronic"] = claims["icd_code"].str.startswith(CHRONIC_ICD_PREFIXES)
    claims["is_acute"] = claims["icd_code"].str.startswith(ACUTE_ICD_PREFIXES)
    
    chronic_counts = claims.groupby("member_id")[["is_chronic", "is_acute"]].sum().reset_index()
    chronic_counts.columns = ["member_id", "n_chronic_claims", "n_acute_claims"]
    claims_by_member = claims_by_member.merge(chronic_counts, on="member_id", how="left")
    claims_by_member["chronic_ratio"] = claims_by_member["n_chronic_claims"] / claims_by_member["n_claims"].clip(lower=1)
    
    # ICD code diversity entropy
    icd_counts = claims.groupby("member_id")["icd_code"].value_counts()
    icd_entropy = icd_counts.groupby(level=0).apply(_entropy).reset_index()
    icd_entropy.columns = ["member_id", "icd_entropy"]
    claims_by_member = claims_by_member.merge(icd_entropy, on="member_id", how="left")
    
    # Claims frequency (claims per day of span)
    claims_by_member["claims_per_span_day"] = claims_by_member["n_claims"] / claims_by_member["claim_span_days"].clip(lower=1)
    
    # Unique claim days
    claims_days = claims.groupby("member_id")["diagnosis_date"].apply(lambda x: x.dt.date.nunique()).reset_index()
    claims_days.columns = ["member_id", "claims_unique_days"]
    claims_by_member = claims_by_member.merge(claims_days, on="member_id", how="left")

    # Focus ICD flags
    for code in FOCUS_CODES:
        members_with_code = claims.loc[claims["icd_code"] == code, "member_id"].unique()
        claims_by_member[f"has_icd_{code}"] = claims_by_member["member_id"].isin(members_with_code)
    
    # Count of focus code claims (not just binary)
    focus_counts = claims[claims["icd_code"].isin(FOCUS_CODES)].groupby("member_id").size().reset_index()
    focus_counts.columns = ["member_id", "n_focus_code_claims"]
    claims_by_member = claims_by_member.merge(focus_counts, on="member_id", how="left")
    claims_by_member["n_focus_code_claims"] = claims_by_member["n_focus_code_claims"].fillna(0)

    # One-hot top ICDs
    top_icd = _one_hot_top_k(claims["icd_code"], cfg.max_one_hot_icd)
    for code in top_icd:
        members_with = claims.loc[claims["icd_code"] == code, "member_id"].unique()
        claims_by_member[f"icd_{code}"] = claims_by_member["member_id"].isin(members_with)

    # ========================================================================
    # LABELS / STATIC FEATURES
    # ========================================================================
    
    labels_feat = labels[["member_id", "signup_date", "outreach", "churn"]].copy()
    labels_feat["tenure_days_at_obs_end"] = (obs_end - labels_feat["signup_date"]).dt.days.astype(float)
    
    # Tenure buckets (non-linear effects)
    labels_feat["is_new_member"] = (labels_feat["tenure_days_at_obs_end"] < 90).astype(int)
    labels_feat["is_long_tenure"] = (labels_feat["tenure_days_at_obs_end"] > 365).astype(int)

    # ========================================================================
    # JOIN ALL FEATURES
    # ========================================================================
    
    drop_cols_app = ["last_app_ts"]
    drop_cols_web = ["last_web_ts"]
    drop_cols_claims = ["last_claim_date", "first_claim_date"]
    
    df = labels_feat.merge(
        app_by_member.drop(columns=[c for c in drop_cols_app if c in app_by_member.columns]), 
        on="member_id", how="left"
    )
    df = df.merge(
        web_by_member.drop(columns=[c for c in drop_cols_web if c in web_by_member.columns]), 
        on="member_id", how="left"
    )
    df = df.merge(web_topics_by_member, on="member_id", how="left")
    df = df.merge(
        claims_by_member.drop(columns=[c for c in drop_cols_claims if c in claims_by_member.columns]), 
        on="member_id", how="left"
    )

    # ========================================================================
    # CROSS-SOURCE INTERACTION FEATURES
    # ========================================================================
    
    # Multi-channel engagement flags
    df["has_app_activity"] = (df["app_sessions"].fillna(0) > 0).astype(int)
    df["has_web_activity"] = (df["web_visits"].fillna(0) > 0).astype(int)
    df["has_claims"] = (df["n_claims"].fillna(0) > 0).astype(int)
    df["n_active_channels"] = df["has_app_activity"] + df["has_web_activity"] + df["has_claims"]
    
    # Digital engagement ratio (app + web vs claims)
    df["digital_engagement"] = df["app_sessions"].fillna(0) + df["web_visits"].fillna(0)
    df["digital_to_claims_ratio"] = df["digital_engagement"] / (df["n_claims"].fillna(0) + 1)
    
    # App vs web preference
    total_digital = df["digital_engagement"] + 1
    df["app_preference_ratio"] = df["app_sessions"].fillna(0) / total_digital
    
    # Engagement density (activity per tenure day)
    df["total_engagement_per_tenure"] = df["digital_engagement"] / df["tenure_days_at_obs_end"].clip(lower=1)
    
    # Combined recency score (min of recency signals - how recent is most recent activity)
    recency_cols = ["days_since_last_app", "days_since_last_web", "days_since_last_claim"]
    for c in recency_cols:
        df[c] = df[c].fillna(999)  # Large value for missing
    df["min_days_since_activity"] = df[recency_cols].min(axis=1)
    
    # Combined engagement trend
    df["combined_trend"] = (
        df["app_engagement_trend"].fillna(0) + df["web_engagement_trend"].fillna(0)
    ) / 2

    # ========================================================================
    # RECENT WINDOW FEATURES
    # ========================================================================

    window_days = cfg.early_window_days

    # --------------------------------------------------------------------
    # App recency (last window vs previous window)
    # --------------------------------------------------------------------
    app_recent = (
        _count_events_in_window(app, "timestamp", obs_end, window_days)
        .reset_index(name="app_recent_sessions")
    )
    app_prev = (
        _count_events_in_window(app, "timestamp", obs_end, window_days, offset_days=window_days)
        .reset_index(name="app_prev_sessions")
    )
    app_window = app_recent.merge(app_prev, on="member_id", how="outer").fillna(0)
    app_window["app_recent_vs_prev_ratio"] = (
        (app_window["app_recent_sessions"] - app_window["app_prev_sessions"])
        / (app_window["app_prev_sessions"] + 1)
    )
    app_window["app_recent_delta"] = app_window["app_recent_sessions"] - app_window["app_prev_sessions"]
    df = df.merge(app_window, on="member_id", how="left")

    # --------------------------------------------------------------------
    # Web recency (counts, topical focus)
    # --------------------------------------------------------------------
    web_recent = _window_slice(web, "timestamp", obs_end, window_days)
    web_prev = _window_slice(web, "timestamp", obs_end, window_days, offset_days=window_days)
    web_recent_counts = (
        _count_events_in_window(web, "timestamp", obs_end, window_days)
        .reset_index(name="web_recent_visits")
    )
    web_prev_counts = (
        _count_events_in_window(web, "timestamp", obs_end, window_days, offset_days=window_days)
        .reset_index(name="web_prev_visits")
    )
    web_recent_topic_ratio = (
        web_recent.groupby("member_id")["any_topic"].mean().reset_index(name="web_recent_topic_ratio")
    )
    web_recent_health_ratio = (
        web_recent.groupby("member_id")["is_health_domain"]
        .mean()
        .reset_index(name="web_recent_health_ratio")
    )
    web_prev_health_ratio = (
        web_prev.groupby("member_id")["is_health_domain"]
        .mean()
        .reset_index(name="web_prev_health_ratio")
    )
    web_window = (
        web_recent_counts
        .merge(web_prev_counts, on="member_id", how="outer")
        .merge(web_recent_topic_ratio, on="member_id", how="outer")
        .merge(web_recent_health_ratio, on="member_id", how="outer")
        .merge(web_prev_health_ratio, on="member_id", how="outer")
    ).fillna(0)
    web_window["web_recent_vs_prev_ratio"] = (
        (web_window["web_recent_visits"] - web_window["web_prev_visits"])
        / (web_window["web_prev_visits"] + 1)
    )
    web_window["web_recent_delta"] = web_window["web_recent_visits"] - web_window["web_prev_visits"]
    web_window["web_health_ratio_delta"] = (
        web_window["web_recent_health_ratio"] - web_window["web_prev_health_ratio"]
    )
    df = df.merge(web_window, on="member_id", how="left")

    # --------------------------------------------------------------------
    # Claims recency (30-day window)
    # --------------------------------------------------------------------
    claims_window_days = CLAIMS_RECENT_WINDOW_DAYS
    claims_recent = (
        _count_events_in_window(claims, "diagnosis_date", obs_end, claims_window_days)
        .reset_index(name="claims_recent_count")
    )
    claims_prev = (
        _count_events_in_window(
            claims, "diagnosis_date", obs_end, claims_window_days, offset_days=claims_window_days
        )
        .reset_index(name="claims_prev_count")
    )
    chronic_recent = (
        _count_events_in_window(claims.loc[claims["is_chronic"]], "diagnosis_date", obs_end, claims_window_days)
        .reset_index(name="claims_recent_chronic")
    )
    chronic_prev = (
        _count_events_in_window(
            claims.loc[claims["is_chronic"]],
            "diagnosis_date",
            obs_end,
            claims_window_days,
            offset_days=claims_window_days,
        )
        .reset_index(name="claims_prev_chronic")
    )
    claims_window = (
        claims_recent
        .merge(claims_prev, on="member_id", how="outer")
        .merge(chronic_recent, on="member_id", how="outer")
        .merge(chronic_prev, on="member_id", how="outer")
    ).fillna(0)
    claims_window["claims_recent_vs_prev_ratio"] = (
        (claims_window["claims_recent_count"] - claims_window["claims_prev_count"])
        / (claims_window["claims_prev_count"] + 1)
    )
    claims_window["claims_recent_delta"] = (
        claims_window["claims_recent_count"] - claims_window["claims_prev_count"]
    )
    claims_window["claims_chronic_recent_ratio"] = (
        claims_window["claims_recent_chronic"] / claims_window["claims_recent_count"].clip(lower=1)
    )
    claims_window["claims_chronic_trend"] = (
        (claims_window["claims_recent_chronic"] - claims_window["claims_prev_chronic"])
        / (claims_window["claims_prev_chronic"] + 1)
    )
    df = df.merge(claims_window, on="member_id", how="left")

    # ========================================================================
    # FILL MISSING VALUES
    # ========================================================================
    
    bool_cols = [c for c in df.columns if c.startswith(("has_icd_", "icd_", "is_", "has_"))]
    for c in bool_cols:
        if df[c].dtype == "object" or df[c].dtype == "boolean":
            df[c] = df[c].astype("boolean").fillna(False).astype(bool)
        else:
            df[c] = df[c].fillna(0).astype(int)

    num_cols = [c for c in df.columns if c not in {"member_id", "signup_date", "churn"} and c not in bool_cols]
    for c in num_cols:
        df[c] = df[c].fillna(0.0)

    # ========================================================================
    # FINAL MATRICES
    # ========================================================================
    
    y = df["churn"].astype(int)
    X = df.drop(columns=["churn", "signup_date"])

    # Make booleans numeric for sklearn models
    for c in bool_cols:
        if c in X.columns:
            X[c] = X[c].astype(int)

    return X, y


