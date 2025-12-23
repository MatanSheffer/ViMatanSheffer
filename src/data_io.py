from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RawTables:
    web_visits: pd.DataFrame
    app_usage: pd.DataFrame
    claims: pd.DataFrame
    labels: pd.DataFrame


def _pick_file(root: Path, base_name: str) -> Path:
    """Pick `base_name` if present, otherwise fall back to `test_{base_name}`."""
    direct = root / base_name
    if direct.exists():
        return direct
    prefixed = root / f"test_{base_name}"
    if prefixed.exists():
        return prefixed
    raise FileNotFoundError(f"Could not find {base_name} (or test_{base_name}) under {root}")


def load_tables(root: Path) -> RawTables:
    """Load a split (train or test) from a directory containing the four CSVs."""
    web_visits = pd.read_csv(
        _pick_file(root, "web_visits.csv"),
        dtype={"member_id": "int64", "url": "string", "title": "string", "description": "string"},
        parse_dates=["timestamp"],
    )

    app_usage = pd.read_csv(
        _pick_file(root, "app_usage.csv"),
        dtype={"member_id": "int64", "event_type": "string"},
        parse_dates=["timestamp"],
    )

    claims = pd.read_csv(
        _pick_file(root, "claims.csv"),
        dtype={"member_id": "int64", "icd_code": "string"},
        parse_dates=["diagnosis_date"],
    )

    labels = pd.read_csv(
        _pick_file(root, "churn_labels.csv"),
        dtype={"member_id": "int64", "churn": "int64", "outreach": "int64"},
        parse_dates=["signup_date"],
    )

    # A tiny bit of normalization
    for col in ["churn", "outreach"]:
        labels[col] = labels[col].astype(int)

    return RawTables(web_visits=web_visits, app_usage=app_usage, claims=claims, labels=labels)


