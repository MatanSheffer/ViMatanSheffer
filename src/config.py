from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root by walking up until we see a `data/` directory."""
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "data").exists():
            return p
    raise FileNotFoundError("Could not find repo root (expected a `data/` directory).")


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    artifacts_dir: Path


def get_paths(start: Path | None = None) -> Paths:
    repo_root = find_repo_root(start)
    data_dir = repo_root / "data"
    return Paths(
        repo_root=repo_root,
        data_dir=data_dir,
        train_dir=data_dir / "train",
        test_dir=data_dir / "test",
        artifacts_dir=repo_root / "artifacts",
    )


