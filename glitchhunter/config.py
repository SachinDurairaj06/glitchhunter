"""Runtime settings, read from GH_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = REPO_ROOT / "games"
MODELS_DIR = REPO_ROOT / "models"

APPROVAL_MODES = ("auto-approve", "auto-reject", "cli", "web")
PLANNERS = ("rule", "bedrock")


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class Settings:
    work_dir: Path
    game: str = "crypt"  # built-in test game (fixture) to hunt in, unless `repo` points elsewhere
    repo: Path | None = None  # any game repository with a glitchhunter.json
    planner: str = "rule"
    bedrock_model_id: str = ""
    aws_region: str = "us-east-1"
    approval_mode: str = "auto-approve"
    approval_timeout_s: float = 900.0
    s3_bucket: str | None = None
    s3_prefix: str = "glitchhunter"
    max_steps: int = 150
    max_seconds: float = 1800.0
    headless: bool = True
    seed: int = 1

    @property
    def fixture_dir(self) -> Path:
        """The game repository the agent works on."""
        if self.repo:
            return self.repo
        return self.work_dir / ("fixture" if self.game == "crypt" else f"fixture-{self.game}")

    @property
    def fixture_manifest(self) -> Path:
        """Ground truth for a built-in game (never inside the repo the agent reads)."""
        return self.work_dir / ("fixture.json" if self.game == "crypt" else f"fixture-{self.game}.json")

    @property
    def checkout_dir(self) -> Path:
        return self.work_dir / "co"

    @property
    def runs_dir(self) -> Path:
        return self.work_dir / "runs"

    @property
    def hunts_dir(self) -> Path:
        return self.work_dir / "hunts"

    @property
    def worktree_dir(self) -> Path:
        return self.work_dir / "wt"


def fs_path(path: Path | str) -> str:
    """A path string for file I/O that also works past Windows' 260-character limit.

    Clones in deep folders (or this repository's vendored three.js addons under one)
    can exceed MAX_PATH when long-path support is off; the extended-length prefix lifts it.
    """
    text = str(path)
    prefix = "\\\\?\\"
    if os.name == "nt" and len(text) >= 240 and not text.startswith(prefix):
        return prefix + os.path.abspath(text).replace("/", "\\")
    return text


def default_work_dir() -> Path:
    """<repo>/.work, unless that path is too long for Windows without long-path support.

    Git object paths under the work dir add ~70 characters, and Windows refuses
    paths over 260 characters unless LongPathsEnabled is set.
    """
    local = REPO_ROOT / ".work"
    if os.name == "nt" and len(str(local)) > 120 and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "glitchhunter" / "work"
    return local


def load_settings(**overrides) -> Settings:
    """Build settings from the environment; keyword arguments win over env vars."""
    settings = Settings(
        work_dir=Path(_env("GH_WORK_DIR", str(default_work_dir()))).resolve(),
        game=_env("GH_GAME", "crypt"),
        repo=Path(os.environ["GH_REPO"]).resolve() if os.environ.get("GH_REPO") else None,
        planner=_env("GH_PLANNER", "rule"),
        bedrock_model_id=_env("GH_BEDROCK_MODEL_ID", ""),
        aws_region=_env("AWS_REGION", _env("AWS_DEFAULT_REGION", "us-east-1")),
        approval_mode=_env("GH_APPROVAL_MODE", "auto-approve"),
        approval_timeout_s=float(_env("GH_APPROVAL_TIMEOUT_S", "900")),
        s3_bucket=os.environ.get("GH_S3_BUCKET") or None,
        s3_prefix=_env("GH_S3_PREFIX", "glitchhunter"),
        max_steps=int(_env("GH_MAX_STEPS", "150")),
        max_seconds=float(_env("GH_MAX_SECONDS", "1800")),
        headless=_env("GH_HEADLESS", "1") != "0",
        seed=int(_env("GH_SEED", "1")),
    )
    for key, value in overrides.items():
        if value is None:
            continue
        if key in ("work_dir", "repo"):
            value = Path(value).resolve()
        setattr(settings, key, value)
    if settings.planner not in PLANNERS:
        raise ValueError(f"GH_PLANNER must be one of {PLANNERS}, got {settings.planner!r}")
    if settings.approval_mode not in APPROVAL_MODES:
        raise ValueError(f"GH_APPROVAL_MODE must be one of {APPROVAL_MODES}, got {settings.approval_mode!r}")
    return settings
