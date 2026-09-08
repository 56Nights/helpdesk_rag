"""Runtime configuration, read from environment variables with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# The repo root when running from a checkout; used for .env discovery and the
# embedding cache, neither of which belongs inside an installed package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The dataset labels resolution with four statuses, two of which mean "fixed".
# The notebook kept only 'Resolved' (505 rows) and silently dropped
# 'Resolved after follow-up' (472 rows) -- nearly half the usable evidence.
RESOLVED_STATUSES = ("Resolved", "Resolved after follow-up")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    """All tunable knobs in one place.

    Attributes
    ----------
    groq_api_key : str
        Groq key. Empty means the app runs in retrieval-only mode.
    groq_model : str
        Model ID. Empty means auto-detect from the account's model list.
    data_csv : str
        Explicit path to a ticket CSV. Empty means try Kaggle, then the
        bundled sample.
    retriever : str
        'tfidf', 'embedding' or 'hybrid'.
    top_k : int
        Knowledge base entries retrieved per question.
    min_score : float
        Absolute similarity floor. Below it, nothing in the corpus is close
        enough to be worth showing.
    min_margin : float
        How far the best match must beat the runner-up. This is the stronger
        of the two signals: an off-topic question scores similarly against
        every entry, so a flat ranking means "not in the corpus" even when
        the top score looks respectable.
    """

    groq_api_key: str = ""
    groq_model: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"

    data_csv: str = ""
    kaggle_dataset: str = "steve1215rogg/tech-support-conversations-dataset"
    resolved_statuses: tuple = RESOLVED_STATUSES

    retriever: str = "hybrid"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 4
    min_score: float = 0.12
    min_margin: float = 0.05

    max_tokens: int = 700
    temperature: float = 0.2

    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / ".cache")

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables (and .env if present)."""
        _load_dotenv()
        return cls(
            groq_api_key=os.environ.get("GROQ_API_KEY", "").strip(),
            groq_model=os.environ.get("GROQ_MODEL", "").strip(),
            data_csv=os.environ.get("DATA_CSV", "").strip(),
            retriever=os.environ.get("RETRIEVER", "hybrid").strip().lower(),
            top_k=_env_int("TOP_K", 4),
            min_score=_env_float("MIN_SCORE", 0.12),
            min_margin=_env_float("MIN_MARGIN", 0.05),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.groq_api_key)


def _load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader so the app has no python-dotenv dependency."""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
