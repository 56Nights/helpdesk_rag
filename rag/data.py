"""Loading, cleaning and consolidating the ticket knowledge base.

The raw export is one row per conversation, which is the wrong shape for
retrieval: the same problem appears hundreds of times with different fixes
attached. This module collapses it to one entry per distinct problem, keeping
every fix that was recorded for it together with how often it was used.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings

REQUIRED_COLUMNS = [
    "Customer_Issue",
    "Tech_Response",
    "Issue_Category",
    "Issue_Status",
]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(settings: Settings) -> tuple[pd.DataFrame, str]:
    """Load the ticket export.

    Tries, in order: an explicit CSV path, a Kaggle download, the bundled
    sample. Returns the dataframe and a human-readable description of where
    it came from, which the UI displays so nobody mistakes sample data for
    the real corpus.
    """
    if settings.data_csv:
        path = Path(settings.data_csv).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"DATA_CSV points at a missing file: {path}")
        return pd.read_csv(path), f"CSV file ({path.name})"

    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(settings.kaggle_dataset))
        csvs = sorted(folder.glob("*.csv"))
        if csvs:
            return pd.read_csv(csvs[0]), f"Kaggle: {settings.kaggle_dataset}"
    except Exception:
        pass  # no kagglehub, no credentials, or no network -- fall through

    # Ships inside the package, so it resolves the same whether the app is run
    # from the repo, from an installed console script, or via `uvx`.
    sample = Path(__file__).parent / "assets" / "sample_tickets.csv"
    return pd.read_csv(sample), "Bundled sample (not real data)"


def _minutes(value) -> float:
    """Parse 'Resolution_Time' strings such as '92 minutes' into a number."""
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise whitespace, parse durations, drop unusable rows."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Ticket export is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        out[col] = out[col].astype(str).str.strip()

    out["minutes"] = (
        out["Resolution_Time"].map(_minutes)
        if "Resolution_Time" in out.columns
        else np.nan
    )

    blank = out["Customer_Issue"].eq("") | out["Tech_Response"].eq("")
    return out[~blank].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
@dataclass
class KnowledgeBase:
    """One row per distinct problem, with its recorded fixes.

    Attributes
    ----------
    entries : pd.DataFrame
        Columns: issue, n_tickets, solutions (list of dicts with text/count/
        share), categories (list of (name, count)), median_minutes.
    resolved : pd.DataFrame
        The resolved tickets the entries were built from.
    source : str
        Where the data came from.
    """

    entries: pd.DataFrame
    resolved: pd.DataFrame
    source: str

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def texts(self) -> list[str]:
        """The strings the retriever indexes.

        Each entry is indexed as its problem statement plus the fixes recorded
        against it. Titles alone are two or three words -- far too little
        surface to match against a sentence someone types in frustration --
        while the fixes carry the vocabulary of the domain ('unlock the
        account', 'credential manager') that makes a question findable.
        Categories are deliberately excluded: in the source export they are
        assigned independently of the problem, so they add noise, not signal.
        """
        return [
            entry["issue"] + " " + " ".join(s["text"] for s in entry["solutions"])
            for _, entry in self.entries.iterrows()
        ]

    def fingerprint(self) -> str:
        """Stable hash of the indexed texts, used to key the embedding cache."""
        joined = "\u241f".join(self.texts).encode("utf-8")
        return hashlib.sha1(joined).hexdigest()[:16]


def build_knowledge_base(df: pd.DataFrame, settings: Settings) -> KnowledgeBase:
    """Filter to resolved tickets and consolidate them by problem."""
    cleaned = clean(df)

    resolved = cleaned[cleaned["Issue_Status"].isin(settings.resolved_statuses)]
    resolved = resolved.reset_index(drop=True)
    if resolved.empty:
        raise ValueError(
            "No resolved tickets found. Statuses present: "
            f"{sorted(cleaned['Issue_Status'].unique())}"
        )

    rows = []
    for issue, group in resolved.groupby("Customer_Issue", sort=False):
        fixes = group["Tech_Response"].value_counts()
        cats = group["Issue_Category"].value_counts()
        rows.append(
            {
                "issue": issue,
                "n_tickets": int(len(group)),
                "solutions": [
                    {
                        "text": text,
                        "count": int(count),
                        "share": float(count / len(group)),
                    }
                    for text, count in fixes.items()
                ],
                "categories": [(name, int(n)) for name, n in cats.items()],
                "median_minutes": float(group["minutes"].median()),
            }
        )

    entries = (
        pd.DataFrame(rows)
        .sort_values("n_tickets", ascending=False)
        .reset_index(drop=True)
    )
    return KnowledgeBase(entries=entries, resolved=resolved, source="")


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association between two categorical variables, in [0, 1].

    0 means the two columns are independent -- for a ticket export, that
    the labels carry no information about the problem.
    """
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    n = table.sum()
    if n == 0 or min(table.shape) < 2:
        return 0.0
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / np.clip(expected, 1e-12, None)).sum())
    return float(np.sqrt((chi2 / n) / (min(table.shape) - 1)))


@dataclass
class QualityReport:
    """What the corpus can and cannot support, in plain numbers."""

    n_rows: int
    n_resolved: int
    statuses: dict
    n_distinct_issues: int
    n_distinct_solutions: int
    issue_solution_assoc: float
    issue_category_assoc: float
    warnings: list

    @property
    def solutions_are_random(self) -> bool:
        return self.issue_solution_assoc < 0.10

    @property
    def categories_are_random(self) -> bool:
        return self.issue_category_assoc < 0.10


def assess(df: pd.DataFrame, kb: KnowledgeBase, settings: Settings) -> QualityReport:
    """Measure whether the corpus can actually support grounded answers."""
    cleaned = clean(df)
    resolved = kb.resolved

    issue_solution = _cramers_v(resolved["Customer_Issue"], resolved["Tech_Response"])
    issue_category = _cramers_v(resolved["Customer_Issue"], resolved["Issue_Category"])

    warnings = []
    if issue_solution < 0.10:
        warnings.append(
            "Recorded fixes are statistically independent of the reported problem. "
            "Every problem lists nearly every fix in equal proportion, so no fix can "
            "be recommended over another. Answers will be low-confidence by design."
        )
    if issue_category < 0.10:
        warnings.append(
            "Categories are independent of the problem text (Wi-Fi tickets are filed "
            "under Performance as often as Network). Category is kept as metadata but "
            "excluded from the search index, where it would only add noise."
        )
    if len(kb) < 25:
        warnings.append(
            f"Only {len(kb)} distinct problems in the knowledge base. Retrieval is "
            "close to exact matching; expect the same entries for most questions."
        )
    return QualityReport(
        n_rows=int(len(cleaned)),
        n_resolved=int(len(resolved)),
        statuses=cleaned["Issue_Status"].value_counts().to_dict(),
        n_distinct_issues=int(resolved["Customer_Issue"].nunique()),
        n_distinct_solutions=int(resolved["Tech_Response"].nunique()),
        issue_solution_assoc=issue_solution,
        issue_category_assoc=issue_category,
        warnings=warnings,
    )


def load_everything(settings: Settings) -> tuple[KnowledgeBase, QualityReport]:
    """Load, consolidate and assess in one call."""
    raw, source = load_raw(settings)
    kb = build_knowledge_base(raw, settings)
    kb.source = source
    return kb, assess(raw, kb, settings)
