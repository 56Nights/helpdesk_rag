"""The RAG pipeline: retrieve, judge the evidence, then generate."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import generator
from .config import Settings
from .data import KnowledgeBase, QualityReport, load_everything
from .retrievers import build_retriever


@dataclass
class Confidence:
    """How much the retrieved evidence actually supports an answer.

    Two independent things can go wrong, and they call for different replies:
    the question may match nothing in the corpus (low `similarity` or a flat
    ranking), or it may match a problem whose recorded fixes disagree (low
    `agreement`). The first means "we have never seen this"; the second means
    "we have seen it and never learned what fixes it".
    """

    similarity: float
    agreement: float
    level: str
    reason: str
    margin: float = 0.0


@dataclass
class Answer:
    """Everything the UI needs to render one response."""

    text: str
    hits: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    confidence: Confidence | None = None
    model: str = ""
    escalated: bool = False


class HelpdeskRAG:
    """Answers helpdesk questions from a corpus of resolved tickets."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.kb: KnowledgeBase | None = None
        self.quality: QualityReport | None = None
        self.retriever = None
        self.note = ""
        self.model = ""

    # -- setup ------------------------------------------------------------- #
    def load(self) -> HelpdeskRAG:
        """Load the corpus and build the search index."""
        self.kb, self.quality = load_everything(self.settings)
        self.set_retriever(self.settings.retriever)
        if self.settings.has_llm:
            try:
                self.model = generator.pick_model(self.settings)
            except Exception as exc:
                self.note = f"{self.note} Model lookup failed: {exc}".strip()
        return self

    def set_retriever(self, kind: str) -> str:
        """Swap the retrieval strategy at runtime. Returns any fallback note."""
        self.retriever, self.note = build_retriever(
            kind, self.settings, self.kb.texts, self.kb.fingerprint()
        )
        return self.note

    # -- scoring ----------------------------------------------------------- #
    def _confidence(self, hits) -> Confidence:
        if not hits:
            return Confidence(0.0, 0.0, "none", "Nothing was retrieved.")

        similarity = hits[0].score
        margin = similarity - (hits[1].score if len(hits) > 1 else 0.0)
        top_entry = self.kb.entries.iloc[hits[0].index]
        shares = [s["share"] for s in top_entry["solutions"]]
        n_fixes = len(shares)
        top_share = shares[0]

        # Agreement: how far the most-used fix rises above an even split.
        # 0 means every recorded fix was used equally often (the history tells
        # us nothing); 1 means one fix resolved every ticket. Measuring excess
        # over uniform keeps it comparable whether a problem has two recorded
        # fixes or twenty.
        if n_fixes <= 1:
            agreement = 1.0
        else:
            uniform = 1.0 / n_fixes
            agreement = (top_share - uniform) / (1.0 - uniform)
        agreement = float(max(0.0, min(1.0, agreement)))

        if similarity < self.settings.min_score or margin < self.settings.min_margin:
            return Confidence(
                similarity, agreement, "none",
                "Nothing in the ticket history stands out as this problem.",
                margin,
            )
        if agreement < 0.12:
            return Confidence(
                similarity, agreement, "low",
                f"Matching tickets exist, but their {n_fixes} recorded fixes were used "
                f"about equally often (top fix: {top_share:.0%}), so none is better "
                "supported than the others.",
                margin,
            )
        if agreement >= 0.35 and margin >= 0.08:
            return Confidence(
                similarity, agreement, "high",
                f"Clear match, and one fix resolved {top_share:.0%} of these tickets.",
                margin,
            )
        return Confidence(
            similarity, agreement, "medium",
            f"Reasonable match; the most-used fix resolved {top_share:.0%} of these "
            "tickets, so treat the others as alternatives worth trying.",
            margin,
        )

    def _sources(self, hits) -> list[dict]:
        out = []
        for hit in hits:
            row = self.kb.entries.iloc[hit.index]
            out.append(
                {
                    "issue": row["issue"],
                    "score": hit.score,
                    "n_tickets": int(row["n_tickets"]),
                    "solutions": row["solutions"],
                    "categories": row["categories"],
                    "median_minutes": row["median_minutes"],
                }
            )
        return out

    # -- main entry point -------------------------------------------------- #
    def answer(self, question: str, top_k: int | None = None) -> Answer:
        """Answer one helpdesk question."""
        question = (question or "").strip()
        if not question:
            return Answer(text="Describe the problem and I'll search past tickets.")
        if self.kb is None:
            self.load()

        hits = self.retriever.retrieve(question, k=top_k or self.settings.top_k)
        confidence = self._confidence(hits)
        sources = self._sources(hits)

        # Nothing matched: refuse before spending a model call. A fluent answer
        # built on irrelevant evidence is worse than an honest escalation.
        if confidence.level == "none":
            return Answer(
                text=(
                    "**No matching tickets.** Nothing in the resolved-ticket history "
                    "resembles this problem, so there is no recorded fix to pass on.\n\n"
                    "Raise a ticket with the exact error text, when it started, and "
                    "anything that changed on the machine beforehand."
                ),
                hits=hits,
                sources=sources,
                confidence=confidence,
                escalated=True,
            )

        if not self.settings.has_llm:
            return Answer(
                text=self._offline_answer(sources),
                hits=hits,
                sources=sources,
                confidence=confidence,
            )

        prompt = generator.build_prompt(question, hits, self.kb.entries, confidence)
        try:
            text = generator.generate(prompt, self.settings, self.model or "")
        except Exception as exc:
            text = (
                f"**The language model is unreachable** ({type(exc).__name__}). "
                f"The recorded fixes are listed below.\n\n{self._offline_answer(sources)}"
            )
        return Answer(
            text=text, hits=hits, sources=sources, confidence=confidence, model=self.model
        )

    def _offline_answer(self, sources: list[dict]) -> str:
        """Deterministic answer straight from the corpus, no model involved."""
        if not sources:
            return "Nothing to show."
        top = sources[0]
        plural = "s" if top["n_tickets"] != 1 else ""
        lines = [
            f"**Closest recorded problem:** {top['issue']} "
            f"({top['n_tickets']} resolved ticket{plural})",
            "",
            "Fixes that resolved it, most used first:",
            "",
        ]
        for i, sol in enumerate(top["solutions"][:5], start=1):
            n = sol["count"]
            lines.append(
                f"{i}. {sol['text']} — used in {n} ticket{'s' if n != 1 else ''} "
                f"({sol['share']:.0%})"
            )
        return "\n".join(lines)
