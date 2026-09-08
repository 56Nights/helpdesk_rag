"""Checks that run without a Groq key, a GPU or a network connection.

    python tests.py
"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
import types
from pathlib import Path

from rag import generator
from rag.config import Settings
from rag.data import clean, load_raw
from rag.pipeline import HelpdeskRAG
from rag.retrievers import TfidfRetriever, normalise

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"{mark} {name}" + (f" — {detail}" if detail else ""))


def random_corpus(path: Path) -> None:
    """A corpus shaped like the Kaggle set: fixes paired at random."""
    random.seed(3)
    issues = ["Cannot connect to Wi-Fi", "Printer not responding", "Forgot password",
              "Blue screen error", "Slow system performance", "Unable to access email"]
    fixes = ["Clear cache and remove unnecessary programs.", "Reinstall the printer drivers.",
             "Run a system diagnostic tool.", "Restart your router.",
             "Reset your password using the link provided.", "Verify your email settings."]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Conversation_ID", "Customer_Issue", "Tech_Response",
                    "Resolution_Time", "Issue_Category", "Issue_Status"])
        for i in range(1200):
            w.writerow([f"CONV-{i:04d}", random.choice(issues), random.choice(fixes),
                        f"{random.randint(5, 120)} minutes",
                        random.choice(["Hardware", "Network", "Software", "Account"]),
                        random.choice(["Resolved", "Pending", "Resolved after follow-up",
                                       "Escalated"])])


def main() -> int:
    # -- text handling ----------------------------------------------------- #
    check("normalise folds Wi-Fi spellings",
          normalise("Wi-Fi") == normalise("wifi") == "wifi")

    # -- data layer -------------------------------------------------------- #
    settings = Settings(retriever="tfidf")
    bot = HelpdeskRAG(settings).load()
    kb, quality = bot.kb, bot.quality
    check("knowledge base is deduplicated",
          len(kb) == kb.entries["issue"].nunique(), f"{len(kb)} problems")
    check("both resolved statuses are kept",
          quality.n_resolved > quality.statuses.get("Resolved", 0),
          f"{quality.n_resolved} tickets vs {quality.statuses.get('Resolved')} 'Resolved' only")
    check("shares sum to 1 per problem",
          all(abs(sum(s["share"] for s in row) - 1) < 1e-9
              for row in kb.entries["solutions"]))
    check("resolution times parsed",
          kb.entries["median_minutes"].notna().all())

    raw, _ = load_raw(settings)
    try:
        clean(raw.drop(columns=["Tech_Response"]))
        check("missing column raises a clear error", False)
    except KeyError as exc:
        check("missing column raises a clear error", "Tech_Response" in str(exc))

    # -- retrieval --------------------------------------------------------- #
    expected = {
        "my laptop won't connect to Wi-Fi after the last update": "Cannot connect to Wi-Fi",
        "wifi keeps dropping when the laptop wakes up": "Wi-Fi drops after waking from sleep",
        "the printer stopped responding this morning": "Printer not responding",
        "locked out after typing my password wrong too many times":
            "Account locked after failed sign-ins",
        "vpn keeps disconnecting": "VPN disconnects every few minutes",
        "my pc wont charge": "Laptop will not charge",
    }
    hits = 0
    for question, want in expected.items():
        got = bot.answer(question).sources[0]["issue"]
        hits += got == want
        if got != want:
            print(f"     miss: {question!r} -> {got!r}")
    check("retrieval finds the right problem", hits == len(expected),
          f"{hits}/{len(expected)}")

    retriever = TfidfRetriever().fit(kb.texts)
    check("k larger than the corpus is clamped",
          len(retriever.retrieve("wifi", k=999)) == len(kb))
    check("scores are sorted descending",
          [h.score for h in retriever.retrieve("printer", k=5)]
          == sorted([h.score for h in retriever.retrieve("printer", k=5)], reverse=True))

    # -- confidence -------------------------------------------------------- #
    off_topic = bot.answer("the coffee machine in the kitchen is broken")
    check("unrelated question escalates instead of guessing",
          off_topic.escalated and off_topic.confidence.level == "none")
    good = bot.answer("my pc wont charge")
    check("clear evidence scores high", good.confidence.level == "high",
          f"agreement={good.confidence.agreement:.2f}")
    check("empty question is handled", bot.answer("   ").sources == [])

    # -- random corpus: the failure mode the real dataset has --------------- #
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "random.csv"
        random_corpus(path)
        noisy = HelpdeskRAG(Settings(data_csv=str(path), retriever="tfidf")).load()
        check("random fix pairing is detected",
              noisy.quality.solutions_are_random,
              f"association={noisy.quality.issue_solution_assoc:.3f}")
        answer = noisy.answer("cannot connect to wifi")
        check("random corpus yields low confidence, not false certainty",
              answer.confidence.level == "low",
              f"agreement={answer.confidence.agreement:.2f}")

    # -- generation (stubbed: no key, no network) --------------------------- #
    captured = {}

    class FakeCompletions:
        def create(self, **kw):
            captured.update(kw)
            msg = types.SimpleNamespace(content="1. Do the thing.\nTickets used: Wi-Fi.")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeClient:
        def __init__(self, **kw):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[types.SimpleNamespace(id="openai/gpt-oss-120b")]))

    generator._client = lambda settings: FakeClient()

    llm_settings = Settings(groq_api_key="gsk_test", retriever="tfidf")
    llm_bot = HelpdeskRAG(llm_settings).load()
    check("model is auto-detected", llm_bot.model == "openai/gpt-oss-120b", llm_bot.model)

    answered = llm_bot.answer("printer not responding")
    check("generated answer is returned", answered.text.startswith("1."))
    check("evidence reaches the prompt",
          "Recorded fixes" in captured["messages"][1]["content"])
    check("fix frequencies reach the prompt",
          "%" in captured["messages"][1]["content"])
    check("reasoning effort set for gpt-oss",
          captured["extra_body"] == {"reasoning_effort": "low"})

    llm_bot.kb.entries.at[0, "solutions"] = [
        {"text": "a", "count": 5, "share": 0.5}, {"text": "b", "count": 5, "share": 0.5}]
    low = llm_bot.answer("cannot connect to wifi")
    check("low evidence adds a caveat to the prompt",
          low.confidence.level != "low"
          or "LOW" in captured["messages"][1]["content"])

    def boom(*a, **k):
        raise RuntimeError("connection reset")

    generator.generate = boom
    survived = llm_bot.answer("printer not responding")
    check("API failure degrades to recorded fixes",
          "unreachable" in survived.text and survived.sources != [])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
