"""Prompt construction and the Groq call."""
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = """You are the first-line assistant on an internal IT helpdesk.
An employee has described a problem. Past tickets that were marked resolved are
supplied as evidence.

Rules:
1. Recommend only fixes that appear in the evidence. Never invent a step that is
   not there, however plausible it sounds.
2. Each recorded fix comes with how often it resolved that problem. Lead with the
   most frequent one and say how much support it has.
3. If the evidence does not cover the problem, or if the recorded fixes are
   spread evenly with no clear winner, say so directly and tell the employee to
   raise a ticket. Do not pad a weak answer with generic troubleshooting advice.
4. Write numbered steps the employee can carry out alone. Be brief.
5. Close with one line naming the tickets you relied on.
"""

MODEL_PREFERENCES = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
]


def _client(settings):
    from openai import OpenAI

    return OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)


def pick_model(settings) -> str:
    """Return a model this account can serve.

    Groq retires model IDs on a rolling schedule (llama-3.3-70b-versatile was
    shut down on 2026-08-16), so the served list is the source of truth rather
    than a hard-coded name.
    """
    if settings.groq_model:
        return settings.groq_model
    try:
        available = {m.id for m in _client(settings).models.list().data}
    except Exception:
        return MODEL_PREFERENCES[0]
    for name in MODEL_PREFERENCES:
        if name in available:
            return name
    chat_models = sorted(m for m in available if "whisper" not in m and "tts" not in m)
    if chat_models:
        return chat_models[0]
    raise RuntimeError("No usable chat model on this Groq account.")


def format_evidence(hits, entries: pd.DataFrame, max_solutions: int = 4) -> str:
    """Render retrieved knowledge base entries as prompt evidence."""
    blocks = []
    for rank, hit in enumerate(hits, start=1):
        row = entries.iloc[hit.index]
        lines = [
            f"[Ticket group {rank}] Reported problem: {row['issue']}",
            f"Resolved tickets: {row['n_tickets']}"
            + (
                f" | typical resolution time: {row['median_minutes']:.0f} min"
                if pd.notna(row["median_minutes"])
                else ""
            ),
            "Recorded fixes:",
        ]
        for sol in row["solutions"][:max_solutions]:
            lines.append(
                f"  - {sol['text']} (used in {sol['count']} of "
                f"{row['n_tickets']} tickets, {sol['share']:.0%})"
            )
        remaining = len(row["solutions"]) - max_solutions
        if remaining > 0:
            lines.append(f"  - ... and {remaining} further fixes with fewer uses")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_prompt(question: str, hits, entries: pd.DataFrame, confidence=None) -> str:
    """Assemble the user turn: evidence first, question last."""
    evidence = format_evidence(hits, entries) or "(nothing relevant was retrieved)"
    rule = "-" * 60

    caveat = ""
    if confidence is not None and confidence.level == "low":
        # Told explicitly, rather than left for the model to infer from the
        # percentages, because models reliably over-commit to weak evidence.
        caveat = (
            "\nEvidence quality: LOW. The fixes above are spread evenly across this "
            "problem, so no single one is better supported. Say this plainly and "
            "recommend raising a ticket instead of picking one at random.\n"
        )

    return (
        f"EVIDENCE FROM RESOLVED TICKETS\n{rule}\n{evidence}\n{rule}\n{caveat}\n"
        f"EMPLOYEE'S PROBLEM:\n{question}\n\n"
        "Answer using only the evidence above."
    )


def generate(prompt: str, settings, model: str) -> str:
    """Call Groq and return the answer text."""
    extra_body = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    completion = _client(settings).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        extra_body=extra_body,
    )
    return (completion.choices[0].message.content or "").strip()
