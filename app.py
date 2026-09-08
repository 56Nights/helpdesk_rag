"""IT helpdesk assistant: answers a ticket from past resolved tickets.

Run:  python app.py
"""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field

import gradio as gr

from rag.config import Settings
from rag.pipeline import HelpdeskRAG

# --------------------------------------------------------------------------- #
# Boot: load the corpus once, at startup, not per request.
# --------------------------------------------------------------------------- #
SETTINGS = Settings.from_env()
BOT = HelpdeskRAG(SETTINGS).load()

# Confidence drives the colour of the evidence bar, so the reader sees the
# strength of the evidence before reading a word of the answer.
SIGNAL = {"high": "#2F7A4E", "medium": "#B4791F", "low": "#A3454A", "none": "#6B7B79"}
VERDICT = {
    "high": "Well supported",
    "medium": "Partly supported",
    "low": "Fixes disagree",
    "none": "No match",
}

CSS = """
:root { --ink:#10242B; --paper:#EEF1F0; --rule:#CFD8D6; --teal:#0F6E6C; --muted:#5B6B69; }
.gradio-container { background:var(--paper) !important; color:var(--ink) !important; }
#masthead { border-bottom:1px solid var(--rule); padding:0 0 14px; margin-bottom:18px; }
#masthead h1 { font-size:1.45rem; font-weight:600; letter-spacing:-.01em; margin:0 0 4px; }
#masthead p { color:var(--muted); margin:0; font-size:.9rem; }
.statline { font-size:.82rem; color:var(--muted); margin-top:8px; }
.statline b { color:var(--ink); font-weight:600; }
.verdict { display:flex; align-items:baseline; gap:10px; margin:0 0 6px; }
.verdict .label { font-weight:600; font-size:.95rem; }
.verdict .why { color:var(--muted); font-size:.85rem; }
.entry { padding:16px 0; border-top:1px solid var(--rule); }
.entry:first-of-type { border-top:none; }
.entry h4 { margin:0 0 2px; font-size:1rem; font-weight:600; }
.entry .meta { color:var(--muted); font-size:.82rem; margin-bottom:10px; }
.sharebar { display:flex; height:9px; border-radius:2px; overflow:hidden; margin-bottom:9px; }
.sharebar span { display:block; }
.fixes { list-style:none; padding:0; margin:0; }
.fixes li { display:flex; gap:10px; align-items:baseline; padding:3px 0; font-size:.88rem; }
.fixes .swatch { width:9px; height:9px; border-radius:2px; flex:none; transform:translateY(1px); }
.fixes .share { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
.notice { border-left:3px solid #B4791F; padding:10px 14px; margin:10px 0; background:#FAF6EE;
          font-size:.87rem; line-height:1.5; }
.empty { color:var(--muted); font-size:.9rem; padding:20px 0; }
"""


def _shades(base: str, n: int) -> list[str]:
    """Tints of the confidence colour, one per recorded fix, strongest first."""
    r, g, b = (int(base[i : i + 2], 16) for i in (1, 3, 5))
    out = []
    for i in range(max(n, 1)):
        t = 0.0 if n < 2 else 0.62 * i / (n - 1)   # fade towards the paper colour
        faded = (int(c + (0xEE - c) * t) for c in (r, g, b))
        out.append("#{:02x}{:02x}{:02x}".format(*faded))
    return out


def render_evidence(answer) -> str:
    """Draw each matched problem with its distribution of recorded fixes."""
    if not answer.sources:
        return "<div class='empty'>Ask a question to see the tickets behind the answer.</div>"

    conf = answer.confidence
    colour = SIGNAL.get(conf.level if conf else "none", SIGNAL["none"])
    parts = [
        "<div class='verdict'>"
        f"<span class='label' style='color:{colour}'>"
        f"{VERDICT.get(conf.level, '')}</span>"
        f"<span class='why'>{html.escape(conf.reason)}</span></div>"
    ]

    if conf and conf.level == "none":
        # The entries below are the nearest neighbours, not answers. Saying so
        # matters: a list of fixes under a "no match" heading still reads like
        # advice unless it is labelled otherwise.
        parts.append(
            "<div class='empty'>Closest entries in the knowledge base, shown for "
            "reference only — none of them describes this problem.</div>"
        )

    for src in answer.sources:
        fixes = src["solutions"][:6]
        shades = _shades(colour, len(fixes))
        bar = "".join(
            f"<span style='width:{f['share']*100:.1f}%;background:{c}'></span>"
            for f, c in zip(fixes, shades, strict=False)
        )
        items = "".join(
            f"<li><span class='swatch' style='background:{c}'></span>"
            f"<span>{html.escape(f['text'])}</span>"
            f"<span class='share'>{f['share']:.0%} · {f['count']}</span></li>"
            for f, c in zip(fixes, shades, strict=False)
        )
        minutes = src["median_minutes"]
        has_timing = minutes is not None and not math.isnan(float(minutes))
        timing = f" · typically {minutes:.0f} min to close" if has_timing else ""
        top_cat = src["categories"][0][0] if src["categories"] else "—"
        parts.append(
            f"<div class='entry'><h4>{html.escape(src['issue'])}</h4>"
            f"<div class='meta'>match {src['score']:.2f} · {src['n_tickets']} resolved "
            f"tickets · filed under {html.escape(top_cat)}{timing}</div>"
            f"<div class='sharebar'>{bar}</div><ul class='fixes'>{items}</ul></div>"
        )
    return "".join(parts)


def status_line() -> str:
    q = BOT.quality
    llm = (
        BOT.model
        if BOT.settings.has_llm and BOT.model
        else "no key — showing recorded fixes only"
    )
    note = f"<br>{html.escape(BOT.note)}" if BOT.note else ""
    return (
        f"<div class='statline'>Source: <b>{html.escape(BOT.kb.source)}</b> · "
        f"<b>{q.n_resolved}</b> resolved tickets covering "
        f"<b>{len(BOT.kb)}</b> distinct "
        f"problems · answering with <b>{html.escape(llm)}</b>{note}</div>"
    )


def kb_overview() -> str:
    q = BOT.quality
    statuses = " · ".join(f"{k}: {v}" for k, v in q.statuses.items())
    warnings = "".join(f"<div class='notice'>{html.escape(w)}</div>" for w in q.warnings)
    return (
        f"<div class='statline'>{q.n_rows} tickets · {statuses}</div>"
        f"<div class='statline'>Kept as evidence: <b>{q.n_resolved}</b> "
        f"({', '.join(SETTINGS.resolved_statuses)}) · {q.n_distinct_issues} distinct "
        f"problems · {q.n_distinct_solutions} distinct fixes</div>"
        f"<div class='statline'>Fix depends on problem: <b>{q.issue_solution_assoc:.2f}</b> · "
        f"category depends on problem: <b>{q.issue_category_assoc:.2f}</b> "
        f"(0 = no relationship, 1 = fully determined)</div>{warnings}"
    )


def kb_table():
    rows = []
    for _, r in BOT.kb.entries.iterrows():
        best = r["solutions"][0]
        rows.append(
            [
                r["issue"],
                r["n_tickets"],
                len(r["solutions"]),
                f"{best['text']} ({best['share']:.0%})",
            ]
        )
    return rows


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def on_ask(question: str, retriever: str, top_k: int):
    if not (question or "").strip():
        return "Describe the problem first — a sentence is enough.", render_evidence(_EMPTY)
    if retriever != BOT.settings.retriever:
        BOT.settings.retriever = retriever
        BOT.set_retriever(retriever)
    answer = BOT.answer(question, top_k=int(top_k))
    return answer.text, render_evidence(answer)


@dataclass
class _Empty:
    """Stand-in answer used to render the evidence panel before the first ask."""

    sources: list = field(default_factory=list)
    confidence: object = None


_EMPTY = _Empty()


with gr.Blocks(title="Helpdesk assistant") as demo:
    gr.HTML(
        "<div id='masthead'><h1>Helpdesk assistant</h1>"
        "<p>Describe a problem. The answer comes from tickets that were actually "
        "resolved — nothing else.</p></div>"
    )

    with gr.Tabs():
        with gr.Tab("Answer a ticket"):
            question = gr.Textbox(
                label="What's the problem?",
                placeholder="My laptop stopped connecting to Wi-Fi after last night's update.",
                lines=3,
            )
            with gr.Row():
                ask = gr.Button("Search past tickets", variant="primary")
                clear = gr.Button("Clear")

            gr.Examples(
                examples=[
                    "My laptop won't connect to Wi-Fi after the last update",
                    "The printer stopped responding this morning",
                    "I'm locked out of my account after typing the wrong password",
                    "The machine has become very slow over the past week",
                ],
                inputs=question,
                label="Common tickets",
            )

            answer_md = gr.Markdown(
                "Ask a question and the recorded fixes appear here.",
                label="Answer",
            )
            with gr.Accordion("Evidence", open=True):
                evidence = gr.HTML(render_evidence(_EMPTY))

            with gr.Accordion("Search settings", open=False):
                retriever = gr.Radio(
                    ["hybrid", "embedding", "tfidf"],
                    value=BOT.settings.retriever,
                    label="Retrieval",
                    info="hybrid fuses keyword and semantic matching; tfidf is keyword only",
                )
                top_k = gr.Slider(
                    1, 8, value=SETTINGS.top_k, step=1,
                    label="Ticket groups to retrieve",
                )

        with gr.Tab("Knowledge base"):
            gr.HTML(kb_overview())
            gr.Dataframe(
                value=kb_table(),
                headers=["Problem", "Resolved tickets", "Distinct fixes", "Most used fix"],
                wrap=True,
                interactive=False,
            )

    gr.HTML(status_line())

    ask.click(on_ask, [question, retriever, top_k], [answer_md, evidence])
    question.submit(on_ask, [question, retriever, top_k], [answer_md, evidence])
    clear.click(
        lambda: ("", "Ask a question and the recorded fixes appear here.", render_evidence(_EMPTY)),
        outputs=[question, answer_md, evidence],
    )


if __name__ == "__main__":
    theme = gr.themes.Base(
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        primary_hue=gr.themes.colors.teal,
    )
    # Gradio 6 takes app-level theme/css on launch(); Gradio 5 takes them on Blocks().
    if int(gr.__version__.split(".")[0]) >= 6:
        demo.launch(theme=theme, css=CSS)
    else:
        demo.theme, demo.css = theme, CSS
        demo.launch()
