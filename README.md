# Helpdesk RAG

A little RAG app that answers IT support questions ("my wifi keeps dropping",
"printer not responding") using only fixes from tickets that were *actually*
resolved before — not whatever an LLM thinks the fix probably is.

## Quickstart

```bash
cp .env.example .env      # paste in a Groq API key
uv run app.py              # opens at http://127.0.0.1:7860
```

I used `uv` instead of pip + venv because `uv run` builds the environment
from `uv.lock` automatically — no `pip install -r requirements.txt`, no
remembering to activate anything. If you don't have `uv`: `pip install uv`,
then the command above just works.

No Groq key? It still runs — you get the matching tickets and their fixes
listed straight from the data, just no generated write-up on top. That was
important to me: the app shouldn't be *useless* without an API key, since the
whole point is that the ticket history is the real source of the answer.

If you want semantic search (matches "device unresponsive" to "printer not
responding" even with no shared words), add `--extra embeddings`. It's
opt-in because it pulls in `torch`, which is ~2 GB:

```bash
uv run --extra embeddings app.py
```

## How it actually answers a question

1. **Retrieve** — search the knowledge base for the closest past problem
   (keyword search, semantic search, or both combined).
2. **Judge the evidence** — before generating anything, check: did we
   actually find something relevant, and do the fixes on record agree with
   each other?
3. **Generate** — only now hand the matched tickets to an LLM (Groq), with
   an explicit instruction not to suggest anything that isn't in the
   evidence.

Step 2 is the part that wasn't in the original notebook, and honestly it's
the part I think actually matters. If nothing matches, the app says so and
skips the model call entirely — there's no point paying for a fluent-sounding
answer built on irrelevant tickets. And if the matched tickets used five
different fixes about equally often, the app tells the model to admit that
instead of just picking the first one and sounding confident about it.

## What I learned from the data (and why the code looks like this)

The dataset (`steve1215rogg/tech-support-conversations-dataset` on Kaggle) is
one row per support conversation. Digging into it before writing any
retrieval code changed almost every design decision below.

**"Resolved" isn't the only status that means resolved.** There are four
values in `Issue_Status`, not the two or three I expected: `Resolved` (505
rows), `Pending` (478), `Resolved after follow-up` (472), and `Escalated`
(441). My first pass filtered on `== "Resolved"` and quietly threw away the
472 tickets that *also* got fixed, just not on the first try. That's almost
half the usable evidence gone because of a naming assumption. Fixed now —
both statuses count (`RESOLVED_STATUSES` in `rag/config.py`).

**The same handful of problems repeat hundreds of times.** Those ~500
resolved rows turned out to be maybe ten distinct problems typed out with
minor wording differences. Retrieving row by row gave the same problem back
three times in the top-4 results with three different fixes and identical
scores, which is not useful to anyone. So `rag/data.py` groups rows by
problem first and keeps every recorded fix for it, with how often each one
was used — `k=4` should mean four different problems, not one problem four
times.

**Categories and fixes are basically assigned at random.** This one
surprised me. I ran Cramér's V (measures how much two categorical columns
actually depend on each other — the Knowledge base tab shows this live) between
the reported problem and the fix that closed it, and got about **0.08**,
which is close enough to "no relationship" to call it that. "Forgot
password" gets closed by "restart your router" about as often as by anything
else. Wi-Fi tickets are filed under "Performance" as often as "Network." That
tells you the labels in this particular export were never meaningfully
attached to the problem — so:

- categories are excluded from search entirely (they'd only add noise, not
  signal),
- the confidence score reports that flatness honestly instead of dressing up
  a coin flip as a recommendation,
- and yes — **on this specific dataset the app will mostly say the evidence
  is inconclusive.** That's not a bug, that's the correct output for data
  this noisy. Point `DATA_CSV` at a cleaner export and the same pipeline
  gives confident, specific answers. The bundled sample just proves the app
  is honest even when the data isn't great.

## Configuration

Everything is optional, set via `.env` or plain environment variables:

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | — | Without it: retrieval-only mode, no generated write-up |
| `GROQ_MODEL` | auto | Leave blank — see note below |
| `DATA_CSV` | — | Path to your own ticket export; blank tries Kaggle, then the sample |
| `RETRIEVER` | `hybrid` | `hybrid`, `embedding`, or `tfidf` |
| `TOP_K` | `4` | How many past problems to retrieve per question |
| `MIN_SCORE` | `0.12` | Below this similarity, nothing counts as a match |
| `MIN_MARGIN` | `0.05` | How much the top match has to beat the runner-up by |

A ticket export needs `Customer_Issue`, `Tech_Response`, `Issue_Category`,
and `Issue_Status` columns. `Resolution_Time` is used if it's there.

I learned the hard way not to pin `GROQ_MODEL`: Groq retires model IDs on a
rolling schedule (`llama-3.3-70b-versatile` got shut down on 16 Aug 2026, mid
build), and a hard-coded name is the first thing that silently breaks.
Leaving it blank means the app just asks the account what it's allowed to
use.

## Retrieval: why "hybrid" and not just one thing

| | Good at | Bad at |
|---|---|---|
| `tfidf` | Exact strings — error codes, model numbers. No download, starts instantly | Doesn't get paraphrasing at all |
| `embedding` | Paraphrasing — "device unresponsive" ≈ "printer not responding" | Needs torch (~2 GB); loses precision on rare identifiers |
| `hybrid` | Both, combined by reciprocal rank fusion | Slowest to start up |

`hybrid` quietly falls back to `tfidf` (with a visible note) if
`sentence-transformers` isn't installed, so the app never just crashes on
startup. Embeddings get cached to `.cache/` keyed by a hash of the indexed
text, so after the first run startup is instant.

I also stem words (just the easy part of Porter stemming — the `-s`, `-ed`,
`-ing` endings) and fold spelling variants before matching, because "my wifi
keeps dropping" needs to reach "Wi-Fi drops after waking from sleep" and
those two strings share almost no tokens otherwise. Each entry is indexed as
its problem text **plus its recorded fixes**, not just the problem title
alone — three words is nowhere near enough surface area to match against a
full sentence someone typed while annoyed.

## Project layout

```
pyproject.toml      dependencies, entry point, lint config
uv.lock             exact resolved versions (commit this)
app.py              the Gradio web UI
cli.py              lets `uv run cli.py` work straight from a checkout
tests.py            checks that run with no API key, no GPU, no network
rag/config.py       settings, read from the environment
rag/data.py         load, clean, consolidate, and quality-check the corpus
rag/retrievers.py   tfidf / embedding / hybrid, one shared interface
rag/generator.py    builds the prompt and calls Groq
rag/pipeline.py     glues it together: retrieve -> judge -> generate
rag/cli.py          the actual `helpdesk` terminal command
rag/assets/         the bundled sample corpus
```

Dependencies are split into groups so you only pay for what you use: the
base install is small, `embeddings` adds torch, `kaggle` adds the dataset
downloader, and `dev` (never installed for regular users) adds `ruff`.

```bash
uv run tests.py                                   # ~21 checks, no key needed
uv run helpdesk "the printer stopped responding"  # ask one question, no UI
uv run ruff check .                               # lint
```

`uv sync` will materialize an actual `.venv` on disk if you want that;
`uv sync --extra all` grabs embeddings + the Kaggle downloader too. To use
the `helpdesk` command from anywhere without cloning first:

```bash
uv tool install .                                 # installs it globally
uvx --from . helpdesk "printer not responding"    # or just run it once
```
