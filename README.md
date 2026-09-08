# Helpdesk assistant

Answers an IT support question from tickets that were actually resolved, and says
so plainly when the ticket history doesn't support an answer.

Built from the RAG notebook (`s08_tp`): same pipeline — retrieve, augment,
generate — restructured into a package that runs as a web app, a CLI, or an
importable library.

```bash
cp .env.example .env          # add your Groq key
uv run app.py                 # http://127.0.0.1:7860
```

`uv run` creates the environment from `uv.lock` on first use, so there is no
install step and no activate step. To include semantic retrieval (downloads
torch, ~2 GB):

```bash
uv run --extra embeddings app.py
```

Without a Groq key it still runs: it retrieves the matching tickets and lists the
fixes that resolved them, just with no generated prose.

## What it does with a question

1. **Retrieve** — finds the closest problems in the knowledge base (keyword,
   semantic, or both fused).
2. **Judge the evidence** — scores whether the match is real and whether the
   recorded fixes agree with each other.
3. **Generate** — sends the fixes and their frequencies to Groq, with an explicit
   instruction to recommend nothing that isn't in the evidence.

Step 2 is the part the notebook didn't have, and it's what makes the app usable:
if nothing matches, it escalates instead of calling the model, and if the matched
tickets disagree about the fix, it tells the model to say so rather than pick one.

## The data, and why the app is shaped this way

The Kaggle export (`steve1215rogg/tech-support-conversations-dataset`) is one row
per conversation. Three properties of it drove the design:

**Half the resolved tickets were being thrown away.** `Issue_Status` has four
values, not three: `Resolved` (505), `Pending` (478), `Resolved after follow-up`
(472) and `Escalated` (441). A filter on `== 'Resolved'` silently discards the
472 tickets that were also fixed, just not on the first attempt. The app keeps
both (`RESOLVED_STATUSES` in `rag/config.py`), roughly doubling the evidence.

**The same problem repeats hundreds of times.** 505 rows contain about ten
distinct problems. Retrieving row-by-row returns the same problem three times
with three different fixes and identical scores. The app consolidates to one
entry per distinct problem, carrying every recorded fix with its frequency, so
`k=4` means four genuinely different problems.

**Fixes and categories are assigned at random.** Measure the association between
problem and fix (Cramér's V, in the Knowledge base tab): on this corpus it is
≈0.08, i.e. none. "Forgot password" is closed by "Restart your router" as often
as by anything else, and Wi-Fi tickets are filed under Performance as often as
Network. So: categories are excluded from the search index, and the confidence
score reports the flat distribution honestly instead of dressing a random pick as
a recommendation. **On this dataset the app will mostly say the evidence is
inconclusive. That is the correct output.** Point `DATA_CSV` at a real export and
the same code produces real answers — the bundled sample demonstrates that shape.

## Configuration

All optional; set in `.env` or the environment.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | — | Without it, retrieval-only mode |
| `GROQ_MODEL` | auto | Blank auto-detects a model the account serves |
| `DATA_CSV` | — | Path to a ticket export; blank tries Kaggle, then the sample |
| `RETRIEVER` | `hybrid` | `hybrid`, `embedding` or `tfidf` |
| `TOP_K` | `4` | Problems retrieved per question |
| `MIN_SCORE` | `0.12` | Similarity floor below which nothing is shown |
| `MIN_MARGIN` | `0.05` | How far the best match must beat the runner-up |

A ticket export needs `Customer_Issue`, `Tech_Response`, `Issue_Category` and
`Issue_Status`; `Resolution_Time` is used if present.

Don't pin `GROQ_MODEL` unless you have to. Groq retires model IDs on a schedule —
`llama-3.3-70b-versatile` was shut down on 16 August 2026 — and a pinned name is
the first thing that breaks. Blank means the app asks the API what it can serve.

## Retrieval

| | Strength | Weakness |
|---|---|---|
| `tfidf` | Exact strings: error codes, model numbers. No downloads, instant startup | Blind to paraphrase |
| `embedding` | Paraphrase: "device unresponsive" ≈ "printer not responding" | Needs torch (~2 GB); vague on rare identifiers |
| `hybrid` | Both, fused by reciprocal rank | Slowest to start |

`hybrid` falls back to `tfidf` with a visible note if sentence-transformers isn't
installed, so the app always starts. Embeddings are cached in `.cache/` keyed by a
hash of the indexed text, so startup is instant after the first run.

Lexical matching stems words (Porter step 1) and folds spelling variants, because
questions and ticket titles never agree on inflection: *"my wifi keeps dropping"*
has to reach *"Wi-Fi drops after waking from sleep"*. Each entry is indexed as its
problem statement **plus its recorded fixes** — a three-word title is too little
surface to match a sentence someone typed in frustration.

## Layout

```
pyproject.toml      dependencies, entry points, lint config
uv.lock             exact resolved versions — commit this
app.py              Gradio UI
cli.py              wrapper so `uv run cli.py` works from a checkout
tests.py            checks that run with no key, no GPU, no network
rag/config.py       settings from environment
rag/data.py         load, clean, consolidate, assess quality
rag/retrievers.py   tfidf / embedding / hybrid, one interface
rag/generator.py    prompt construction and the Groq call
rag/pipeline.py     retrieve -> judge -> generate
rag/cli.py          terminal interface (the `helpdesk` command)
rag/assets/         bundled sample corpus
```

Dependencies are grouped so a deployment only pays for what it uses: the base
install is small, `embeddings` adds torch, `kaggle` adds the dataset downloader,
and `dev` (a dependency group, never installed by consumers) adds ruff.

```bash
uv run tests.py                                   # 21 checks, no key needed
uv run helpdesk "the printer stopped responding"  # one-off question
uv run ruff check .                               # lint (dev group)
```

`uv sync` materialises `.venv` if you want it on disk; `uv sync --extra all`
adds embeddings and the Kaggle downloader. To answer questions from anywhere on
the machine without cloning first:

```bash
uv tool install .            # provides the `helpdesk` command globally
uvx --from . helpdesk "printer not responding"   # or run it once, no install
```

## Deploying

Any host that runs a Python process works. For Hugging Face Spaces, push the
folder with `GROQ_API_KEY` set as a Space secret; `app.py` is the entry point
Spaces expects. Set `RETRIEVER=tfidf` on a small instance to skip the torch
download.

Spaces installs from `requirements.txt` rather than the lock file, so generate
one from the lock at deploy time instead of maintaining it by hand:

```bash
uv export --no-hashes --no-dev --no-emit-project > requirements.txt
```
