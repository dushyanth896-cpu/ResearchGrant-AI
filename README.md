# ResearchGrant AI

A small RAG (Retrieval-Augmented Generation) app that matches a research
proposal against a catalog of grant opportunities, then uses Gemini to
produce a source-grounded analysis of fit, eligibility, and missing
requirements.

## How it works

1. `build_index.py` reads every grant in `data/*.json`, turns each one
   into plain text, embeds it with `all-MiniLM-L6-v2`
   (sentence-transformers), and saves everything to `grant_index.pkl`.
2. `rag.py` loads that index, embeds the user's proposal, ranks grants
   by cosine similarity, and sends the top matches to Gemini along with
   a strict prompt that keeps the analysis grounded in the retrieved
   documents.
3. `app.py` is the Streamlit front end: proposal box in, formatted
   report out (matched grants, eligibility notes, missing info, and a
   downloadable report).

A prebuilt `grant_index.pkl` for the 9 sample grants is included, so
you can run the app immediately without rebuilding it.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your Google AI Studio key into .env
streamlit run app.py
```

Get a Gemini API key at <https://aistudio.google.com/apikey>.

### Rebuilding the index

Only needed if you add/edit/remove files in `data/`:

```bash
python build_index.py
```

### Evaluating retrieval quality

`evaluate.py` runs a small hand-labeled set of sample proposals through
the retrieval step and reports Hit@K and MRR, so you can sanity-check
changes to the embedding model, chunking, or grant data:

```bash
python evaluate.py
```

## Project structure

```
app.py              Streamlit UI
rag.py               Retrieval + Gemini analysis
build_index.py        Builds grant_index.pkl from data/*.json
evaluate.py            Retrieval evaluation harness
ingest.py               Separate/unused PDF -> FAISS pipeline (see note below)
test_rag.py              Quick manual smoke test for rag.analyze_proposal
data/                     Grant JSON files
grant_index.pkl            Prebuilt embeddings for the grants in data/
.env.example                 Template for your Gemini API key
.streamlit/config.toml         Color theme for the Streamlit UI
```

## What was fixed / changed from the uploaded version

- **The app couldn't read your API key.** `rag.py` never called
  `load_dotenv()`, so `GOOGLE_API_KEY` from `.env` was invisible to it
  unless it happened to be exported in your shell. Fixed.
- **The Gemini model name (`gemini-3.8-flash`) doesn't exist.** Updated
  to `gemini-3.5-flash`, the current stable flash-tier model. Google
  retires model names fairly often — Gemini 2.5 is scheduled to shut
  down 16 Oct 2026 — so if you see a "model not found" error later,
  check <https://ai.google.dev/gemini-api/docs/models>.
- **File paths were relative to the current working directory**, so
  `streamlit run app.py` from a different folder, or a different
  working directory in deployment, could fail to find
  `grant_index.pkl` or `data/`. Paths are now anchored to each
  script's own location.
- **A bug in `app.py`'s `get_grant_data()`** returned a 2-item tuple on
  one error path but was always unpacked as 3 items elsewhere, which
  would crash the app if a malformed result ever reached it. Fixed to
  always return a 3-tuple.
- **Performance**: `grant_index.pkl` was reloaded from disk on every
  single "Analyze" click. It's now cached in memory after first load.
- **`requirements.txt` listed packages the app doesn't use** (`faiss-cpu`,
  `pypdf`, `langchain`, `langchain-community`) mixed in with the ones
  it does, and was missing `numpy`, which both `rag.py` and
  `build_index.py` import directly. Split into "core app" vs.
  "`ingest.py` only" so it's clear what's actually required.
- **`evaluate.py` was an empty file.** Added a working retrieval
  evaluation (Hit@K, MRR) against a small labeled proposal set.
- **`ingest.py` is a different, unused pipeline** (PDF documents →
  FAISS via LangChain) that doesn't connect to `app.py` or `rag.py`,
  which instead read JSON grants via `build_index.py`. Left in place
  with a clarifying note, in case you want PDF ingestion later.
- **UI**: added focus-area/eligibility tag chips and a color-coded
  similarity badge on each grant card, a slider to control how many
  grants get retrieved, a "load example proposal" button, a richer
  downloadable report, and a matching `.streamlit/config.toml` theme.
  Grant names and filenames are now HTML-escaped before being
  rendered, since they're inserted into raw HTML.

## A note on your API key

The `_env` file you uploaded contained a live-looking Google API key.
I didn't put that real value into any file here — `.env.example` has a
placeholder instead. Since that key has now been shared outside your
own machine, it's worth rotating it in Google AI Studio and updating
your local `.env` with the new one, just as you would for any
credential that's been pasted somewhere else.

## Limitations

- The Gemini call, and the eligibility/matching logic that depends on
  it, wasn't runnable in the environment I built this in (no access
  to Google's API from there), so I verified it by careful reading
  rather than a live end-to-end run. Test with your real key before
  relying on it.
- Retrieval is small-scale semantic search over 9 grants — it's not
  meant to prove eligibility, only to surface plausibly relevant
  opportunities for Gemini (and you) to evaluate further.
