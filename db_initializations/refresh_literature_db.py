#!/usr/bin/env python3
"""Rebuild the LiteratureDB ChromaDB vector store from OpenAlex.

This replaces ``db_initializations/make_chromadb_visualization_v2.ipynb`` with a
re-runnable script.

What it does
------------
1. Searches OpenAlex for genitourinary-cancer clinical literature
   (see ``DEFAULT_SEARCH_PHRASE`` / ``--search-phrase``).
2. Fetches the VariantScape paper set by OpenAlex id and prepends it, so that
   literature answers can draw on the papers behind Variantscape's associations.
   The set is ``variantscape/variantscape_papers.csv`` (written by the
   Variantscape pipeline's build and deployed with its artifacts: the papers
   behind verified associations); older deployments without it fall back to
   ``cleaned_df_v4_corrected.csv`` (the notebooks' paper set). Skip with
   ``--no-variantscape``.
3. Embeds ``title + " " + abstract`` for every paper with the same
   sentence-transformer model the app queries with and writes a persistent
   Chroma collection (cosine space), plus a ``build_manifest.json`` recording
   what was built and when.
4. Builds ``filter_metadata.csv`` in the same directory -- the one-hot matrix
   behind the left-panel publication filters -- by classifying every paper's
   title+abstract with the answer-path LLM against the columns defined in
   ``literature.FILTER_GROUPS``. Aligned to the corpus by construction.
   Disable with ``--no-filter-metadata``. Papers already classified in the
   previous build (``--reuse-from``, default: the current live one) are copied
   over by OpenAlex id and skip the LLM -- but only while the filter columns
   are unchanged, so editing ``FILTER_GROUPS`` forces a full re-classify.

Typical use
-----------
    python db_initializations/refresh_literature_db.py --from-date 2015-01-01

Set ``OPENALEX_API_KEY`` (and ``OPENALEX_EMAIL``) in evidence-database/.env or the
environment: without an API key OpenAlex's rate limits are much lower and a full
build runs into "429 Too Many Requests". Fetched papers are cached under
``<out-dir>/openalex/``: if a build is interrupted, run the same command with the same
``--out-dir`` and it continues where it stopped.

Rebuilding without repeating the whole search (e.g. after a new Variantscape build)::

    python db_initializations/refresh_literature_db.py --dedupe \
        --reuse-search-from chroma_data_YYYYMMDD --top-up

takes the search papers of that earlier build, adds papers published since then
(``--top-up``; 30 days overlap) and fetches only the VariantScape papers anew.

then point the app at the new store (the exact line is printed at the end)::

    # evidence-database/.env
    EVIDENCE_DB_CHROMA_PATH=chroma_data_YYYYMMDD

``EVIDENCE_DB_CHROMA_PATH`` also repoints ``LITERATURE_METADATA_CSV`` at the
matching ``filter_metadata.csv``, so one variable moves both.

To (re)run only step 4 against an already-built collection -- e.g. after fixing
an LLM-connectivity problem::

    python db_initializations/refresh_literature_db.py \\
        --filter-metadata-only --out-dir chroma_data_YYYYMMDD

Step 4 uses the same OPENAI_* settings as the app's answer path. If it reports
"Connection error." for every row while the app still works, the script is
almost certainly not seeing the same environment as the running service
(``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` /
virtualenv). The endpoint and key it resolved are printed at the start of
step 4 -- compare them with the app's.

Caveats
-------
* Always build into a NEW ``--out-dir`` (the default is dated). Re-embedding into
  the live directory risks duplicate / colliding document ids -- the script
  refuses to add to a non-empty collection unless ``--recreate`` is given.
* Step 4 is ~one LLM call per *new* paper (unchanged papers are reused from the
  previous build). It is resumable (Ctrl-C is safe; rerun to continue from
  ``filter_metadata.csv.partial.jsonl``) but the first run after a filter change
  reclassifies the whole corpus and costs real API time/money. Use
  ``--filter-metadata-limit`` for a trial run, ``--no-reuse`` to force a rebuild.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# config loads evidence-database/.env without overriding variables already set in the
# shell: remember whether the LLM key came from the shell, to say so (a stale exported
# OPENAI_API_KEY silently wins over .env)
_KEY_FROM_SHELL = "OPENAI_API_KEY" in os.environ
import config  # noqa: E402  (needs REPO_ROOT on the path first)

OPENALEX_WORKS = "https://api.openalex.org/works"

# Genitourinary cancers, matching the cancer-type options in
# templates/literature.html, AND-ed with clinical-evidence study types.
DEFAULT_SEARCH_PHRASE = (
    '('
    '"prostate cancer" OR "prostatic carcinoma" OR "CRPC" OR "mCRPC" '
    'OR "bladder cancer" OR "urothelial carcinoma" OR "transitional cell carcinoma" '
    'OR "kidney cancer" OR "renal cell carcinoma" OR "RCC" '
    'OR "testicular cancer" OR "germ cell tumour" OR "germ cell tumor" OR "seminoma" '
    'OR "penile cancer" OR "penile carcinoma" '
    'OR "urethral cancer" OR "urethral carcinoma" '
    'OR "upper tract urothelial carcinoma" OR "renal pelvis carcinoma" OR "ureteral carcinoma" '
    'OR "genitourinary cancer" OR "urologic oncology"'
    ') AND ('
    '"clinical trial" OR "randomized controlled trial" OR "randomised controlled trial" OR "RCT" '
    'OR "systematic review" OR "meta-analysis" OR "phase II" OR "phase III" '
    'OR "cohort study" OR "guideline"'
    ')'
)


# --------------------------------------------------------------------------- #
# OpenAlex fetching
# --------------------------------------------------------------------------- #
def reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Rebuild plain text from OpenAlex's abstract_inverted_index."""
    if not inverted_index:
        return None
    positions = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


RETRY_STATUS = {429, 500, 502, 503, 504}


def _get_json(params: dict, *, retries: int = 8, timeout: int = 60) -> dict:
    """GET /works with retries. Rate limits (429) and server errors wait for OpenAlex's
    ``Retry-After`` or back off exponentially (up to 2 min). Messages never include the
    URL, which carries the API key."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(OPENALEX_WORKS, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last, wait = type(exc).__name__, min(2 ** (attempt + 1), 120)
        else:
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS:
                raise RuntimeError(f"OpenAlex request failed: HTTP {r.status_code} {r.text[:300]}")
            last = f"HTTP {r.status_code}"
            retry_after = r.headers.get("Retry-After", "")
            wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else min(2 ** (attempt + 1), 120)
        print(f"  OpenAlex: {last}; retry {attempt + 1}/{retries} in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"OpenAlex request failed after {retries} attempts: {last}"
                       + ("" if params.get("api_key") else " (no OpenAlex API key: set OPENALEX_API_KEY)"))


def _auth(params: dict, email=None, api_key=None) -> dict:
    if email:
        params["mailto"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def to_record(work: dict, abstract: str | None) -> dict:
    """What the build keeps of an OpenAlex work: title, abstract and the app's metadata."""
    return {"id": work.get("id") or "", "title": work.get("title") or "Unknown Title",
            "abstract": abstract or "", "metadata": build_metadatas([work])[0]}


class FetchCache:
    """Papers fetched so far for one part of the build (JSONL) and where to continue
    (JSON state), under ``<out-dir>/openalex/``. Lets an interrupted build resume."""

    def __init__(self, cache_dir: Path | None, name: str, signature: str):
        self.records_path = cache_dir / f"{name}.jsonl" if cache_dir else None
        self.state_path = cache_dir / f"{name}_state.json" if cache_dir else None
        self.signature = signature
        self.state = {"signature": signature}
        self.records: list[dict] = []
        if self.state_path and self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("signature") == signature and self.records_path.exists():
                self.state = state
                with self.records_path.open(encoding="utf-8") as fh:
                    self.records = [json.loads(line) for line in fh if line.strip()]
                print(f"  resuming from cache: {len(self.records)} papers in {self.records_path}")
            else:
                print(f"  cache in {self.state_path.parent} is for a different query; starting over")
                self.records_path.unlink(missing_ok=True)

    def add(self, records: list[dict], **state):
        self.records += records
        self.state.update(state)
        if self.records_path:
            self.records_path.parent.mkdir(parents=True, exist_ok=True)
            with self.records_path.open("a", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
            self.state_path.write_text(json.dumps(self.state))


def _signature(*parts) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def search_openalex(search_phrase, *, max_results, from_date=None, to_date=None,
                    email=None, api_key=None, page_pause=0.0, cache_dir=None, cache_name="search"):
    """Cursor-paginate an OpenAlex `default.search`. Returns a list of paper records."""
    filters = [f"default.search:{search_phrase}", "has_abstract:true", "language:en"]
    if from_date:
        filters.append(f"from_publication_date:{from_date}")
    if to_date:
        filters.append(f"to_publication_date:{to_date}")
    params = {"filter": ",".join(filters), "sort": "relevance_score:desc", "per_page": 200}
    cache = FetchCache(cache_dir, cache_name, _signature(params))
    if cache.state.get("done") or len(cache.records) >= max_results:
        return cache.records[:max_results]
    params = _auth({**params, "cursor": cache.state.get("cursor", "*")}, email, api_key)

    page = cache.state.get("pages", 0)
    while params["cursor"] and len(cache.records) < max_results:
        data = _get_json(params)
        batch = data.get("results", [])
        if not batch:
            params["cursor"] = None
            break
        params["cursor"] = data.get("meta", {}).get("next_cursor")
        page += 1
        cache.add([to_record(w, reconstruct_abstract(w.get("abstract_inverted_index"))) for w in batch],
                  cursor=params["cursor"], pages=page)
        total = data.get("meta", {}).get("count")
        print(f"  page {page}: +{len(batch)} (have {len(cache.records)}"
              + (f" / ~{total}" if total else "") + ")")
        if page_pause:
            time.sleep(page_pause)
    if not params["cursor"]:          # all results fetched (not just up to --max-results)
        cache.add([], done=True)
    return cache.records[:max_results]


def fetch_papers_by_ids(paper_ids, *, batch_size=100, email=None, api_key=None, cache_dir=None):
    """Fetch specific works by OpenAlex id, in batches. Returns a list of paper records."""
    ids = [str(x).strip() for x in paper_ids if str(x).strip()]
    cache = FetchCache(cache_dir, "variantscape", _signature(ids, batch_size))
    batches = range(cache.state.get("batches_done", 0) * batch_size, len(ids), batch_size)
    for start in batches:
        batch = ids[start:start + batch_size]
        data = _get_json(_auth({"filter": f"openalex_id:{'|'.join(batch)}", "per_page": len(batch)},
                               email, api_key))
        got = [to_record(w, reconstruct_abstract(w.get("abstract_inverted_index"))) for w in data.get("results", [])]
        cache.add(got, batches_done=start // batch_size + 1)
        print(f"  batch {start // batch_size + 1}: {len(cache.records)}/{len(ids)} fetched")
    return cache.records


def load_previous_search(prev_dir: Path, collection: str, legacy_variantscape_csv: Path) -> tuple[list[dict], dict]:
    """The search part of an earlier build, as paper records (no new OpenAlex search).

    Builds made by this script keep their fetched papers in ``openalex/search.jsonl``;
    for older builds they are read back from the Chroma collection, leaving out the
    VariantScape papers at its start (count from ``build_manifest.json``, else the
    leading papers that are in the legacy VariantScape CSV)."""
    manifest_path = prev_dir / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    state_path = prev_dir / "openalex" / "search_state.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("done"):
        with (prev_dir / "openalex" / "search.jsonl").open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()], manifest

    import chromadb
    client = chromadb.PersistentClient(path=str(prev_dir), settings=chromadb.Settings(anonymized_telemetry=False))
    got = client.get_collection(collection).get(include=["documents", "metadatas"])
    rows = sorted(zip(got["ids"], got["documents"], got["metadatas"]), key=lambda r: int(r[0]))
    if "variantscape_papers" in manifest:
        skip = int(manifest["variantscape_papers"])
    else:
        legacy = {f"https://openalex.org/{x}" for x in pd.read_csv(legacy_variantscape_csv).iloc[:, 0].astype(str)}
        skip = next((i for i, (_, _, m) in enumerate(rows) if (m or {}).get("openAlex_id") not in legacy), len(rows))
    records = []
    for _, doc, meta in rows[skip:]:
        title = (meta or {}).get("titles") or "Unknown Title"
        abstract = doc[len(title):].strip() if doc.startswith(title) else doc
        records.append({"id": (meta or {}).get("openAlex_id", ""), "title": title, "abstract": abstract,
                        "metadata": meta})
    print(f"  {len(records)} search papers from the collection in {prev_dir} (skipped {skip} VariantScape papers)")
    return records, manifest


# --------------------------------------------------------------------------- #
# Building the collection
# --------------------------------------------------------------------------- #
def build_metadatas(works: list[dict]) -> list[dict]:
    """Port of the notebook's metadata cell. Keys must match what the app reads."""
    metas = []
    for w in works:
        authorships = w.get("authorships") or []
        country = ""
        if authorships:
            countries = authorships[0].get("countries") or []
            if countries:
                country = countries[0]

        first_author = "Unknown Author"
        if authorships:
            first_author = (authorships[0].get("author") or {}).get("display_name") or "Unknown Author"

        ploc = w.get("primary_location") or {}
        source = ploc.get("source") or {}
        journal = source.get("display_name") or "Unknown Journal"

        metas.append({
            "titles": w.get("title") or "Unknown Title",
            "first_author": first_author,
            "journal": journal,
            "year": w.get("publication_year") or "Unknown Publication Year",
            "openAlex_id": w.get("id") or "Unknown OpenAlex ID",
            "countryMainAuthor": country,
        })
    return metas


def make_documents(titles, abstracts):
    return [
        f"{titles[i]} {abstracts[i] or ''}".strip()
        for i in range(len(titles))
    ]


def write_collection(out_dir: Path, name: str, model: str, ids, documents, metadatas,
                     *, recreate: bool, add_batch_size: int):
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(out_dir))
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

    existing = {c.name for c in client.list_collections()}
    if name in existing:
        current = client.get_collection(name)
        if recreate:
            print(f"  --recreate: dropping existing collection '{name}' ({current.count()} docs)")
            client.delete_collection(name)
        elif current.count() > 0:
            raise SystemExit(
                f"Collection '{name}' in {out_dir} already has {current.count()} documents.\n"
                f"Use a fresh --out-dir, or pass --recreate to rebuild it."
            )

    if name in {c.name for c in client.list_collections()}:
        collection = client.get_collection(name=name, embedding_function=embed)
    else:
        collection = client.create_collection(
            name=name, embedding_function=embed, metadata={"hnsw:space": "cosine"},
        )

    n = len(ids)
    for start in range(0, n, add_batch_size):
        end = min(start + add_batch_size, n)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  embedded {end}/{n}")

    return collection.count()


def write_manifest(out_dir: Path, info: dict):
    (out_dir / "build_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Left-panel filter metadata (one-hot matrix, LLM-classified)
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = (
    "You extract structured trial-population metadata from a study title and "
    "abstract. Mark an option only when the text explicitly supports it for the "
    "study population; if a category is not mentioned, return an empty list for "
    "it. Respond with a single JSON object and nothing else."
)


def _filter_schema():
    """(groups, label->column map, ordered column list) from literature.FILTER_GROUPS."""
    import literature  # lightweight (only imports config); done lazily

    groups = [(key, [(col, label) for col, label in opts])
              for key, _display, opts in literature.FILTER_GROUPS]
    label_to_col = {key: {label: col for col, label in opts} for key, opts in groups}
    columns = [col for _key, opts in groups for col, _label in opts]
    return groups, label_to_col, columns


def _extract_prompt(text, groups):
    lines = ["Study title + abstract:", (text or "").strip()[:6000], "",
             "For each category, list the options supported by the study population:"]
    for key, opts in groups:
        lines.append(f"- {key}: {json.dumps([label for _c, label in opts])}")
    lines += [
        "",
        "Guidance:",
        '- "Renal Cell Cancer" is a kind of "Kidney Cancer"; for an RCC population include both.',
        '- "Pelvis Cancer" = renal pelvis / upper-tract urothelial carcinoma.',
        '- cancer_type may contain several entries for pan-tumour studies.',
        "",
        "Return JSON with exactly these keys: "
        + json.dumps([key for key, _ in groups]),
    ]
    return "\n".join(lines)


def _parse_json_object(raw):
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
        raw = raw[4:].strip() if raw.lower().startswith("json") else raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


_JSON_MODE = {"enabled": True}  # dropped process-wide once an endpoint rejects it


def _classify_one(client, model, text, groups, *, retries=4, backoff=2.0):
    """Classify one document. Returns the parsed dict (possibly {}), or None if
    the request never succeeded (connection / server errors) -- None rows are
    left blank AND not cached, so a rerun retries only them."""
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": _extract_prompt(text, groups)},
    ]
    last = None
    for attempt in range(retries):
        try:
            kwargs = dict(model=model, messages=messages, temperature=0)
            if _JSON_MODE["enabled"]:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:
                # Only a 400/422 from the server means the endpoint rejects JSON mode;
                # connection errors (no response) and 401/403/429/5xx are auth,
                # rate-limit or server problems and fall through to retry/backoff.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if _JSON_MODE["enabled"] and status in (400, 422):
                    _JSON_MODE["enabled"] = False
                    print("  note: endpoint rejected response_format; retrying without JSON mode")
                    resp = client.chat.completions.create(
                        model=model, messages=messages, temperature=0)
                else:
                    raise
            return _parse_json_object(resp.choices[0].message.content)
        except Exception as exc:  # network / rate limit / server error
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    cause = getattr(last, "__cause__", None)
    message = f"  classify failed after {retries} tries: {last!r}" + (f"  (cause: {cause!r})" if cause else "")
    key = str(getattr(client, "api_key", "") or "").strip()
    if len(key) >= 8:   # e.g. "Illegal header value b'Bearer <key>\\n'" would print the key
        message = message.replace(key, key[:2] + "…" + key[-4:])
    print(message)
    return None


_NO_OAID = "Unknown OpenAlex ID"


def _cache_key(oaid, pos):
    """Cache key for a paper: its OpenAlex id, or a positional fallback."""
    return oaid if (oaid and oaid != _NO_OAID) else f"__pos{pos}"


def _load_reuse_map(reuse_from: Path | None, columns):
    """{openalex_id: {col: bool}} from a previous filter_metadata.csv.

    Only used when that file's filter columns are byte-for-byte the current
    schema -- so changing FILTER_GROUPS automatically forces a full re-classify.
    """
    if not reuse_from or not reuse_from.exists():
        return {}
    with open(reuse_from, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        header = reader.fieldnames or []
        prev_cols = [c for c in header if c not in ("id", "Text", "openalex_id")]
        if "openalex_id" not in header or prev_cols != list(columns):
            print(f"  reuse: {reuse_from} predates the current filter set "
                  f"-> classifying every paper")
            return {}
        out = {}
        for row in reader:
            oaid = (row.get("openalex_id") or "").strip()
            if not oaid or oaid == _NO_OAID:
                continue
            out[oaid] = {c: (row.get(c, "").strip().upper() == "TRUE") for c in columns}
    print(f"  reuse: {len(out)} classified papers available from {reuse_from}")
    return out


def build_filter_metadata(ids, documents, openalex_ids, out_path: Path, *,
                          model, workers=8, limit=None, reuse_from=None):
    """Classify each document into the FILTER_GROUPS buckets and write the CSV.

    Papers already classified under the same filter schema -- in ``reuse_from``
    (a previous build) or ``<out_path>.partial.jsonl`` (an interrupted run) --
    are reused by OpenAlex id and skip the LLM. Ctrl-C is safe; rerun to resume.
    """
    groups, label_to_col, columns = _filter_schema()
    n = len(ids) if limit is None else min(limit, len(ids))
    blank = {c: False for c in columns}

    endpoint = config.OPENAI_BASE_URL or "https://api.openai.com (OpenAI SDK default)"
    api_key = (config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY") or "").strip()
    source = ("from the shell environment (overrides evidence-database/.env)" if _KEY_FROM_SHELL
              else "from evidence-database/.env")
    print(f"  LLM endpoint: {endpoint}  |  api key: "
          + (f"set ({len(api_key)} chars, ...{api_key[-4:]}, {source})" if api_key else f"NOT SET ({source})"))
    if not api_key and limit != 0:
        raise SystemExit(
            "No LLM API key: OPENAI_API_KEY is empty in this environment, so every classification would "
            "fail (\"Illegal header value b'Bearer '\").\n"
            "Set OPENAI_API_KEY in evidence-database/.env (the running app may get it from its service "
            "definition instead), then classify the existing collection with:\n"
            "  python db_initializations/refresh_literature_db.py --filter-metadata-only --out-dir <this build>")

    def cols_from_data(data):
        cols = dict(blank)
        for gkey, _opts in groups:
            picked = data.get(gkey) or []
            if isinstance(picked, str):
                picked = [picked]
            for label in picked:
                col = label_to_col.get(gkey, {}).get(label)
                if col:
                    cols[col] = True
        return cols

    # cache: key -> {col: bool}. Never reuse the very file we are about to write.
    reuse_path = Path(reuse_from) if reuse_from else None
    if reuse_path and reuse_path.resolve() == Path(out_path).resolve():
        reuse_path = None
    cache = _load_reuse_map(reuse_path, columns)
    reused_keys = set(cache)

    partial = Path(str(out_path) + ".partial.jsonl")
    resumed = 0
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") and "cols" in rec and rec["key"] not in cache:
                cache[rec["key"]] = rec["cols"]
                resumed += 1
        if resumed:
            print(f"  resuming: {resumed} rows from {partial.name}")

    # Unique positions still needing an LLM call (de-duped by cache key).
    todo, seen = [], set()
    for i in range(n):
        key = _cache_key(openalex_ids[i], i)
        if key in cache or key in seen:
            continue
        seen.add(key)
        todo.append(i)

    reused_n = sum(1 for i in range(n) if _cache_key(openalex_ids[i], i) in reused_keys)
    print(f"  {n} papers: {reused_n} reused, {resumed} resumed, "
          f"{len(todo)} to classify with {model}")

    failed = 0
    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=config.OPENAI_BASE_URL)
        written = 0
        with partial.open("a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_classify_one, client, model, documents[i], groups): i
                       for i in todo}
            try:
                for fut in as_completed(futures):
                    i = futures[fut]
                    data = fut.result()
                    if data is None:            # request never succeeded
                        failed += 1
                        if written == 0 and failed >= 20:
                            # Nothing works (key, endpoint, network): stop instead of
                            # retrying every remaining paper
                            for f in futures:
                                f.cancel()
                            ex.shutdown(wait=False, cancel_futures=True)
                            raise SystemExit(
                                f"Stopping: the first {failed} classifications all failed (see the errors "
                                f"above; endpoint {endpoint}). Fix the LLM settings, then rerun with "
                                f"--filter-metadata-only --out-dir {Path(out_path).parent}")
                        continue                # not cached -> retried on rerun
                    ckey = _cache_key(openalex_ids[i], i)
                    cols = cols_from_data(data)
                    cache[ckey] = cols
                    fh.write(json.dumps({"key": ckey, "cols": cols}) + "\n")
                    fh.flush()
                    written += 1
                    if (written + failed) % 100 == 0 or written + failed == len(todo):
                        print(f"  classified {written}/{len(todo)}"
                              + (f" ({failed} failed)" if failed else ""))
            except KeyboardInterrupt:
                print("\n  interrupted -- partial progress saved; rerun to resume")
                raise

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["id", "Text", "openalex_id"] + columns)
        for i in range(n):
            cols = cache.get(_cache_key(openalex_ids[i], i), blank)
            writer.writerow([ids[i], documents[i], openalex_ids[i]]
                            + ["TRUE" if cols.get(c) else "FALSE" for c in columns])
    partial.unlink(missing_ok=True)
    print(f"  wrote {n} rows ({len(columns)} filter columns) -> {out_path}")
    if failed:
        print(f"  WARNING: {failed} papers could not be classified (left blank). "
              f"Fix connectivity and rerun (optionally --filter-metadata-only) to retry just those.")
    return {
        "rows": n,
        "reused_from_previous": reused_n,
        "llm_calls": len(todo),
        "failed": failed,
        "reuse_from": str(reuse_from) if reuse_from else None,
    }


def _run_filter_metadata_only(args, out_dir: Path, reuse_from, parser):
    """(Re)build filter_metadata.csv from an already-embedded collection."""
    if not out_dir.exists():
        parser.error(f"--filter-metadata-only needs an existing build; {out_dir} not found")

    import chromadb

    print(f"Loading documents from collection '{args.collection}' in {out_dir} ...")
    client = chromadb.PersistentClient(path=str(out_dir))
    try:
        col = client.get_collection(args.collection)
    except Exception as exc:
        parser.error(f"cannot open collection '{args.collection}' in {out_dir}: {exc}")

    got = col.get(include=["documents", "metadatas"])
    rows = sorted(
        zip(got["ids"], got["documents"], got.get("metadatas") or []),
        key=lambda t: int(t[0]) if str(t[0]).lstrip("-").isdigit() else (1 << 62),
    )
    ids = [t[0] for t in rows]
    documents = [t[1] for t in rows]
    openalex_ids = [((t[2] or {}).get("openAlex_id") or _NO_OAID) for t in rows]
    print(f"  {len(ids)} documents")

    fm_path = out_dir / "filter_metadata.csv"
    fm = build_filter_metadata(
        ids, documents, openalex_ids, fm_path,
        model=args.llm_model, workers=args.filter_metadata_workers,
        limit=args.filter_metadata_limit, reuse_from=reuse_from,
    )

    mpath = out_dir / "build_manifest.json"
    manifest = {}
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except ValueError:
            manifest = {}
    manifest.update({
        "filter_metadata_rebuilt_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "filter_metadata_rows": fm["rows"],
        "filter_metadata_reused": fm["reused_from_previous"],
        "filter_metadata_llm_calls": fm["llm_calls"],
        "filter_metadata_failed": fm["failed"],
        "filter_metadata_model": args.llm_model,
    })
    write_manifest(out_dir, manifest)
    print(f"\nDone. filter_metadata.csv: {fm['rows']} rows, "
          f"{fm['llm_calls'] - fm['failed']} newly classified, {fm['failed']} failed.")
    if fm["failed"]:
        print("Rerun this same command to retry the failed rows once connectivity is fixed.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--search-phrase", default=DEFAULT_SEARCH_PHRASE,
                   help="OpenAlex default.search query (default: all GU cancers + trial types)")
    p.add_argument("--from-date", default=os.environ.get("EVIDENCE_DB_LIT_FROM_DATE"),
                   help="OpenAlex from_publication_date, YYYY-MM-DD (default: no lower bound)")
    p.add_argument("--to-date", default=None,
                   help="OpenAlex to_publication_date, YYYY-MM-DD (default: today, i.e. no bound)")
    p.add_argument("--max-results", type=int, default=100_000,
                   help="cap on papers pulled from the search (default: 100000)")
    p.add_argument("--no-variantscape", dest="include_variantscape", action="store_false",
                   help="do NOT merge the VariantScape paper set (it is included by default)")
    p.add_argument("--variantscape-csv", default=str(
        config.VARIANTSCAPE_PAPERS_PATH if config.VARIANTSCAPE_PAPERS_PATH.exists()
        else REPO_ROOT / "cleaned_df_v4_corrected.csv"),
        help="paper set to merge (column --variantscape-id-column, OpenAlex ids W...)")
    p.add_argument("--variantscape-id-column", default="PaperId")
    p.add_argument("--dedupe", action="store_true",
                   help="drop search papers whose OpenAlex id is already in the VariantScape set")
    p.add_argument("--out-dir",
                   default=str(REPO_ROOT / f"chroma_data_{dt.date.today():%Y%m%d}"),
                   help="destination Chroma directory (default: dated, e.g. chroma_data_YYYYMMDD)")
    p.add_argument("--collection", default=config.CHROMA_COLLECTION)
    p.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    p.add_argument("--recreate", action="store_true",
                   help="rebuild the collection if it already exists in --out-dir")
    p.add_argument("--add-batch-size", type=int, default=100)
    p.add_argument("--email", default=os.environ.get("OPENALEX_EMAIL"),
                   help="contact email for the OpenAlex polite pool (or set OPENALEX_EMAIL)")
    p.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"),
                   help="OpenAlex API key (or set OPENALEX_API_KEY; without one the rate limits are much lower)")
    p.add_argument("--reuse-search-from", default=None,
                   help="take the search papers from an earlier build directory (e.g. the live "
                        "chroma_data_YYYYMMDD) instead of searching OpenAlex again")
    p.add_argument("--top-up", action="store_true",
                   help="with --reuse-search-from: also search for papers published since that build "
                        "(30 days overlap) and add the new ones")
    p.add_argument("--page-pause", type=float, default=0.0,
                   help="seconds to sleep between search pages")
    p.add_argument("--no-filter-metadata", dest="build_filter_metadata_csv",
                   action="store_false",
                   help="do NOT build the left-panel filter metadata CSV (built by default)")
    p.add_argument("--llm-model", default=config.LLM_MODEL,
                   help="model for filter-metadata classification (default: the answer-path model)")
    p.add_argument("--filter-metadata-workers", type=int, default=8,
                   help="concurrent LLM requests while classifying (default: 8)")
    p.add_argument("--filter-metadata-limit", type=int, default=None,
                   help="classify only the first N documents (for a trial run)")
    p.add_argument("--reuse-from",
                   default=str(config.CHROMA_DB_PATH / "filter_metadata.csv"),
                   help="previous filter_metadata.csv to reuse classifications from, "
                        "matched by OpenAlex id (default: the current live build; "
                        "ignored automatically if its filter columns differ)")
    p.add_argument("--no-reuse", action="store_true",
                   help="re-classify every paper even if a previous result exists")
    p.add_argument("--filter-metadata-only", action="store_true",
                   help="skip the OpenAlex fetch + embedding: (re)build only "
                        "filter_metadata.csv from the existing collection in --out-dir")
    p.add_argument("--dry-run", action="store_true",
                   help="fetch and report counts but do not write the collection")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    live_dir = Path(str(config.CHROMA_DB_PATH)).resolve()
    reuse_from = None if args.no_reuse else args.reuse_from

    if args.filter_metadata_only:
        return _run_filter_metadata_only(args, out_dir, reuse_from, p)

    if out_dir == live_dir and not args.recreate:
        p.error(f"--out-dir is the live store ({live_dir}). Use a fresh directory, or --recreate.")

    started = dt.datetime.now(dt.timezone.utc)
    cache_dir = out_dir / "openalex"     # fetched papers; a rerun with the same --out-dir resumes
    print(f"OpenAlex search phrase:\n  {args.search_phrase}")
    print(f"OpenAlex API key: {'set' if args.api_key else 'NOT SET (set OPENALEX_API_KEY)'}; "
          f"fetch cache: {cache_dir}\n")

    # 1. VariantScape paper set (prepended -> keeps its ids stable at 0..N)
    vs_records = []
    if args.include_variantscape:
        csv_path = Path(args.variantscape_csv)
        print(f"Fetching VariantScape papers from {csv_path.name} ...")
        col = pd.read_csv(csv_path)[args.variantscape_id_column]
        vs_records = fetch_papers_by_ids(col, email=args.email, api_key=args.api_key, cache_dir=cache_dir)
        print(f"  -> {len(vs_records)} VariantScape papers\n")
    else:
        print("Skipping VariantScape merge (--no-variantscape)\n")

    # 2. GU-cancer literature search (or the search of an earlier build, optionally topped up)
    reused_from, topped_up = None, 0
    if args.reuse_search_from:
        reused_from = Path(args.reuse_search_from).resolve()
        print(f"Reusing the search papers of {reused_from} ...")
        s_records, prev_manifest = load_previous_search(reused_from, args.collection,
                                                        REPO_ROOT / "cleaned_df_v4_corrected.csv")
        if prev_manifest.get("search_phrase") not in (None, args.search_phrase):
            print("  NOTE: that build used a different search phrase")
        if args.top_up:
            built = prev_manifest.get("built_at") or dt.datetime.fromtimestamp(
                reused_from.stat().st_mtime, dt.timezone.utc).isoformat()
            since = (dt.date.fromisoformat(built[:10]) - dt.timedelta(days=30)).isoformat()
            print(f"Topping up: papers published since {since} ...")
            new = search_openalex(args.search_phrase, max_results=args.max_results, from_date=since,
                                  to_date=args.to_date, email=args.email, api_key=args.api_key,
                                  page_pause=args.page_pause, cache_dir=cache_dir, cache_name="topup")
            have = {r["id"] for r in s_records}
            added = [r for r in new if r["id"] not in have]
            s_records, topped_up = added + s_records, len(added)
            print(f"  +{topped_up} new papers")
    else:
        print("Searching OpenAlex for GU-cancer literature ...")
        s_records = search_openalex(
            args.search_phrase, max_results=args.max_results,
            from_date=args.from_date, to_date=args.to_date,
            email=args.email, api_key=args.api_key, page_pause=args.page_pause, cache_dir=cache_dir,
        )
    print(f"  -> {len(s_records)} search papers\n")

    # 3. Optional de-duplication of the search set against VariantScape
    removed = 0
    if args.dedupe and vs_records:
        seen = {r["id"] for r in vs_records}
        kept = [r for r in s_records if r["id"] not in seen]
        removed = len(s_records) - len(kept)
        s_records = kept
        print(f"De-dupe: removed {removed} search papers already in the VariantScape set\n")

    records = vs_records + s_records
    ids = [str(i) for i in range(len(records))]
    metadatas = [r["metadata"] for r in records]
    openalex_ids = [m["openAlex_id"] for m in metadatas]
    documents = make_documents([r["title"] for r in records], [r["abstract"] for r in records])
    print(f"Total documents to embed: {len(documents)}")

    manifest = {
        "built_at": started.isoformat(),
        "search_phrase": args.search_phrase,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "openalex_search_papers": len(s_records),
        "search_reused_from": str(reused_from) if reused_from else None,
        "search_topped_up": topped_up,
        "variantscape_papers": len(vs_records),
        "variantscape_csv": str(args.variantscape_csv) if args.include_variantscape else None,
        "deduped": bool(args.dedupe and vs_records),
        "search_duplicates_removed": removed,
        "total_documents": len(documents),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "chroma_path": str(out_dir),
        "filter_metadata": bool(args.build_filter_metadata_csv),
        "filter_metadata_reuse_from": None if args.no_reuse else args.reuse_from,
    }

    if args.dry_run:
        print("\n--dry-run: not writing the collection or filter metadata. "
              "Manifest that would be written:")
        print(json.dumps(manifest, indent=2))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting collection '{args.collection}' to {out_dir} ...")
    count = write_collection(
        out_dir, args.collection, args.embedding_model,
        ids, documents, metadatas,
        recreate=args.recreate, add_batch_size=args.add_batch_size,
    )
    manifest["documents_in_collection"] = count

    if args.build_filter_metadata_csv:
        fm_path = out_dir / "filter_metadata.csv"
        print(f"\nBuilding left-panel filter metadata -> {fm_path}")
        fm = build_filter_metadata(
            ids, documents, openalex_ids, fm_path,
            model=args.llm_model,
            workers=args.filter_metadata_workers,
            limit=args.filter_metadata_limit,
            reuse_from=reuse_from,
        )
        manifest["filter_metadata_rows"] = fm["rows"]
        manifest["filter_metadata_reused"] = fm["reused_from_previous"]
        manifest["filter_metadata_llm_calls"] = fm["llm_calls"]
        manifest["filter_metadata_model"] = args.llm_model

    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(out_dir, manifest)

    rel = os.path.relpath(out_dir, REPO_ROOT)
    print(f"\nDone. {count} documents in '{args.collection}'.")
    print(f"Manifest: {out_dir / 'build_manifest.json'}")
    print("\nNext steps:")
    print("  1. Point the app at the new store -- in evidence-database/.env:")
    print(f"       EVIDENCE_DB_CHROMA_PATH={rel}")
    print("     (this also repoints the left-panel filter metadata CSV).")
    print("  2. Restart the app.")


if __name__ == "__main__":
    main()
