#!/usr/bin/env python3
"""Rebuild the LiteratureDB ChromaDB vector store from OpenAlex.

This replaces ``db_initializations/make_chromadb_visualization_v2.ipynb`` with a
re-runnable script.

What it does
------------
1. Searches OpenAlex for genitourinary-cancer clinical literature
   (see ``DEFAULT_SEARCH_PHRASE`` / ``--search-phrase``).
2. Fetches the VariantScape paper set by OpenAlex id from
   ``cleaned_df_v4_corrected.csv`` and prepends it -- the /variantscape part of
   the app relies on these papers being in the collection. Skip with
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

then point the app at the new store (the exact line is printed at the end)::

    # evidence-database/.env
    EVIDENCE_DB_CHROMA_PATH=chroma_data_YYYYMMDD

``EVIDENCE_DB_CHROMA_PATH`` also repoints ``LITERATURE_METADATA_CSV`` at the
matching ``filter_metadata.csv``, so one variable moves both.

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


def _get_json(params: dict, *, retries: int = 4, backoff: float = 2.0, timeout: int = 30) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(OPENALEX_WORKS, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = exc
            wait = backoff ** attempt
            print(f"  request failed ({exc}); retry {attempt + 1}/{retries} in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"OpenAlex request failed after {retries} attempts: {last}")


def search_openalex(search_phrase, *, max_results, from_date=None, to_date=None,
                    email=None, page_pause=0.0):
    """Cursor-paginate an OpenAlex `default.search`. Returns (works, abstracts)."""
    filters = [f"default.search:{search_phrase}", "has_abstract:true", "language:en"]
    if from_date:
        filters.append(f"from_publication_date:{from_date}")
    if to_date:
        filters.append(f"to_publication_date:{to_date}")

    params = {
        "filter": ",".join(filters),
        "sort": "relevance_score:desc",
        "per_page": 200,
        "cursor": "*",
    }
    if email:
        params["mailto"] = email

    works, abstracts = [], []
    page = 0
    while params["cursor"] and len(works) < max_results:
        data = _get_json(params)
        batch = data.get("results", [])
        if not batch:
            break
        for w in batch:
            works.append(w)
            abstracts.append(reconstruct_abstract(w.get("abstract_inverted_index")))
        params["cursor"] = data.get("meta", {}).get("next_cursor")
        page += 1
        total = data.get("meta", {}).get("count")
        print(f"  page {page}: +{len(batch)} (have {len(works)}"
              + (f" / ~{total}" if total else "") + ")")
        if page_pause:
            time.sleep(page_pause)

    return works[:max_results], abstracts[:max_results]


def fetch_papers_by_ids(paper_ids, *, batch_size=100, email=None):
    """Fetch specific works by OpenAlex id, in batches. Returns (works, abstracts)."""
    ids = [str(x).strip() for x in paper_ids if str(x).strip()]
    works, abstracts = [], []

    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        params = {
            "filter": f"openalex_id:{'|'.join(batch)}",
            "per_page": len(batch),
            "cursor": "*",
        }
        if email:
            params["mailto"] = email

        while params["cursor"]:
            data = _get_json(params)
            got = data.get("results", [])
            for w in got:
                works.append(w)
                abstracts.append(reconstruct_abstract(w.get("abstract_inverted_index")))
            params["cursor"] = data.get("meta", {}).get("next_cursor")
            if not got:
                break
        print(f"  batch {start // batch_size + 1}: {len(works)}/{len(ids)} fetched")

    return works, abstracts


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


def _classify_one(client, model, text, groups, *, retries=4, backoff=2.0):
    messages = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {"role": "user", "content": _extract_prompt(text, groups)},
    ]
    last = None
    for attempt in range(retries):
        try:
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception:  # endpoint may not support response_format
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=0,
                )
            return _parse_json_object(resp.choices[0].message.content)
        except Exception as exc:  # network / rate limit / server error
            last = exc
            time.sleep(backoff ** attempt)
    print(f"  classify failed after {retries} tries ({last}); row left blank")
    return {}


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

    # cache: key -> {col: bool}
    cache = _load_reuse_map(Path(reuse_from) if reuse_from else None, columns)
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

    if todo:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY or None, base_url=config.OPENAI_BASE_URL)
        written = 0
        with partial.open("a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_classify_one, client, model, documents[i], groups): i
                       for i in todo}
            try:
                for fut in as_completed(futures):
                    i = futures[fut]
                    key = _cache_key(openalex_ids[i], i)
                    cols = cols_from_data(fut.result())
                    cache[key] = cols
                    fh.write(json.dumps({"key": key, "cols": cols}) + "\n")
                    fh.flush()
                    written += 1
                    if written % 100 == 0 or written == len(todo):
                        print(f"  classified {written}/{len(todo)}")
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
    return {
        "rows": n,
        "reused_from_previous": reused_n,
        "llm_calls": len(todo),
        "reuse_from": str(reuse_from) if reuse_from else None,
    }


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
    p.add_argument("--variantscape-csv", default=str(REPO_ROOT / "cleaned_df_v4_corrected.csv"))
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
    p.add_argument("--dry-run", action="store_true",
                   help="fetch and report counts but do not write the collection")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    live_dir = Path(str(config.CHROMA_DB_PATH)).resolve()
    if out_dir == live_dir and not args.recreate:
        p.error(f"--out-dir is the live store ({live_dir}). Use a fresh directory, or --recreate.")

    started = dt.datetime.now(dt.timezone.utc)
    print(f"OpenAlex search phrase:\n  {args.search_phrase}\n")

    # 1. VariantScape paper set (prepended -> keeps its ids stable at 0..N)
    vs_works, vs_abstracts = [], []
    if args.include_variantscape:
        csv_path = Path(args.variantscape_csv)
        print(f"Fetching VariantScape papers from {csv_path.name} ...")
        col = pd.read_csv(csv_path)[args.variantscape_id_column]
        vs_works, vs_abstracts = fetch_papers_by_ids(col, email=args.email)
        print(f"  -> {len(vs_works)} VariantScape papers\n")
    else:
        print("Skipping VariantScape merge (--no-variantscape)\n")

    # 2. GU-cancer literature search
    print("Searching OpenAlex for GU-cancer literature ...")
    s_works, s_abstracts = search_openalex(
        args.search_phrase, max_results=args.max_results,
        from_date=args.from_date, to_date=args.to_date,
        email=args.email, page_pause=args.page_pause,
    )
    print(f"  -> {len(s_works)} search papers\n")

    # 3. Optional de-duplication of the search set against VariantScape
    removed = 0
    if args.dedupe and vs_works:
        seen = {w.get("id") for w in vs_works}
        kept = [(w, a) for w, a in zip(s_works, s_abstracts) if w.get("id") not in seen]
        removed = len(s_works) - len(kept)
        s_works = [w for w, _ in kept]
        s_abstracts = [a for _, a in kept]
        print(f"De-dupe: removed {removed} search papers already in the VariantScape set\n")

    works = vs_works + s_works
    abstracts = vs_abstracts + s_abstracts
    ids = [str(i) for i in range(len(works))]
    titles = [w.get("title") or "Unknown Title" for w in works]
    metadatas = build_metadatas(works)
    openalex_ids = [m["openAlex_id"] for m in metadatas]
    documents = make_documents(titles, abstracts)
    print(f"Total documents to embed: {len(documents)}")

    reuse_from = None if args.no_reuse else args.reuse_from

    manifest = {
        "built_at": started.isoformat(),
        "search_phrase": args.search_phrase,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "openalex_search_papers": len(s_works),
        "variantscape_papers": len(vs_works),
        "variantscape_csv": str(args.variantscape_csv) if args.include_variantscape else None,
        "deduped": bool(args.dedupe and vs_works),
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
