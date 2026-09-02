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

Typical use
-----------
    python db_initializations/refresh_literature_db.py --from-date 2015-01-01

then point the app at the new store (the exact line is printed at the end)::

    # evidence-database/.env
    EVIDENCE_DB_CHROMA_PATH=chroma_data_YYYYMMDD

Caveats
-------
* Always build into a NEW ``--out-dir`` (the default is dated). Re-embedding into
  the live directory risks duplicate / colliding document ids -- the script
  refuses to add to a non-empty collection unless ``--recreate`` is given.
* The left-panel publication filters read a positionally-aligned one-hot CSV
  (``static/evidence-db-angular/assets/prostate-metadata.csv``). A rebuilt corpus
  makes that file stale; regenerate it with its own pipeline, or the filters
  will mis-map to the wrong papers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
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
    documents = make_documents(titles, abstracts)
    print(f"Total documents to embed: {len(documents)}")

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
    }

    if args.dry_run:
        print("\n--dry-run: not writing the collection. Manifest that would be written:")
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
    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_manifest(out_dir, manifest)

    rel = os.path.relpath(out_dir, REPO_ROOT)
    print(f"\nDone. {count} documents in '{args.collection}'.")
    print(f"Manifest: {out_dir / 'build_manifest.json'}")
    print("\nNext steps:")
    print(f"  1. Point the app at the new store -- in evidence-database/.env:")
    print(f"       EVIDENCE_DB_CHROMA_PATH={rel}")
    print(f"  2. Regenerate static/evidence-db-angular/assets/prostate-metadata.csv")
    print(f"     (the left-panel filters are positionally aligned to the corpus).")
    print(f"  3. Restart the app.")


if __name__ == "__main__":
    main()
