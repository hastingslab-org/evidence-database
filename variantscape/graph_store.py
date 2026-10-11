import functools
import os
import sqlite3
import sys
from collections import defaultdict

import networkx as nx
import pandas as pd

# Make the project root importable regardless of the process working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

 # Load datasets needed for network query (variantscape)
print("[variantscape] Loading the network graph...")
G = nx.read_gml(str(config.VARIANTSCAPE_GRAPH_PATH))
print("[variantscape] Loading the consensus file...")
df_consensus = pd.read_csv(config.VARIANTSCAPE_CONSENSUS_PATH)
# === Prepare consensus lookup ===
df_consensus["Variant_Treatment_Pair"] = (
df_consensus["Variant_Treatment_Pair"]
.str.strip()
.str.lower()
)
# Curated-only pairs have no literature (LLM) consensus
_literature_consensus = df_consensus.dropna(subset=["Resolved_Prediction"])
consensus_dict = dict(
    zip(_literature_consensus["Variant_Treatment_Pair"], _literature_consensus["Resolved_Prediction"])
)
metadata_mapping = pd.read_csv(config.VARIANTSCAPE_METADATA_MAPPING_PATH, low_memory=False)

# === Curated (CIViC) associations, written by the Variantscape pipeline ===
# Optional: older artifact sets have no curated associations.
# CURATED_TRIPLES[(variant, cancer, treatment)] and CURATED_VARIANT_CANCER[(variant, cancer)]
# map lower-cased graph node names to curated records, best (highest score) first.
CURATED_TRIPLES: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
CURATED_VARIANT_CANCER: dict[tuple[str, str], list[dict]] = defaultdict(list)
if os.path.exists(config.VARIANTSCAPE_CURATED_PATH):
    print("[variantscape] Loading curated associations...")
    _curated = pd.read_csv(config.VARIANTSCAPE_CURATED_PATH, keep_default_na=False)
    for _rec in _curated.sort_values("score", ascending=False).to_dict("records"):
        _v, _c, _t = (str(_rec[k]).strip().lower() for k in ("variant", "cancer", "treatment"))
        CURATED_VARIANT_CANCER[(_v, _c)].append(_rec)
        if _t:
            CURATED_TRIPLES[(_v, _c, _t)].append(_rec)
else:
    print("[variantscape] No curated associations file; showing literature associations only.")

# === Verified literature associations (pipeline's abstract-level LLM verifier) ===
# VERIFIED_TRIPLES[(variant, cancer, treatment)] -> record (relation, weight, papers,
# example quote) for associations an abstract was verified to report. Without this
# file (older artifacts) the graph weights are plain co-occurrence counts.
VERIFICATION_AVAILABLE = os.path.exists(config.VARIANTSCAPE_VERIFIED_PATH)
VERIFIED_TRIPLES: dict[tuple[str, str, str], dict] = {}
if VERIFICATION_AVAILABLE:
    print("[variantscape] Loading verified associations...")
    _verified = pd.read_csv(config.VARIANTSCAPE_VERIFIED_PATH, keep_default_na=False, dtype={"paper_ids": str,
                                                                                              "example_paper_id": str})
    for _rec in _verified.to_dict("records"):
        _key = tuple(str(_rec[k]).strip().lower() for k in ("variant", "cancer", "treatment"))
        VERIFIED_TRIPLES[_key] = _rec

# === Gene aliases (HGNC aliases / previous symbols of the graph's genes) ===
# GENE_ALIASES[ALIAS] -> symbol used in the graph, so that searches by an alias work.
# Optional: older artifact sets have no alias file.
GENE_ALIASES: dict[str, str] = {}
if os.path.exists(config.VARIANTSCAPE_GENE_ALIASES_PATH):
    _aliases = pd.read_csv(config.VARIANTSCAPE_GENE_ALIASES_PATH, keep_default_na=False, dtype=str)
    GENE_ALIASES = {a.strip().upper(): s.strip() for a, s in zip(_aliases["alias"], _aliases["symbol"]) if a and s}

# === Cancer type hierarchy (OncoTree lineage of the cancer nodes) ===
# CANCER_SUBTYPES[node] -> cancer nodes below it in OncoTree (lung cancer -> non-small
# cell lung cancer, lung adenocarcinoma, ...), so that searches for a broader type
# include the evidence for its subtypes. Empty for older artifacts without OncoTree.
CANCER_SUBTYPES: dict[str, list[str]] = defaultdict(list)
_node_of_code = {d["oncotree_code"]: n for n, d in G.nodes(data=True)
                 if d.get("category") == "Cancer" and d.get("oncotree_code")}
for _node, _data in G.nodes(data=True):
    if _data.get("category") != "Cancer" or not _data.get("oncotree_lineage"):
        continue
    for _code in _data["oncotree_lineage"].split(";")[:-1]:
        if _code in _node_of_code and _node_of_code[_code] != _node:
            CANCER_SUBTYPES[_node_of_code[_code]].append(_node)
for _subtypes in CANCER_SUBTYPES.values():
    _subtypes.sort()

# === Per-paper evidence of verified associations and the papers behind them ===
# Large (abstracts), and the app runs several worker processes: on first use each CSV is
# converted once into an indexed SQLite file next to it and queried per request; if that
# directory is not writable the CSV is loaded into memory instead.
EVIDENCE_AVAILABLE = os.path.exists(config.VARIANTSCAPE_EVIDENCE_PATH)


def _sqlite_copy(csv_path, table, key_columns):
    """Path of an up-to-date SQLite copy of ``csv_path`` (lower-cased ``key_columns``
    indexed as ``k_<column>``), or None if it cannot be written."""
    csv_path = str(csv_path)
    db_path = os.path.splitext(csv_path)[0] + ".sqlite"
    if os.path.exists(db_path) and os.path.getmtime(db_path) >= os.path.getmtime(csv_path):
        return db_path
    tmp = f"{db_path}.{os.getpid()}.tmp"
    try:
        print(f"[variantscape] Indexing {os.path.basename(csv_path)} ...")
        with sqlite3.connect(tmp) as conn:
            for chunk in pd.read_csv(csv_path, keep_default_na=False, dtype=str, chunksize=20000):
                for col in key_columns:
                    chunk[f"k_{col}"] = chunk[col].str.lower()
                chunk.to_sql(table, conn, if_exists="append", index=False)
            conn.execute(f"CREATE INDEX idx_key ON {table} ({', '.join(f'k_{c}' for c in key_columns)})")
        os.replace(tmp, db_path)   # atomic: concurrent workers never see a partial file
        return db_path
    except OSError as exc:
        print(f"[variantscape] Cannot write {db_path} ({exc}); keeping {os.path.basename(csv_path)} in memory")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


@functools.lru_cache(maxsize=1)
def _evidence_db():
    return _sqlite_copy(config.VARIANTSCAPE_EVIDENCE_PATH, "evidence", ["variant", "cancer", "treatment"]) \
        if EVIDENCE_AVAILABLE else None


@functools.lru_cache(maxsize=1)
def _papers_db():
    path = config.VARIANTSCAPE_PAPERS_PATH
    return _sqlite_copy(path, "papers", ["paper_id"]) if os.path.exists(path) else None


@functools.lru_cache(maxsize=1)
def _evidence_in_memory():
    df = pd.read_csv(config.VARIANTSCAPE_EVIDENCE_PATH, keep_default_na=False, dtype=str)
    index = defaultdict(list)
    for rec in df.to_dict("records"):
        index[(rec["variant"].lower(), rec["cancer"].lower(), rec["treatment"].lower())].append(rec)
    return index


@functools.lru_cache(maxsize=1)
def _papers_in_memory():
    df = pd.read_csv(config.VARIANTSCAPE_PAPERS_PATH, keep_default_na=False, dtype=str)
    return {rec["paper_id"]: rec for rec in df.to_dict("records")}


def _query(db_path, sql, params):
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


def association_papers(variant: str, cancer: str, treatment: str) -> list[dict]:
    """One record per paper verified to report the association (relation, quote, ...)."""
    if not EVIDENCE_AVAILABLE:
        return []
    key = (variant.lower(), cancer.lower(), (treatment or "").lower())
    db = _evidence_db()
    if db is None:
        return list(_evidence_in_memory().get(key, []))
    return _query(db, "SELECT * FROM evidence WHERE k_variant=? AND k_cancer=? AND k_treatment=?", key)


def paper_details(paper_ids) -> dict[str, dict]:
    """OpenAlex id without the W -> title, year, authors, abstract (papers not found are left out)."""
    paper_ids = [str(p) for p in paper_ids]
    if not paper_ids or not os.path.exists(config.VARIANTSCAPE_PAPERS_PATH):
        return {}
    db = _papers_db()
    if db is None:
        papers = _papers_in_memory()
        return {p: papers[p] for p in paper_ids if p in papers}
    out = {}
    for i in range(0, len(paper_ids), 500):
        chunk = paper_ids[i:i + 500]
        for rec in _query(db, f"SELECT * FROM papers WHERE k_paper_id IN ({','.join('?' * len(chunk))})", chunk):
            out[rec["paper_id"]] = rec
    return out
