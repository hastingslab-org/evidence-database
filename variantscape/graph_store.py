import os
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
