from __future__ import annotations

import numpy as np
import sys
import os
from typing import NamedTuple
import pandas as pd
from rapidfuzz import fuzz
import ast
import json
import re

# Make the project root importable regardless of the process working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Single import of the graph store (importing it both as ``.graph_store`` and
# ``graph_store`` loaded the graph into memory twice)
from .graph_store import (G, consensus_dict, metadata_mapping, CURATED_TRIPLES, CURATED_VARIANT_CANCER,
                          VERIFICATION_AVAILABLE, VERIFIED_TRIPLES, GENE_ALIASES, CANCER_SUBTYPES,
                          EVIDENCE_AVAILABLE, association_papers, paper_details)

# === Adjustable thresholds (see config.py / .env.example to override) === #
TREATMENT_THRESHOLD_PERCENTILE = config.TREATMENT_THRESHOLD_PERCENTILE  # highlight top X% of treatment weights
CANCER_THRESHOLD_PERCENTILE = config.CANCER_THRESHOLD_PERCENTILE        # highlight top X% of cancer–variant weights

MATCHING_RATIO_THRESH = config.MATCHING_RATIO_THRESH  # fuzzy-match acceptance ratio (0-1)

EXCLUDED_TREATMENTS = {
    'chemotherapy', 'tyrosine kinase inhibitor', 'radiotherapy', 'hormone therapy',
    'adjuvant chemotherapy', 'immunotherapy', 'immune checkpoint inhibitor',
    'mrna vaccine', 'mtor inhibitor', 'radiation ionizing radiotherapy'
}

CANCER_SYNONYM_MAP = {
    "carcinoma": "cancer",
    "sarcoma": "cancer",
    "tumor": "cancer",
    "tumour": "cancer",
    "neoplasm": "cancer",
    "malignancy": "cancer",
    "lesion": "cancer"
}

def compute_cancer_alias_map():
    # Load and normalize the synonym dataframe
    CIVIC_cancer_synonyms_df = pd.read_csv(config.CANCER_SYNONYMS_PATH)
    df_syn = CIVIC_cancer_synonyms_df.copy()
    df_syn["name"] = df_syn["name"].str.strip().str.lower()

    def safe_parse_synonyms(x):
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return [s.strip().lower() for s in parsed if isinstance(s, str)]
            except Exception as e:
                print(f"Warning: could not parse synonym entry '{x}': {e}")
        return []

    df_syn["synonyms"] = df_syn["synonyms"].apply(safe_parse_synonyms)

    # Build the alias map: synonym → canonical cancer name
    cancer_alias_map = {}
    for _, row in df_syn.iterrows():
        canonical = row["name"]
        for synonym in row["synonyms"]:
            cancer_alias_map[synonym] = canonical

        #get cancer synonyms in right format: 
    synonyms = CIVIC_cancer_synonyms_df["synonyms"].to_numpy()
    syn_series = pd.Series(synonyms)        
    syn_series = syn_series.dropna()
    syn_series = syn_series.map(
        lambda x:                                            # x is one cell
            ast.literal_eval(x) if isinstance(x, str) and x.lstrip().startswith('[')
            else [x]                                         # keep singletons like '0'
    )
    synonyms_flat = syn_series.explode().astype(str)         
    synonyms_array = synonyms_flat.to_numpy()                
    synonyms_array_unique = pd.unique(synonyms_array)

    return cancer_alias_map, synonyms_array_unique

#TODO define as object with methods
CANCER_ALIAS_MAP, SYNONYM_ARRAY = compute_cancer_alias_map()

def normalise_cancer_name(raw: str) -> str:
    """
    → Lower-case, trim, swap synonyms for 'cancer',
      then apply the alias map if present.
    """
    # basic cleanup
    name = raw.strip().lower()

    direct_alias = CANCER_ALIAS_MAP.get(name)
    if direct_alias is not None:
        return direct_alias
    # A name that already is a cancer node is used as is (otherwise e.g. "histiocytic
    # sarcoma" became "histiocytic cancer" and could not be found)
    if name in G and G.nodes[name].get("category") == "Cancer":
        return name

    # synonym-to-'cancer' pass  (word boundaries ⇒ no 'adenocarcinoma' → 'adenocancer')
    for syn, replacement in CANCER_SYNONYM_MAP.items():
        pattern = rf"\b{re.escape(syn)}\b"
        name = re.sub(pattern, replacement, name)
    name = " ".join(name.split())

    return CANCER_ALIAS_MAP.get(name, name)

# Curated predictions shown in the sensitive / resistant lists
CURATED_SENSITIVE = {"Sensitive"}
CURATED_RESISTANT = {"Resistant", "Reduced sensitivity", "No response"}
# Verified literature association whose papers disagree on the direction (no 60% majority)
MIXED = "Mixed"
MAX_CURATED_SHOWN = 10
MAX_LITERATURE_SHOWN = 6

# Evidence tiers, best first. "literature" is a verified association when the
# artifacts include verification, or a plain co-occurrence for older artifacts.
TIER_CURATED, TIER_LITERATURE, TIER_COOCCURRENCE = "curated", "literature", "cooccurrence"
_TIER_RANK = {TIER_CURATED: 0, TIER_LITERATURE: 1, TIER_COOCCURRENCE: 2}


class Association(NamedTuple):
    """One result row: a treatment or cancer type with its supporting evidence.

    ``weight``: study-weighted number of papers verified to report the association
    (co-occurrence weight for the co-occurrence tier and for older artifacts).
    ``curated``: best curated (CIViC) record or ``None``. ``tier``: curated,
    literature or cooccurrence. ``evidence``: papers / example quote for verified
    associations.
    """
    name: str
    weight: float
    curated: dict | None = None
    tier: str = TIER_LITERATURE
    evidence: dict | None = None


def _curated_info(records):
    """Summarise the best curated record for display, or ``None``."""
    if not records:
        return None
    best = records[0]
    ids = list(dict.fromkeys(r["record_id"] for r in records))
    return {
        "source": "CIViC",
        "level": best["level"],
        "prediction": best["prediction"],
        "evidence_type": best["evidence_type"],
        "score": float(best["score"]),
        "ids": ids,
        "pmids": [str(r["pmid"]) for r in records if str(r.get("pmid", "")).strip()],
    }


def _label_counts(record):
    try:
        return {k: int(v) for k, v in json.loads(record.get("label_counts") or "{}").items()}
    except (TypeError, ValueError):
        return {}


def _verified_info(record):
    if not record:
        return None
    counts = _label_counts(record)
    return {
        "relation": record["relation"],
        "papers": int(record["papers"]),
        "quote": record.get("example_quote", ""),
        "paper_id": str(record.get("example_paper_id", "")),
        "via": record.get("via"),
        # Papers reporting response vs. resistance / no response (shown for "Mixed" evidence)
        "sensitive_papers": sum(counts.get(k, 0) for k in CURATED_SENSITIVE),
        "against_papers": sum(counts.get(k, 0) for k in CURATED_RESISTANT),
        "mixed": record["relation"] == MIXED,
        # The association the record is for (may be a codon class / subtype, see "via"),
        # and whether its supporting papers can be listed
        "variant": record.get("variant", ""), "cancer": record.get("cancer", ""),
        "treatment": record.get("treatment", ""), "link": EVIDENCE_AVAILABLE,
    }


def _ranked(items, max_curated=MAX_CURATED_SHOWN, max_literature=MAX_LITERATURE_SHOWN):
    """Tiered ordering: curated (by curated score, then weight), then verified
    literature, then co-occurrence (each by weight). ``None`` = no limit."""
    curated = sorted((a for a in items if a.tier == TIER_CURATED),
                     key=lambda a: (a.curated["score"], a.weight), reverse=True)
    rest = sorted((a for a in items if a.tier != TIER_CURATED), key=lambda a: (-_TIER_RANK[a.tier], a.weight),
                  reverse=True)
    return curated[:max_curated] + rest[:max_literature]


def _edge_weight(a, b, attr="weight"):
    return G[a][b].get(attr, 0) if G.has_edge(a, b) else 0


def _cooccurrence_weight(a, b):
    # Older artifacts have no separate co-occurrence weight: their weight is co-occurrence
    return _edge_weight(a, b, "cooccurrence_weight" if VERIFICATION_AVAILABLE else "weight")


_LOWERCASE_NODES = {key.lower(): key for key in G.nodes}
# Short forms of fusion partners (as in the pipeline)
_FUSION_PARTNER_ALIASES = {"abl": "abl1", "mll": "kmt2a", "pdgfr": "pdgfrb", "ntrk": "ntrk1"}
_FUSION_INPUT_RE = re.compile(r"fusion|rearrang|translocation|::", re.IGNORECASE)
_PARTNER_RE = re.compile(r"([A-Za-z][A-Za-z0-9]{1,9})\s*(?:::|-|/)\s*([A-Za-z][A-Za-z0-9]{1,9})")
_AMPLIFICATION_INPUTS = {"amp", "amplification", "amplified", "amplifications", "copynumbergain", "gain"}
_ITD_INPUTS = {"itd", "internaltandemduplication"}
_CODON_INPUT_RE = re.compile(r"^p?\.?([a-z])(\d{1,4})(?:x|mut|mutant|mutation|mutations|any)?$")


_GRAPH_GENES = {n.rpartition("_")[2].upper() for n, d in G.nodes(data=True) if d.get("category") == "Variant"}


def resolve_gene_symbol(gene_name):
    """Gene symbol as used in the graph for a gene name or alias (MEK1 -> MAP2K1,
    HER2 -> ERBB2, H3F3A -> H3-3A); unknown names are returned unchanged."""
    gene = (gene_name or "").strip()
    written = re.sub(r"\s+", "", gene.upper())
    if written in _GRAPH_GENES:
        return written
    compact = re.sub(r"[-_]", "", written)
    return GENE_ALIASES.get(written) or GENE_ALIASES.get(compact) or gene


def _resolve_variant(gene_name, variant_name):
    """Graph node for a gene/variant search, including alteration classes.

    Besides exact variants (``V600E`` + ``BRAF`` -> ``v600e_BRAF``) this accepts
    fusions (``ALK`` + ``fusion``, ``EML4-ALK``), amplifications (``ERBB2`` +
    ``amplification``), ITD (``FLT3`` + ``ITD``) and hotspot codons (``BRAF`` + ``V600``).
    Gene aliases are resolved to the graph's symbols (``MEK1`` + ``K57N`` -> ``k57n_MAP2K1``).
    """
    written_gene = (gene_name or "").strip()   # as typed, for fusion partners (EML4-ALK)
    gene = resolve_gene_symbol(written_gene)
    variant = (variant_name or "").strip()
    exact = _LOWERCASE_NODES.get(f"{variant}_{gene}".lower())
    if exact:
        return exact
    compact = re.sub(r"[\s_\-.()/]", "", variant.lower())
    if _FUSION_INPUT_RE.search(variant) or _FUSION_INPUT_RE.search(written_gene) or _PARTNER_RE.search(written_gene):
        partners = _PARTNER_RE.search(f"{written_gene} {variant}")
        candidates = ([partners.group(2), partners.group(1)] if partners else []) + [gene]
        for candidate in candidates:
            candidate = resolve_gene_symbol(candidate).lower()
            node = _LOWERCASE_NODES.get(f"fusion_{_FUSION_PARTNER_ALIASES.get(candidate, candidate)}")
            if node:
                return node
    if compact in _AMPLIFICATION_INPUTS:
        return _LOWERCASE_NODES.get(f"amplification_{gene}".lower())
    if compact in _ITD_INPUTS:
        return _LOWERCASE_NODES.get(f"itd_{gene}".lower())
    m = _CODON_INPUT_RE.match(compact)
    if m:
        return _LOWERCASE_NODES.get(f"{m.group(1)}{m.group(2)}_{gene}".lower())
    return None


def _codon_parent(node):
    """Hotspot-codon class node of a specific variant (v600e_BRAF -> v600_BRAF), if in the graph."""
    variant, _, gene = node.rpartition("_")
    m = re.match(r"^([a-z])(\d{1,4})[a-z*]", variant)
    if not m or G.nodes[node].get("variant_class", "specific") != "specific":
        return None
    parent = _LOWERCASE_NODES.get(f"{m.group(1)}{m.group(2)}_{gene}".lower())
    return parent if parent and G.nodes[parent].get("variant_class") == "codon" else None


def cancer_scope(cancer):
    """The cancer node and its OncoTree subtypes in the graph, the node itself first."""
    return [cancer] + CANCER_SUBTYPES.get(cancer, [])


def display_variant(node):
    """v600e_BRAF -> BRAF V600E, fusion_ALK -> ALK fusion"""
    variant, _, gene = node.rpartition("_")
    if variant in {"fusion", "amplification"}:
        return f"{gene} {variant}"
    return f"{gene} {variant.upper()}"


def _pair_association(a, b, curated_records, show_unverified=True):
    """Association for a variant-cancer pair under the tier rules, or None."""
    curated = _curated_info(curated_records)
    verified = _edge_weight(a, b) if VERIFICATION_AVAILABLE else 0
    variant, cancer = (a, b) if G.nodes[a].get("category") == "Variant" else (b, a)
    evidence = _verified_info(VERIFIED_TRIPLES.get((variant.lower(), cancer.lower(), ""))) if verified else None
    if curated:
        return Association(b, verified or _cooccurrence_weight(a, b), curated, TIER_CURATED, evidence)
    if not VERIFICATION_AVAILABLE:
        weight = _edge_weight(a, b)
        return Association(b, weight, tier=TIER_LITERATURE) if weight > 0 else None
    if verified > 0:
        return Association(b, verified, tier=TIER_LITERATURE, evidence=evidence)
    co = _cooccurrence_weight(a, b)
    if show_unverified and co > 0:
        return Association(b, co, tier=TIER_COOCCURRENCE)
    return None


RELATION_ORDER = ["Sensitive", "Resistant", "No response", "Prognostic", "Diagnostic", "Reported"]


def association_evidence(variant, cancer, treatment="", relation=None, limit=100):
    """The papers verified to report an association, for the evidence page.

    Returns ``None`` if the association has no verified evidence, else a dict with the
    association summary, the per-relation paper counts and up to ``limit`` papers
    (optionally only one ``relation``): title, year, authors, the paper's relation, its
    supporting quote and the abstract. Papers are ordered by relation, then newest first.
    """
    key = (variant.lower(), cancer.lower(), (treatment or "").lower())
    records = association_papers(*key)
    if not records:
        return None
    summary = VERIFIED_TRIPLES.get(key) or {}
    papers = paper_details({r["paper_id"] for r in records})
    counts = {}
    for rec in records:
        counts[rec["relation"]] = counts.get(rec["relation"], 0) + 1
    selected = [r for r in records if relation in (None, "", r["relation"])]
    order = {rel: i for i, rel in enumerate(RELATION_ORDER)}
    selected.sort(key=lambda r: (order.get(r["relation"], 99),
                                 -int(papers.get(r["paper_id"], {}).get("year") or 0), r["paper_id"]))
    rows = []
    for rec in selected[:limit]:
        p = papers.get(rec["paper_id"], {})
        rows.append({"paper_id": rec["paper_id"], "url": f"https://openalex.org/W{rec['paper_id']}",
                     "title": p.get("title") or f"OpenAlex W{rec['paper_id']}", "year": p.get("year", ""),
                     "authors": p.get("authors", ""), "abstract": p.get("abstract", ""),
                     "relation": rec["relation"], "quote": rec["quote"], "study_design": rec.get("study_design", "")})
    return {
        "variant": display_variant(records[0]["variant"]), "cancer": records[0]["cancer"],
        "treatment": records[0]["treatment"], "relation": summary.get("relation", ""),
        "papers": len(records), "counts": sorted(counts.items(), key=lambda kv: order.get(kv[0], 99)),
        "selected": len(selected), "shown": rows,
    }


#TODO create a Graph class that contains the following methods
def check_variant_in_graph(user_gene_name, user_variant_name):
    """Check if the variant exists in the graph."""
    return _resolve_variant(user_gene_name, user_variant_name) in G.nodes

def check_cancer_in_graph(user_cancer_name):
    """Check if the cancer exists in the graph."""
    cancer_of_interest = normalise_cancer_name(user_cancer_name)
    return cancer_of_interest in G.nodes


def get_associated_cancer_types_from_variant(gene_name, variant_name):
    """Cancer types associated with the variant: curated, verified, then co-occurrence."""
    v = _resolve_variant(gene_name, variant_name)
    items = [
        a for n in G.neighbors(v) if G.nodes[n]['category'] == 'Cancer'
        for a in [_pair_association(v, n, CURATED_VARIANT_CANCER.get((v.lower(), n.lower())))] if a
    ]
    return [a.name.capitalize() for a in _ranked(items, max_curated=None, max_literature=None)]

def get_associated_variants_from_cancer_type(cancer_name):
    """Variants associated with the cancer type: curated, verified, then co-occurrence."""
    c = normalise_cancer_name(cancer_name)
    best = {}   # variant -> association for the cancer itself, else for its first subtype with one
    for cancer in cancer_scope(c):
        for n in G.neighbors(cancer):
            if G.nodes[n]['category'] != 'Variant' or n in best:
                continue
            a = _pair_association(cancer, n, CURATED_VARIANT_CANCER.get((n.lower(), cancer.lower())))
            if a:
                best[n] = a._replace(name=n)
    items = list(best.values())
    return [a.name.capitalize() for a in _ranked(items, max_curated=None, max_literature=None)]

def compute_associations(gene_name, variant_name, cancer_name, show_unverified=False):
    """Sensitive / resistant treatments and other cancer types for a variant in a cancer type.

    Tiers, in this order:
    1. curated (CIViC) records for this exact variant, cancer type and treatment;
    2. literature associations verified against the abstracts for the exact
       triple, labelled by the verifier's consensus relation;
    3. with ``show_unverified``: co-occurrence only, labelled by the variant-
       treatment consensus.
    For older artifacts without verification, tier 2 is the previous
    co-occurrence behaviour. Percentiles are computed over tier 2 weights.
    """
    variant_of_interest = _resolve_variant(gene_name, variant_name)
    cancer_of_interest = normalise_cancer_name(cancer_name)
    v_key = variant_of_interest.lower()
    # The cancer type and its OncoTree subtypes (evidence for a subtype is marked "via")
    scope = cancer_scope(cancer_of_interest)

    # === Step 1: Cancer-associated treatments ===
    treatments = sorted({
        n for c in scope for n in G.neighbors(c)
        if G.nodes[n]['category'] == 'Treatment'
        and n.lower() not in EXCLUDED_TREATMENTS
    })

    def cancer_cooccurrence(t):
        return max(_cooccurrence_weight(c, t) for c in scope)

    def cancer_edge(t):
        return max(_edge_weight(c, t) for c in scope)

    # === Step 2: Variant + cancer associations ===
    sensitive, resistant = [], []

    def add(prediction, association):
        if prediction == MIXED and association.evidence:
            # Listed on the side most papers support (both on a tie), marked as mixed evidence
            s, r = association.evidence["sensitive_papers"], association.evidence["against_papers"]
            if s >= r:
                sensitive.append(association)
            if r >= s:
                resistant.append(association)
        elif prediction in CURATED_SENSITIVE:
            sensitive.append(association)
        elif prediction in CURATED_RESISTANT:
            resistant.append(association)

    # A specific hotspot variant (e.g. V600E) also draws on its codon class (V600), marked "via"
    parent = _codon_parent(variant_of_interest)
    parent_key = parent.lower() if parent else None

    variants_in_scope = [(v_key, None)] + ([(parent_key, f"{display_variant(parent)} (codon level)")]
                                           if parent_key else [])

    def lookup(table, treatment, strength):
        """Best record for the variant (or its codon class) in the cancer type (or a
        subtype), preferring the exact cancer, then the exact variant, then ``strength``;
        returns (record, via) where via describes a codon class / subtype it came from."""
        hits = []
        for ci, cancer in enumerate(scope):
            for vi, (variant, variant_via) in enumerate(variants_in_scope):
                records = table.get((variant, cancer.lower(), treatment.lower()))
                if records:
                    via = [x for x in (variant_via, f"{cancer} (subtype)" if ci else None) if x]
                    hits.append(((ci > 0, vi), -strength(records), records, ", ".join(via) or None))
        if not hits:
            return None, None
        _, _, records, via = min(hits, key=lambda h: h[:2])
        return records, via

    for t in treatments:
        verified, verified_via = lookup(VERIFIED_TRIPLES, t, lambda r: float(r["weight"]))
        cooccurrence = cancer_cooccurrence(t) + _cooccurrence_weight(variant_of_interest, t)
        curated_records, curated_via = lookup(CURATED_TRIPLES, t, lambda r: float(r[0]["score"]))
        curated = _curated_info(curated_records)
        if curated and curated_via:
            curated["via"] = curated_via
        if verified and verified_via:
            verified = {**verified, "via": verified_via}
        if curated:
            weight = float(verified["weight"]) if verified else cooccurrence
            add(curated["prediction"], Association(t, weight, curated, TIER_CURATED, _verified_info(verified)))
        elif not VERIFICATION_AVAILABLE:
            # Older artifacts: co-occurrence edges, direction from the LLM consensus
            if cancer_edge(t) > 0 and _edge_weight(variant_of_interest, t) > 0:
                add(consensus_dict.get(f"{variant_of_interest} + {t}".lower()), Association(t, cooccurrence))
        elif verified:
            add(verified["relation"], Association(t, float(verified["weight"]), tier=TIER_LITERATURE,
                                                  evidence=_verified_info(verified)))
        elif show_unverified and cancer_cooccurrence(t) > 0 \
                and _cooccurrence_weight(variant_of_interest, t) > 0:
            add(consensus_dict.get(f"{variant_of_interest} + {t}".lower()),
                Association(t, cooccurrence, tier=TIER_COOCCURRENCE))

    top_sens = _ranked(sensitive)
    top_res = _ranked(resistant)
    sens_w = [a.weight for a in sensitive if a.tier == TIER_LITERATURE]
    res_w = [a.weight for a in resistant if a.tier == TIER_LITERATURE]
    sens_pct = np.percentile(sens_w, TREATMENT_THRESHOLD_PERCENTILE) if sens_w else 0
    res_pct = np.percentile(res_w, TREATMENT_THRESHOLD_PERCENTILE) if res_w else 0

    # === Step 3: Other cancers for variant ===
    other_cancers = []
    for c in G.neighbors(variant_of_interest):
        if G.nodes[c]['category'] != 'Cancer' or c in scope:
            continue
        a = _pair_association(variant_of_interest, c, CURATED_VARIANT_CANCER.get((v_key, c.lower())),
                              show_unverified=show_unverified)
        if a:
            other_cancers.append(a)
    top_var_c = [a._replace(name=a.name.capitalize()) for a in _ranked(other_cancers)]
    vc_w = [a.weight for a in other_cancers if a.tier == TIER_LITERATURE]
    cancer_pct = np.percentile(vc_w, CANCER_THRESHOLD_PERCENTILE) if vc_w else 0

    # get gene and variant name from the variant of interest
    variant_name, gene_name = variant_of_interest.rsplit("_", 1)

    return cancer_of_interest, top_sens, top_res, top_var_c, sens_pct, res_pct, cancer_pct, gene_name, variant_name


def autosuggest_item(user_input: str, item_type: str, corresponding_gene = None, FUZZY_MATCH = False) -> list:
    """
    Simple contains-filter.
    - Case-insensitive
    - Suggestions whose *start* matches rank higher
    - Return unique, presorted list
    """
    out = []
    alias_hits = []
    q = user_input.strip().lower()

    if item_type == 'Gene':
        entities = metadata_mapping[metadata_mapping['Category'] == "Variant"]['Entity'] #no genes in the mapping, but only concatenated varinates and genes
    else:
        entities = metadata_mapping[metadata_mapping['Category'] == item_type]['Entity']
    
    if item_type in ['Variant', 'Gene']:
        parsed = [ent.split("_", 1) for ent in entities if "_" in ent]
        if item_type == 'Variant':
            corresponding_gene = resolve_gene_symbol(corresponding_gene)
            if any(g.lower() == corresponding_gene.lower() for _, g in parsed):
                filtered = [(var, g) for var, g in parsed if g.lower() == corresponding_gene.lower()]
            else:
                filtered = parsed
            out = pd.Series([var.upper() for var, _ in filtered])
        else:
            # Genes whose alias matches the input are suggested by their graph symbol (MEK1 -> MAP2K1)
            alias_hits = sorted({s for a, s in GENE_ALIASES.items() if a.lower().startswith(q)}) if q else []
            out = pd.Series([g for _, g in parsed])
    else:
        out = pd.concat([entities, pd.Series(SYNONYM_ARRAY, index=range(len(entities), len(entities)+len(SYNONYM_ARRAY)))],
                ignore_index=True
        ).map(lambda x: str(x).title())

    matching_names = []
    if FUZZY_MATCH:
        for name in entities:
            fuzz_ratio = fuzz.ratio(user_input.lower(), name.lower())
            if fuzz_ratio > MATCHING_RATIO_THRESH * 100 and name.lower() not in out.str.lower():
                matching_names.append(name)

    # two-pass ranking with duplicates removed
    starts = out[out.str.lower().str.startswith(q)]
    contains = out[out.str.lower().str.contains(q) & ~out.isin(starts)]
    suggestions = list(dict.fromkeys(list(starts) + list(contains) + alias_hits + list(matching_names)))
    
    return suggestions