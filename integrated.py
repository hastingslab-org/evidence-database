"""Personalised Treatment Query -- front-page integration layer.

Ties the three EvidenceDB sub-systems together for a single clinician question:

1. ``extract_profile`` turns the free-text patient description + question into a
   structured profile via one non-streaming LLM call (DeepInfra, same model as
   the rest of the app).
2. When the profile mentions a gene or variant, ``gather_genomic_context`` and
   ``gather_variantscape_context`` look it up in GenomicsDB (CIViC) and
   Variantscape respectively. Both degrade gracefully -- variant -> gene -> skip
   -- and never raise into the request.
3. ``format_findings_for_prompt`` renders those findings as a provenance-labelled
   text block that :func:`llm.call_llm_stream` appends to the synthesis prompt,
   and ``build_retrieval_query`` enriches the literature-retrieval query with the
   gene / variant / treatment terms.

The actual literature retrieval, guideline retrieval, answer streaming and
``qa_data`` caching are all reused unchanged from the LiteratureDB pipeline in
``app.py`` / ``llm.py`` / ``guidelines.py``.
"""

import json
import os
import sqlite3

import config
import llm
from genomics.genomics_data import (
    check_variant_in_database,
    get_item_by_name,
    get_items_by_name_fuzzy,
    get_items_from_ids,
)
from variantscape.variantscape import (
    EXCLUDED_TREATMENTS,
    check_cancer_in_graph,
    check_variant_in_graph,
    compute_associations,
    get_associated_cancer_types_from_variant,
)
from variantscape.graph_store import G

_DB_PATH = str(config.SQLITE_DB_PATH)
_MAX_DESC = 600  # CIViC descriptions can be very long; trim for the prompt.

# Tables the GenomicsDB helpers read. ``database.db`` is git-ignored and rebuilt
# from CIViC by db_initializations/update_genomics_db.py; a fresh deployment only
# has ``qa_data`` until that script runs, and every gene/variant lookup then
# raises "no such table: genes".
_GENOMICS_TABLES = ("genes", "variants", "diseases", "molecular_profiles")


def genomics_db_status() -> dict:
    """Report whether the GenomicsDB (CIViC) tables are queryable.

    Returned as ``genomic["status"]`` from :func:`gather_genomic_context` and
    echoed in the /integrated_answer response, so a deployment that has not run
    the genomics rebuild is visible in the browser (and at startup) instead of
    silently dropping GenomicsDB from the answer.
    """
    status = {
        "path": _DB_PATH,
        "exists": os.path.exists(_DB_PATH),
        "missing_tables": list(_GENOMICS_TABLES),
        "ok": False,
        "error": None,
    }
    try:
        con = sqlite3.connect(f"file:{_DB_PATH}?mode=ro", uri=True)
        try:
            present = {
                row[0]
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            con.close()
        status["missing_tables"] = [t for t in _GENOMICS_TABLES if t not in present]
        status["ok"] = not status["missing_tables"]
    except Exception as e:  # pragma: no cover - defensive
        status["error"] = str(e)
    return status


# --------------------------------------------------------------------------- #
# 1. Structured extraction
# --------------------------------------------------------------------------- #
def _clean_str(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        s = _clean_str(item)
        if s and s.lower() not in {v.lower() for v in out}:
            out.append(s)
    return out


def _clean_variants(value, genes) -> list[dict]:
    """Normalise the ``variants`` array to ``[{"gene": ..., "change": ...}]``."""
    out = []
    if not isinstance(value, list):
        value = []
    for item in value:
        gene = change = ""
        if isinstance(item, dict):
            gene = _clean_str(item.get("gene"))
            change = _clean_str(item.get("change") or item.get("variant") or item.get("name"))
        elif isinstance(item, str):
            parts = item.strip().split()
            if len(parts) >= 2:
                gene, change = parts[0], " ".join(parts[1:])
            else:
                change = item.strip()
        gene = gene.upper()
        if not gene and len(genes) == 1:
            gene = genes[0]
        if gene or change:
            key = (gene.lower(), change.lower())
            if key not in {(v["gene"].lower(), v["change"].lower()) for v in out}:
                out.append({"gene": gene, "change": change})
    return out


def extract_profile(patient_text: str, query: str) -> dict:
    """Return a structured patient profile.

    Shape::

        {
          "cancer_type": str, "stage": str,
          "genes": [str], "variants": [{"gene": str, "change": str}],
          "prior_treatments": [str], "performance_status": str, "other": [str],
          "patient_data": {display-key: value},   # feeds retrieval + guidelines
          "raw_text": str,
          "extraction_ok": bool,
        }

    On any LLM / parse failure the structured fields are empty and
    ``extraction_ok`` is False, so the caller still produces a
    literature-and-guidelines answer from the raw text.
    """
    patient_text = (patient_text or "").strip()
    query = (query or "").strip()

    user_msg = (
        f"PATIENT DESCRIPTION:\n{patient_text or '(none provided)'}\n\n"
        f"CLINICIAN QUESTION:\n{query or '(none provided)'}"
    )
    parsed = llm.call_llm_json(config.LLM_EXTRACTION_SYSTEM_MSG, user_msg)
    extraction_ok = bool(parsed)

    genes = _clean_list(parsed.get("genes"))
    genes = [g.upper() for g in genes]
    variants = _clean_variants(parsed.get("variants"), genes)
    # A variant implies its gene even if the model omitted it from `genes`.
    for v in variants:
        if v["gene"] and v["gene"] not in genes:
            genes.append(v["gene"])

    profile = {
        "cancer_type": _clean_str(parsed.get("cancer_type")),
        "stage": _clean_str(parsed.get("stage")),
        "genes": genes,
        "variants": variants,
        "prior_treatments": _clean_list(parsed.get("prior_treatments")),
        "performance_status": _clean_str(parsed.get("performance_status")),
        "other": _clean_list(parsed.get("other")),
        "raw_text": patient_text,
        "extraction_ok": extraction_ok,
    }
    profile["patient_data"] = _profile_to_patient_data(profile)
    return profile


def _profile_to_patient_data(profile: dict) -> dict:
    """Build the display / retrieval dict.

    The ``"Cancer type"`` key is spelled exactly as ``guidelines.get_relevant_
    guidelines`` expects so tumour-specific guideline filtering still fires.
    """
    pd: dict[str, str] = {}
    if profile["cancer_type"]:
        pd["Cancer type"] = profile["cancer_type"]
    if profile["stage"]:
        pd["Stage"] = profile["stage"]
    if profile["variants"]:
        pd["Genomic alterations"] = ", ".join(
            f"{v['gene']} {v['change']}".strip() for v in profile["variants"]
        )
    elif profile["genes"]:
        pd["Genes"] = ", ".join(profile["genes"])
    if profile["prior_treatments"]:
        pd["Prior treatments"] = ", ".join(profile["prior_treatments"])
    if profile["performance_status"]:
        pd["Performance status"] = profile["performance_status"]
    if profile["other"]:
        pd["Other"] = ", ".join(profile["other"])
    if not pd and profile["raw_text"]:
        pd["Patient description"] = profile["raw_text"]
    return pd


# --------------------------------------------------------------------------- #
# 2a. GenomicsDB (CIViC) lookup
# --------------------------------------------------------------------------- #
def _trim(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= _MAX_DESC else text[:_MAX_DESC].rstrip() + "…"


def _names_from_ids(raw, table) -> list[str]:
    try:
        return get_items_from_ids(raw, table, db_path=_DB_PATH)
    except Exception:
        return []


def gather_genomic_context(profile: dict) -> dict:
    """Look up each variant (then failing that, each gene) in GenomicsDB.

    Returns ``{"variants": [...], "genes": [...], "found": bool, "status": {...}}``
    where each entry is a small JSON-safe dict for the UI. When the GenomicsDB
    tables are not present (fresh deployment, rebuild not yet run) the lookups
    are skipped and ``status["ok"]`` is False.
    """
    status = genomics_db_status()
    out: dict = {"variants": [], "genes": [], "found": False, "status": status}
    if not status["ok"]:
        return out
    # Genes whose *variant* was resolved in GenomicsDB -- for these the
    # gene-level lookup is redundant. A gene mentioned only via an unresolved
    # variant (e.g. "germline BRCA2 pathogenic variant", no specific change)
    # must still get a gene-level lookup, so it is NOT added here.
    resolved_genes: set[str] = set()

    for v in profile.get("variants", []):
        gene, change = v.get("gene", ""), v.get("change", "")
        entry = {"gene": gene, "change": change, "in_genomicsdb": False}
        try:
            if change and check_variant_in_database(gene, change, db_path=_DB_PATH):
                rec = get_item_by_name(change, "variants", _DB_PATH) or {}
                entry.update(
                    in_genomicsdb=True,
                    description=_trim(rec.get("description", "")),
                    molecular_profiles=_names_from_ids(rec.get("molecular_profiles"), "molecular_profiles")[:6],
                    diseases=_names_from_ids(rec.get("diseases"), "diseases")[:6],
                )
                out["found"] = True
                if gene:
                    resolved_genes.add(gene.upper())
        except Exception as e:  # pragma: no cover - defensive
            print(f"[integrated] genomics variant lookup failed for {gene} {change}: {e}")
        out["variants"].append(entry)

    done_gene_lookups: set[str] = set()
    for gene in profile.get("genes", []):
        g = gene.upper()
        if g in resolved_genes or g in done_gene_lookups:
            continue
        done_gene_lookups.add(g)
        entry = {"gene": g, "in_genomicsdb": False}
        try:
            matches = get_items_by_name_fuzzy(g, "genes", _DB_PATH) or []
            exact = next((m for m in matches if m.get("name", "").upper() == g), None)
            rec = exact or (matches[0] if matches else None)
            if rec:
                entry.update(
                    in_genomicsdb=True,
                    name=rec.get("name", g),
                    description=_trim(rec.get("description", "")),
                    diseases=_names_from_ids(rec.get("diseases"), "diseases")[:6],
                )
                out["found"] = True
        except Exception as e:  # pragma: no cover - defensive
            print(f"[integrated] genomics gene lookup failed for {g}: {e}")
        out["genes"].append(entry)

    return out


# --------------------------------------------------------------------------- #
# 2b. Variantscape lookup
# --------------------------------------------------------------------------- #
def _assoc_list(pairs) -> list[dict]:
    out = []
    for item in pairs or []:
        try:
            name, weight = item
        except (TypeError, ValueError):
            name, weight = item, None
        out.append({"name": str(name), "weight": round(float(weight), 1) if weight is not None else None})
    return out


def _variantscape_gene_summary(gene: str) -> dict | None:
    """Aggregate literature-mined associations across every variant record for
    ``gene`` in the Variantscape graph.

    Used when the patient description names a gene but no specific variant that
    resolves to a graph node (common for tumour-suppressor mentions such as
    "germline BRCA2 pathogenic variant"). The weights are summed co-occurrence
    strengths across all of the gene's variants, so the direction (sensitising
    vs resistant) is *not* resolved -- the caller labels them accordingly.
    """
    suffix = "_" + gene.lower()
    nodes = [
        n for n in G.nodes
        if G.nodes[n].get("category") == "Variant" and n.lower().endswith(suffix)
    ]
    if not nodes:
        return None

    tw: dict[str, float] = {}
    cw: dict[str, float] = {}
    for vn in nodes:
        for nb in G.neighbors(vn):
            cat = G.nodes[nb].get("category")
            w = G[vn][nb].get("weight", 0) or 0
            if cat == "Treatment" and nb.lower() not in EXCLUDED_TREATMENTS:
                tw[nb] = tw.get(nb, 0) + w
            elif cat == "Cancer":
                cw[nb] = cw.get(nb, 0) + w

    def _top(d):
        return [
            {"name": str(k).title(), "weight": round(float(v), 1)}
            for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True)[:8]
        ]

    return {
        "gene": gene,
        "n_variants": len(nodes),
        "top_treatments": _top(tw),
        "top_cancer_types": _top(cw),
    }


def gather_variantscape_context(profile: dict) -> dict:
    """Query Variantscape for each variant, guarded by graph-membership checks.

    When a gene has no variant that resolves to a graph node, fall back to a
    gene-level aggregate (:func:`_variantscape_gene_summary`).
    """
    cancer = profile.get("cancer_type", "")
    cancer_in_graph = bool(cancer) and _safe(check_cancer_in_graph, cancer, default=False)

    out: dict = {
        "cancer_type": cancer,
        "cancer_in_graph": cancer_in_graph,
        "variants": [],
        "genes": [],
        "found": False,
    }

    for v in profile.get("variants", []):
        gene, change = v.get("gene", ""), v.get("change", "")
        entry = {"gene": gene, "change": change, "in_variantscape": False}
        if not (gene and change and _safe(check_variant_in_graph, gene, change, default=False)):
            out["variants"].append(entry)
            continue
        entry["in_variantscape"] = True
        out["found"] = True

        if cancer_in_graph:
            try:
                (_ct, top_sens, top_res, top_var_c, *_rest) = compute_associations(gene, change, cancer)
                entry["sensitising"] = _assoc_list(top_sens)
                entry["resistance"] = _assoc_list(top_res)
                entry["other_cancer_types"] = _assoc_list(top_var_c)
            except Exception as e:  # pragma: no cover - defensive
                print(f"[integrated] variantscape compute_associations failed for {gene} {change}: {e}")
        if "sensitising" not in entry:
            # No cancer context (or it failed) -> at least list associated tumours.
            cancers = _safe(get_associated_cancer_types_from_variant, gene, change, default=[]) or []
            entry["associated_cancer_types"] = [str(c) for c in cancers[:10]]
        out["variants"].append(entry)

    # Gene-level fallback for genes with no variant-level hit.
    matched = {v["gene"].upper() for v in out["variants"] if v.get("in_variantscape")}
    seen: set[str] = set()
    for gene in profile.get("genes", []):
        g = gene.upper()
        if g in matched or g in seen:
            continue
        seen.add(g)
        summary = _safe(_variantscape_gene_summary, g, default=None)
        if summary and (summary["top_treatments"] or summary["top_cancer_types"]):
            out["genes"].append(summary)
            out["found"] = True

    return out


def _safe(fn, *args, default=None):
    try:
        return fn(*args)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[integrated] {fn.__name__} failed: {e}")
        return default


# --------------------------------------------------------------------------- #
# 3. Prompt / retrieval assembly
# --------------------------------------------------------------------------- #
def _fmt_assoc(entries, limit=5) -> str:
    parts = []
    for e in entries[:limit]:
        w = f" (weight {e['weight']})" if e.get("weight") is not None else ""
        parts.append(f"{e['name']}{w}")
    return "; ".join(parts)


def format_findings_for_prompt(genomic: dict, variantscape: dict) -> str:
    """Render the GenomicsDB + Variantscape findings as one labelled text block."""
    lines: list[str] = []

    g_lines: list[str] = []
    for v in genomic.get("variants", []):
        if not v.get("in_genomicsdb"):
            continue
        bits = [f"{v['gene']} {v['change']}".strip() + ":"]
        if v.get("description"):
            bits.append(v["description"])
        if v.get("diseases"):
            bits.append("Diseases: " + ", ".join(v["diseases"]) + ".")
        if v.get("molecular_profiles"):
            bits.append("Molecular profiles: " + ", ".join(v["molecular_profiles"]) + ".")
        g_lines.append("- " + " ".join(bits))
    for g in genomic.get("genes", []):
        if not g.get("in_genomicsdb"):
            continue
        bits = [f"{g.get('name', g['gene'])}:"]
        if g.get("description"):
            bits.append(g["description"])
        if g.get("diseases"):
            bits.append("Diseases: " + ", ".join(g["diseases"]) + ".")
        g_lines.append("- " + " ".join(bits))
    if g_lines:
        lines.append("GenomicsDB (CIViC, expert-curated):")
        lines.extend(g_lines)

    vs_lines: list[str] = []
    for v in variantscape.get("variants", []):
        if not v.get("in_variantscape"):
            continue
        label = f"{v['gene']} {v['change']}".strip()
        if v.get("sensitising") or v.get("resistance"):
            ctx = variantscape.get("cancer_type", "")
            sens = _fmt_assoc(v.get("sensitising", [])) or "none identified"
            res = _fmt_assoc(v.get("resistance", [])) or "none identified"
            vs_lines.append(
                f"- {label} in {ctx}: sensitising associations - {sens}; "
                f"resistance associations - {res}."
            )
            if v.get("other_cancer_types"):
                vs_lines.append(
                    f"  Other cancer types associated with {label}: "
                    f"{_fmt_assoc(v['other_cancer_types'])}."
                )
        elif v.get("associated_cancer_types"):
            vs_lines.append(
                f"- {label}: associated cancer types - "
                f"{', '.join(v['associated_cancer_types'])}."
            )
    for g in variantscape.get("genes", []):
        treats = _fmt_assoc(g.get("top_treatments", []), limit=8)
        cancers = _fmt_assoc(g.get("top_cancer_types", []), limit=8)
        detail = []
        if treats:
            detail.append(f"treatments - {treats}")
        if cancers:
            detail.append(f"cancer types - {cancers}")
        vs_lines.append(
            f"- {g['gene']} (gene-level: aggregated across {g['n_variants']} "
            f"literature-mined variant records; co-occurrence strength only, "
            f"sensitising/resistant direction NOT resolved): " + "; ".join(detail) + "."
        )
    if vs_lines:
        lines.append("")
        lines.append("Variantscape (literature-mined associations, not expert-curated):")
        lines.extend(vs_lines)

    if not lines:
        return ""
    return "STRUCTURED FINDINGS FROM THE EVIDENCE DATABASE:\n" + "\n".join(lines)


def build_retrieval_query(query: str, profile: dict, genomic: dict, variantscape: dict) -> str:
    """Enrich the literature-retrieval query with gene / variant / treatment terms."""
    terms: list[str] = []
    for v in profile.get("variants", []):
        t = f"{v['gene']} {v['change']}".strip()
        if t:
            terms.append(t)
    for g in profile.get("genes", []):
        if g not in " ".join(terms):
            terms.append(g)
    if profile.get("cancer_type"):
        terms.append(profile["cancer_type"])
    for v in variantscape.get("variants", []):
        for e in (v.get("sensitising", []) + v.get("resistance", []))[:4]:
            terms.append(e["name"])
    for g in variantscape.get("genes", []):
        for e in g.get("top_treatments", [])[:4]:
            terms.append(e["name"])

    seen: set[str] = set()
    uniq = [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]
    extra = " ".join(uniq)
    return f"{query} {extra}".strip() if extra else query


def resources_consulted(genomic: dict, variantscape: dict) -> list[str]:
    used = ["LiteratureDB (papers + guidelines)"]
    if genomic.get("found"):
        used.insert(0, "GenomicsDB")
    if variantscape.get("found"):
        idx = 1 if genomic.get("found") else 0
        used.insert(idx, "Variantscape")
    return used
