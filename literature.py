"""LiteratureDB helpers: publication filters and overview aggregation.

The left-hand filters operate on a semicolon-delimited one-hot matrix
(``filter_metadata.csv``, config ``LITERATURE_METADATA_CSV``) with one row per
paper keyed by an integer ``id`` that matches the ChromaDB document id. That
file is (re)built alongside the vector store by
``db_initializations/refresh_literature_db.py`` and kept in the same directory,
so it always matches the active corpus; the legacy shipped copy
(``LITERATURE_METADATA_CSV_LEGACY``) is used until the first rebuild. The
columns are defined once here in :data:`FILTER_GROUPS`.

The overview panel aggregates the ChromaDB metadata (year / journal / country)
for whatever subset of ids the filters select.
"""

import csv
import json
import os
from collections import Counter
from functools import lru_cache

import config

# --------------------------------------------------------------------------- #
# Filter definitions
# --------------------------------------------------------------------------- #
# group key -> (display label, [(csv column, option label), ...])
# Within a group the selected options are OR-ed; groups are AND-ed together.
FILTER_GROUPS = [
    ("cancer_type", "Cancer Type", [
        ("Cancer type - Bladder Cancer", "Bladder Cancer"),
        ("Cancer type - Kidney Cancer", "Kidney Cancer"),
        ("Cancer type - Penile Cancer", "Penile Cancer"),
        ("Cancer type - Prostate Cancer", "Prostate Cancer"),
        ("Cancer type - Renal Cell Cancer", "Renal Cell Cancer"),
        ("Cancer type - Pelvis Cancer", "Pelvis Cancer"),
        ("Cancer type - Testicular Cancer", "Testicular Cancer"),
        ("Cancer type - Urethral Cancer", "Urethral Cancer"),
    ]),
    ("ecog", "ECOG Performance Status", [
        ("ECOG0", "0"), ("ECOG1", "1"), ("ECOG2", "2"),
        ("ECOG3", "3"), ("ECOG4", "4"), ("ECOG5", "5"),
    ]),
    ("ethnicity", "Ethnicity", [
        ("Ethnicity-Caucasian", "Caucasian"),
        ("Ethnicity-African American", "African American"),
        ("Ethnicity-Asian", "Asian"),
        ("Ethnicity-Hispanic", "Hispanic"),
        ("Ethnicity-Other", "Other"),
    ]),
    ("t_stage", "T stage", [
        ("TX", "TX"), ("T0", "T0"), ("Tis", "Tis"), ("T1", "T1"),
        ("T2", "T2"), ("T3", "T3"), ("T4", "T4"),
    ]),
    ("n_stage", "N stage", [
        ("NX", "NX"), ("N0", "N0"), ("N1", "N1"), ("N2", "N2"),
    ]),
    ("m_stage", "M stage", [
        ("MX", "MX"), ("M0", "M0"), ("M1", "M1"),
    ]),
    ("staging", "Staging Group", [
        ("Staging Group 0", "0"), ("Staging Group IA", "IA"),
        ("Staging Group IB", "IB"), ("Staging Group II", "II"),
        ("Staging Group III", "III"), ("Staging Group IV", "IV"),
    ]),
    ("metastasis", "Metastasis", [
        ("Metastasis - bone only", "Bone only"),
        ("Metastasis lymph only", "Lymph only"),
        ("Metastasis bone and lymph only", "Bone and lymph only"),
        ("Metastasis other than bone and lymph node", "Other"),
    ]),
    ("genomic", "Genomic Features", [
        ("DDR Deficiency", "DDR deficiency"),
    ]),
    ("risk", "Trial-Based Classification", [
        ("High Risk", "High risk"),
    ]),
]
# Removed with the move to full-GU coverage: the prostate-only "Disease
# Sensitivity" (hormone-sensitive / castration-resistant), "Treatment History"
# (ADT / ARPI) and "High volume" groups/options -- none applied to two or more
# of the cancer types above.

# All valid column names, for validating incoming filter payloads.
_VALID_COLUMNS = {col for _, _, opts in FILTER_GROUPS for col, _ in opts}
_GROUP_COLUMNS = {key: [c for c, _ in opts] for key, _, opts in FILTER_GROUPS}


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def _resolve_metadata_csv():
    """Prefer the corpus-aligned CSV; fall back to the legacy shipped copy."""
    primary = str(config.LITERATURE_METADATA_CSV)
    if os.path.exists(primary):
        return primary
    legacy = str(getattr(config, "LITERATURE_METADATA_CSV_LEGACY", "") or "")
    if legacy and os.path.exists(legacy):
        return legacy
    return primary


@lru_cache(maxsize=1)
def _load_metadata_rows():
    """Return {paper_id (str): {column: bool}} from the one-hot CSV."""
    path = _resolve_metadata_csv()
    rows = {}
    try:
        fh = open(path, newline="", encoding="utf-8")
    except OSError:
        print(f"[literature] filter metadata CSV not found ({path}); publication filters disabled")
        return rows
    with fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            pid = (row.get("id") or "").strip()
            if not pid:
                continue
            rows[pid] = {
                col: (row.get(col, "").strip().upper() == "TRUE")
                for col in _VALID_COLUMNS
            }
    return rows


def parse_filters(raw):
    """Normalise an incoming filter payload into {group_key: [column, ...]}.

    Accepts a JSON string or a dict. Unknown groups/columns are dropped.
    """
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    if not isinstance(raw, dict):
        return {}

    cleaned = {}
    for group, cols in raw.items():
        if group not in _GROUP_COLUMNS:
            continue
        if isinstance(cols, str):
            cols = [cols]
        valid = [c for c in cols if c in set(_GROUP_COLUMNS[group])]
        if valid:
            cleaned[group] = valid
    return cleaned


def matching_ids(filters):
    """Return the set of paper ids matching ``filters``, or None if no filters.

    None means "no restriction" (the whole corpus).
    """
    filters = parse_filters(filters)
    if not filters:
        return None

    rows = _load_metadata_rows()
    if not rows:
        # Metadata unavailable -> don't restrict (better than an empty result set).
        return None
    result = None
    for _group, cols in filters.items():
        group_ids = {
            pid for pid, flags in rows.items()
            if any(flags.get(col) for col in cols)
        }
        result = group_ids if result is None else (result & group_ids)
        if not result:
            return set()
    return result


# --------------------------------------------------------------------------- #
# Overview aggregation
# --------------------------------------------------------------------------- #
_ALPHA2_TO_ALPHA3 = {
    "AE": "ARE", "AL": "ALB", "AM": "ARM", "AO": "AGO", "AR": "ARG", "AT": "AUT",
    "AU": "AUS", "BA": "BIH", "BD": "BGD", "BE": "BEL", "BG": "BGR", "BR": "BRA",
    "CA": "CAN", "CH": "CHE", "CL": "CHL", "CN": "CHN", "CO": "COL", "CR": "CRI",
    "CU": "CUB", "CW": "CUW", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EC": "ECU", "EE": "EST", "EG": "EGY", "ES": "ESP", "ET": "ETH", "FI": "FIN",
    "FR": "FRA", "GB": "GBR", "GR": "GRC", "HK": "HKG", "HR": "HRV", "HU": "HUN",
    "ID": "IDN", "IE": "IRL", "IL": "ISR", "IN": "IND", "IQ": "IRQ", "IR": "IRN",
    "IT": "ITA", "JM": "JAM", "JO": "JOR", "JP": "JPN", "KE": "KEN", "KH": "KHM",
    "KR": "KOR", "KW": "KWT", "KZ": "KAZ", "LB": "LBN", "LK": "LKA", "LT": "LTU",
    "LU": "LUX", "LV": "LVA", "MA": "MAR", "MC": "MCO", "MO": "MAC", "MX": "MEX",
    "MY": "MYS", "NG": "NGA", "NL": "NLD", "NO": "NOR", "NZ": "NZL", "PA": "PAN",
    "PE": "PER", "PH": "PHL", "PK": "PAK", "PL": "POL", "PR": "PRI", "PS": "PSE",
    "PT": "PRT", "QA": "QAT", "RO": "ROU", "RS": "SRB", "RU": "RUS", "SA": "SAU",
    "SD": "SDN", "SE": "SWE", "SG": "SGP", "SI": "SVN", "SK": "SVK", "SN": "SEN",
    "SX": "SXM", "SY": "SYR", "TH": "THA", "TN": "TUN", "TR": "TUR", "TW": "TWN",
    "TZ": "TZA", "UA": "UKR", "UG": "UGA", "US": "USA", "ZA": "ZAF",
}

_TOP_JOURNALS = 15
_full_metadata_cache = None


def _all_metadata(collection):
    """Cached list of every document's metadata dict in the collection."""
    global _full_metadata_cache
    if _full_metadata_cache is None:
        got = collection.get(include=["metadatas"])
        _full_metadata_cache = list(zip(got["ids"], got["metadatas"]))
    return _full_metadata_cache


def _fetch_metadata(collection, ids):
    if ids is None:
        return _all_metadata(collection)
    ids = sorted(ids)
    if not ids:
        return []
    got = collection.get(ids=ids, include=["metadatas"])
    return list(zip(got["ids"], got["metadatas"]))


def overview_stats(collection, ids=None, paper_limit=None):
    """Aggregate the overview panel for the given subset of document ids.

    ``ids`` of None means the whole corpus.
    """
    if paper_limit is None:
        paper_limit = config.LITERATURE_PAPER_LIST_LIMIT

    records = _fetch_metadata(collection, ids)

    years, journals, countries = Counter(), Counter(), Counter()
    papers = []
    for doc_id, meta in records:
        meta = meta or {}
        year = meta.get("year")
        if isinstance(year, (int, float)) and year:
            years[int(year)] += 1
        journal = (meta.get("journal") or "").strip()
        if journal:
            journals[journal] += 1
        alpha3 = _ALPHA2_TO_ALPHA3.get((meta.get("countryMainAuthor") or "").strip())
        if alpha3:
            countries[alpha3] += 1
        papers.append({
            "id": doc_id,
            "title": meta.get("titles") or "(untitled)",
            "first_author": meta.get("first_author") or "",
            "year": int(year) if isinstance(year, (int, float)) and year else None,
            "journal": journal,
            "openalex_id": meta.get("openAlex_id") or "",
        })

    papers.sort(key=lambda p: (p["year"] is None, -(p["year"] or 0), p["title"]))

    return {
        "total": len(records),
        "by_year": [[y, years[y]] for y in sorted(years)],
        "by_journal": [[j, c] for j, c in journals.most_common(_TOP_JOURNALS)],
        "by_country": [[a3, c] for a3, c in countries.most_common()],
        "papers": papers[:paper_limit],
        "paper_limit": paper_limit,
    }
