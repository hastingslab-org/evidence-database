"""LiteratureDB clinical-guideline retrieval.

Adds official clinical-practice-guideline recommendations for genitourinary
cancers alongside the primary-literature RAG results, so the answer produced by
the search page is anchored to guidance from ESMO, the EAU, Onkopedia / SGMO and
Swiss consensus statements.

The recommendation snippets live in ``guideline_sources/guidelines_data.json``.
They are short, paraphrased summaries with a deep link back to the official
document -- *not* verbatim reproductions -- because most GU guideline texts are
copyrighted (EAU) or licensed CC BY-NC-ND (ESMO / Onkopedia). Replace or extend
that file with your institution's vetted wording; see the README beside it.

Retrieval uses the same sentence-transformer model as the literature collection
via a small in-memory Chroma collection. If Chroma or the model is unavailable
it falls back to lexical token-overlap scoring, so the feature never hard-fails.
"""

import json
import re
from functools import lru_cache

import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _raw() -> dict:
    try:
        with open(config.GUIDELINES_DATA_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[guidelines] could not load {config.GUIDELINES_DATA_PATH}: {exc}")
        return {}


@lru_cache(maxsize=1)
def load_guidelines() -> list[dict]:
    """Return the list of guideline recommendation dicts (normalised)."""
    data = _raw()
    items = data.get("recommendations", []) if isinstance(data, dict) else (data or [])

    out = []
    for i, item in enumerate(items):
        item = dict(item)
        item.setdefault("id", f"guideline-{i}")
        tum = item.get("tumour") or item.get("tumours") or []
        item["tumour"] = [tum] if isinstance(tum, str) else list(tum)
        out.append(item)
    return out


@lru_cache(maxsize=1)
def source_catalogue() -> dict:
    """Return the {source key: {name, url, licence, region}} attribution map."""
    data = _raw()
    return data.get("sources", {}) if isinstance(data, dict) else {}


def all_sources() -> list[dict]:
    """Flat, template-friendly list of the issuing bodies."""
    return [{"key": k, **v} for k, v in source_catalogue().items()]


def disclaimer() -> str:
    meta = _raw().get("_meta", {}) if isinstance(_raw(), dict) else {}
    return meta.get("disclaimer", "")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _flatten(value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _embed_text(item: dict) -> str:
    """The text we match a query against for one recommendation."""
    return " ".join(
        part for part in (
            ", ".join(item.get("tumour", [])),
            item.get("title", ""),
            item.get("recommendation", ""),
        ) if part
    )


def _tumour_matches(item: dict, cancer_type: str | None) -> bool:
    if not cancer_type:
        return True
    ct = cancer_type.strip().lower()
    tums = [t.lower() for t in item.get("tumour", [])]
    if not tums or "any" in tums:
        return True
    return any(ct in t or t in ct for t in tums)


def _public(item: dict, score: float) -> dict:
    """JSON-safe view of a recommendation, with source metadata folded in."""
    src_key = item.get("source", "")
    src = source_catalogue().get(src_key, {})
    return {
        "id": item["id"],
        "source": src_key,
        "source_name": src.get("name", src_key),
        "title": item.get("title", ""),
        "tumour": item.get("tumour", []),
        "year": item.get("year"),
        "strength": item.get("strength", ""),
        "recommendation": item.get("recommendation", ""),
        "url": item.get("url") or src.get("url", ""),
        "licence": item.get("licence") or src.get("licence", ""),
        "region": item.get("region") or src.get("region", ""),
        "relevance": round(float(score), 3),
    }


# --------------------------------------------------------------------------- #
# In-memory embedding collection (optional; lexical fallback otherwise)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _collection():
    """Build an ephemeral Chroma collection of the guideline snippets.

    Returns ``None`` if Chroma or the embedding model cannot be loaded, in
    which case :func:`_rank` falls back to lexical scoring.
    """
    items = load_guidelines()
    if not items:
        return None
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.EphemeralClient(
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        col = client.create_collection(name="guidelines", embedding_function=ef)
        col.add(
            ids=[it["id"] for it in items],
            documents=[_embed_text(it) for it in items],
            metadatas=[{"source": it.get("source", "")} for it in items],
        )
        return col
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"[guidelines] embedding retrieval unavailable ({exc}); using lexical match")
        return None


def _rank(query: str, pool: list[dict], n: int) -> list[tuple[dict, float]]:
    by_id = {it["id"]: it for it in load_guidelines()}
    pool_ids = {it["id"] for it in pool}

    col = _collection()
    if col is not None:
        try:
            res = col.query(
                query_texts=[query or ""],
                n_results=min(len(by_id), max(n * 4, n)),
            )
            ranked = []
            for gid, dist in zip(res["ids"][0], res["distances"][0]):
                if gid in pool_ids:
                    # Chroma returns ascending distance; keep that order. The
                    # score is for display only (0 = far), clamped for L2 metrics.
                    ranked.append((by_id[gid], max(0.0, 1.0 - float(dist))))
                if len(ranked) >= n:
                    break
            if ranked:
                return ranked
        except Exception as exc:  # pragma: no cover
            print(f"[guidelines] query failed ({exc}); using lexical match")

    qt = _tokens(query)
    scored = [
        (it, len(qt & _tokens(_embed_text(it))) / (len(qt) + 1))
        for it in pool
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:n]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_relevant_guidelines(query, patient_data=None, cancer_type=None, n=None):
    """Return up to ``n`` guideline recommendations relevant to the question.

    ``cancer_type`` (or ``patient_data['Cancer type']``) restricts the pool to
    matching tumours before ranking; if nothing matches, the whole set is used
    rather than returning an empty list.
    """
    if not config.GUIDELINES_ENABLED:
        return []
    if n is None:
        n = config.GUIDELINES_NUM_RESULTS

    items = load_guidelines()
    if not items:
        return []

    if cancer_type is None and patient_data:
        cancer_type = patient_data.get("Cancer type") or patient_data.get("Cancer Type")

    pool = [it for it in items if _tumour_matches(it, cancer_type)] or items

    q = query or ""
    if patient_data:
        q = q + " " + " ".join(_flatten(v) for v in patient_data.values())

    return [_public(it, score) for it, score in _rank(q, pool, n)]


def format_guidelines_for_prompt(guidelines) -> str:
    """Render retrieved recommendations as a text block for the LLM prompt."""
    if not guidelines:
        return ""
    lines = []
    for g in guidelines:
        tag = " ".join(
            part for part in (g.get("source_name"), str(g.get("year") or "")) if part
        ).strip()
        strength = f" [strength: {g['strength']}]" if g.get("strength") else ""
        lines.append(f"- ({tag}) {g.get('recommendation', '')}{strength} Source: {g.get('url', '')}")
    return (
        "OFFICIAL CLINICAL PRACTICE GUIDELINE RECOMMENDATIONS "
        "(paraphrased summaries; verify against the cited source):\n"
        + "\n".join(lines)
    )
