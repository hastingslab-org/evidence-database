"""Centralised configuration for the EvidenceDB app.

All settings are read from environment variables (optionally populated from a
local ``.env`` file next to this module) with sensible defaults, so the app
runs out of the box for development but every value can be overridden in
production without touching code.

Import this module from anywhere in the project::

    import config
    conn = sqlite3.connect(config.SQLITE_DB_PATH)

Path settings are resolved relative to this file's directory, so they no
longer depend on the process working directory.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional; env vars still work without it
    def load_dotenv(*_args, **_kwargs):
        return False

BASE_DIR = Path(__file__).resolve().parent

# Load a local .env if present (does not override already-set env vars).
load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _path(env_name: str, default: str) -> Path:
    """Return an absolute Path from ``env_name`` or ``default``.

    Relative values are anchored to :data:`BASE_DIR`.
    """
    raw = os.environ.get(env_name, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p)


def _bool(env_name: str, default: bool = False) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Flask / web server
# --------------------------------------------------------------------------- #
SECRET_KEY = os.environ.get("EVIDENCE_DB_SECRET_KEY", "dev-insecure-change-me")
HOST = os.environ.get("EVIDENCE_DB_HOST", "127.0.0.1")
PORT = int(os.environ.get("EVIDENCE_DB_PORT", "5000"))
DEBUG = _bool("EVIDENCE_DB_DEBUG", False)

# Comma-separated list of allowed CORS origins, or "*" for all.
_cors = os.environ.get("EVIDENCE_DB_CORS_ORIGINS", "*").strip()
CORS_ORIGINS = "*" if _cors == "*" else [o.strip() for o in _cors.split(",") if o.strip()]


# --------------------------------------------------------------------------- #
# SQLite (GenomicsDB tables + LiteratureDB qa_data)
# --------------------------------------------------------------------------- #
SQLITE_DB_PATH = _path("EVIDENCE_DB_SQLITE_PATH", "database.db")
RESPONSES_SQL_PATH = _path("EVIDENCE_DB_RESPONSES_SQL", "responses.sql")
GENOMICS_SQL_PATH = _path("EVIDENCE_DB_GENOMICS_SQL", "genomics.sql")

# Answer cache (qa_data): keep only the N most recent generated answers so the
# table stays bounded under multi-user load.
QA_CACHE_MAX_ROWS = int(os.environ.get("EVIDENCE_DB_QA_CACHE_MAX_ROWS", "500"))
# Seconds a DB connection waits for a competing write before erroring.
SQLITE_BUSY_TIMEOUT = float(os.environ.get("EVIDENCE_DB_SQLITE_BUSY_TIMEOUT", "30"))


# --------------------------------------------------------------------------- #
# ChromaDB vector store (LiteratureDB RAG)
# --------------------------------------------------------------------------- #
CHROMA_DB_PATH = _path("EVIDENCE_DB_CHROMA_PATH", "chroma_data_20250603")
CHROMA_COLLECTION = os.environ.get(
    "EVIDENCE_DB_CHROMA_COLLECTION", "searchable_db_collection_fd"
)
# The `searchable_db_collection_fd` collection is built by
# db_initializations/refresh_literature_db.py with all-MiniLM-L6-v2.
# The embedding model used for queries MUST match the one used to build the
# collection, so this is a single source of truth for every route.
# After a rebuild, set EVIDENCE_DB_CHROMA_PATH to the new dated directory.
EMBEDDING_MODEL = os.environ.get(
    "EVIDENCE_DB_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)


# --------------------------------------------------------------------------- #
# LLM (used by llm.py)
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# The app has historically run against DeepInfra's OpenAI-compatible endpoint.
# Set OPENAI_BASE_URL to an empty string in the environment to target real OpenAI.
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", "https://api.deepinfra.com/v1/openai"
) or None
LLM_MODEL = os.environ.get(
    "EVIDENCE_DB_LLM_MODEL", "meta-llama/Meta-Llama-3.1-70B-Instruct"
)
LLM_SYSTEM_MSG = os.environ.get(
    "EVIDENCE_DB_LLM_SYSTEM_MSG",
    "You are a medical expert assisting doctors and clinicians in decision making",
)
LLM_NUM_PAPERS = int(os.environ.get("EVIDENCE_DB_LLM_NUM_PAPERS", "5"))
# Extra system-prompt instruction appended when guideline recommendations are
# injected into the answer prompt (see guidelines.py / llm.py).
LLM_GUIDELINE_SYSTEM_MSG = os.environ.get(
    "EVIDENCE_DB_LLM_GUIDELINE_SYSTEM_MSG",
    "When official clinical practice guideline recommendations are supplied, use "
    "them as the primary basis for the answer, cite the issuing body and year "
    'inline (e.g. "(ESMO 2024)"), and state explicitly where the retrieved '
    "primary literature agrees with or diverges from the guideline. Remind the "
    "reader to confirm against the official guideline text.",
)


# --------------------------------------------------------------------------- #
# GenomicsDB upstream source (CIViC GraphQL)
# --------------------------------------------------------------------------- #
CIVIC_API_URL = os.environ.get(
    "EVIDENCE_DB_CIVIC_API_URL", "https://civicdb.org/api/graphql"
)


# --------------------------------------------------------------------------- #
# Variantscape data files
# --------------------------------------------------------------------------- #
VARIANTSCAPE_GRAPH_PATH = _path(
    "EVIDENCE_DB_VARIANTSCAPE_GRAPH", "variantscape/network_graph_weighted.gml"
)
VARIANTSCAPE_CONSENSUS_PATH = _path(
    "EVIDENCE_DB_VARIANTSCAPE_CONSENSUS",
    "variantscape/final_variant_treatment_consensus.csv",
)
VARIANTSCAPE_METADATA_MAPPING_PATH = _path(
    "EVIDENCE_DB_VARIANTSCAPE_METADATA_MAPPING",
    "variantscape/metadata_mapping_transposed.csv",
)
CANCER_SYNONYMS_PATH = _path(
    "EVIDENCE_DB_CANCER_SYNONYMS", "static/Network_cancer_synonyms.csv"
)


# --------------------------------------------------------------------------- #
# LiteratureDB
# --------------------------------------------------------------------------- #
# One-hot patient-attribute matrix (one row per paper) that backs the
# left-hand publication filters. Semicolon-delimited. Built next to the Chroma
# store by db_initializations/refresh_literature_db.py so it always matches the
# active corpus; the legacy shipped copy is used until the first rebuild.
LITERATURE_METADATA_CSV = _path(
    "EVIDENCE_DB_LITERATURE_METADATA_CSV",
    str(CHROMA_DB_PATH / "filter_metadata.csv"),
)
LITERATURE_METADATA_CSV_LEGACY = _path(
    "EVIDENCE_DB_LITERATURE_METADATA_CSV_LEGACY",
    "static/evidence-db-angular/assets/prostate-metadata.csv",
)
# How many papers to show in the browsable "matching publications" list.
LITERATURE_PAPER_LIST_LIMIT = int(
    os.environ.get("EVIDENCE_DB_LITERATURE_PAPER_LIST_LIMIT", "100")
)

# --- Clinical guideline retrieval (search-page answer) ---
# Curated, paraphrased guideline recommendations injected alongside the
# literature RAG results. See guideline_sources/README.md.
GUIDELINES_ENABLED = _bool("EVIDENCE_DB_GUIDELINES_ENABLED", True)
GUIDELINES_DATA_PATH = _path(
    "EVIDENCE_DB_GUIDELINES_DATA", "guideline_sources/guidelines_data.json"
)
GUIDELINES_NUM_RESULTS = int(os.environ.get("EVIDENCE_DB_GUIDELINES_NUM_RESULTS", "4"))


# --------------------------------------------------------------------------- #
# Variantscape tuning parameters
# --------------------------------------------------------------------------- #
# Absolute total-weight floors for highlighting a result in the UI.
TREATMENT_MIN_HIGHLIGHT = int(os.environ.get("EVIDENCE_DB_TREATMENT_MIN_HIGHLIGHT", "300"))
CANCER_MIN_HIGHLIGHT = int(os.environ.get("EVIDENCE_DB_CANCER_MIN_HIGHLIGHT", "80"))
# Percentile cut-offs for "top" treatment / cancer weights.
TREATMENT_THRESHOLD_PERCENTILE = int(
    os.environ.get("EVIDENCE_DB_TREATMENT_THRESHOLD_PERCENTILE", "80")
)
CANCER_THRESHOLD_PERCENTILE = int(
    os.environ.get("EVIDENCE_DB_CANCER_THRESHOLD_PERCENTILE", "80")
)
# Fuzzy-match acceptance ratio (0-1) shared by the genomics and variantscape lookups.
MATCHING_RATIO_THRESH = float(os.environ.get("EVIDENCE_DB_MATCHING_RATIO_THRESH", "0.8"))


# --------------------------------------------------------------------------- #
# Partner logos (bottom banner)
# --------------------------------------------------------------------------- #
PARTNERS_FILE = _path("EVIDENCE_DB_PARTNERS_FILE", "partners.json")
