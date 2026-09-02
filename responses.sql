-- Answer cache for the LiteratureDB /answer page.
--
-- One row per generated answer, keyed by the response_id held in the user's
-- session, so a page reload / back-navigation can re-render without re-running
-- retrieval or the LLM. The table is bounded by _prune_qa_data() in app.py
-- (keeps the EVIDENCE_DB_QA_CACHE_MAX_ROWS most recent rows).
--
-- NOTE: not dropped on startup -- init_db() creates it if absent and adds any
-- missing columns, so cached answers survive a restart and schema changes are
-- applied in place.

CREATE TABLE IF NOT EXISTS qa_data (
    id VARCHAR(255) PRIMARY KEY,
    query TEXT,
    patient_data TEXT,
    papers TEXT,
    response TEXT,
    filters TEXT,
    guidelines TEXT,
    created_at TEXT
);

-- Missing columns on an older table, and the created_at index, are added by
-- init_db._migrate_qa_data() after this script runs.
