# Clinical guideline sources for LiteratureDB

This folder holds the curated guideline knowledge base that the search page
(`/literature` → `/answer`) uses to anchor its answer to official clinical
practice guidance for genitourinary cancers, alongside the primary-literature
RAG results.

## Files

- `guidelines_data.json` – the knowledge base. Two sections:
  - `sources` – the issuing bodies (attribution, link, licence, region).
  - `recommendations` – short, **paraphrased** recommendation snippets, each
    tagged with `source`, `tumour`, `year`, `strength` and a deep `url`.

The module `guidelines.py` (repo root) loads this file, embeds the snippets
(same `all-MiniLM-L6-v2` model as the literature collection, via an in-memory
Chroma collection) and returns the most relevant few for a given question and
patient. If Chroma or the embedding model is unavailable it falls back to a
lexical token-overlap match, so the feature never hard-fails.

## Licensing posture — important

The full text of most GU guidelines is **not** freely redistributable:

| Body | Access | Reuse |
|------|--------|-------|
| **ESMO** (Annals of Oncology) | Open access | Usually CC BY-NC-ND 4.0 – redistribution for non-commercial use **with attribution, no derivatives**. Retrieve + quote + link; do not republish reformatted text. |
| **EAU** (uroweb.org) | Free to read | © EAU – reproduction needs **written permission**. Link + cite by default; email the EAU Guidelines Office to store full text. |
| **Onkopedia / SGMO** | Free to read | © DGHO – check portal terms; parts CC BY-NC-ND. |
| **Swiss Medical Weekly** consensus statements | Open access | **CC BY 4.0** – the friendliest; may be ingested and reworded with attribution. |
| **ASCO** (JCO) | Free to read | © ASCO – reuse by permission. |
| **NCCN** | Free with registration | Redistribution/derivatives tightly licensed – **do not embed**. |

Because of this, `guidelines_data.json` deliberately contains only **our own
concise paraphrases** plus a deep link back to the authoritative document. It is
not a copy of any guideline. Every answer and every card shown to the user
carries the "paraphrased – verify against the official source" disclaimer and
the source link, which satisfies the attribution requirement.

## Extending / replacing

1. Add an object to `recommendations`. Required keys: `source` (must exist in
   `sources`), `tumour` (list; values should match the cancer-type options in
   `templates/literature.html`, or `["Any"]`), `recommendation` (your paraphrase,
   1–3 sentences), `url` (deep link to the exact section).
   Optional: `id` (stable slug – auto-generated if omitted), `year`, `strength`,
   `title`, `region`, `licence`.
2. To add a new issuing body, add an entry to `sources` and reference its key.
3. Bump `_meta.last_reviewed`.
4. No rebuild step: the in-memory collection is rebuilt on app start. Restart the
   Flask process to pick up edits (the loader is `lru_cache`d).

For a production-grade version, replace these paraphrases with wording your
institution has vetted, and consider the **NICE Syndication API** (structured,
licensed) or the **CC BY** Swiss Medical Weekly statements as machine-readable
feeds.

## Config

Set in `.env` (see `.env.example`):

- `EVIDENCE_DB_GUIDELINES_ENABLED` – `true` / `false` (default `true`).
- `EVIDENCE_DB_GUIDELINES_DATA` – path to the JSON (default this file).
- `EVIDENCE_DB_GUIDELINES_NUM_RESULTS` – how many recommendations to inject
  (default `4`).
- `EVIDENCE_DB_LLM_GUIDELINE_SYSTEM_MSG` – extra system-prompt instruction for
  how the model should use and cite the guideline block.
