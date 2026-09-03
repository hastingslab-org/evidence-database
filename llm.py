from openai import OpenAI
import json

import config

# Retrieval / prompt parameters (see config.py / .env.example to override)
SYSTEM_MSG = config.LLM_SYSTEM_MSG
GUIDELINE_SYSTEM_MSG = config.LLM_GUIDELINE_SYSTEM_MSG
INTEGRATED_SYSTEM_MSG = config.LLM_INTEGRATED_SYSTEM_MSG
MODEL = config.LLM_MODEL
NB_PAPERS_LLM = config.LLM_NUM_PAPERS


def _client() -> OpenAI:
    return OpenAI(
        api_key=config.OPENAI_API_KEY or None,
        base_url=config.OPENAI_BASE_URL,
    )


def call_llm_json(system_msg: str, user_msg: str) -> dict:
    """Single non-streaming completion that is expected to return a JSON object.

    Used by the front-page Personalised Treatment Query to turn a free-text
    patient description into a structured profile. Returns ``{}`` on any
    transport or parse failure so the caller can degrade to a
    literature-and-guidelines-only answer rather than erroring.
    """
    try:
        kwargs = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        try:
            resp = _client().chat.completions.create(
                response_format={"type": "json_object"}, **kwargs
            )
        except Exception:
            # Not every OpenAI-compatible backend accepts response_format.
            resp = _client().chat.completions.create(**kwargs)
        content = (resp.choices[0].message.content or "").strip()
        # Tolerate a fenced or prose-wrapped object.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        print(f"[llm] call_llm_json failed: {e}")
        return {}


def get_relevant_papers(user_query, collection, patient_data=None, allowed_ids=None):
    """ Gets the n most relevant papers from a chroma-db collection (where n is NB_PAPERS_LLM).

    If ``allowed_ids`` is given, retrieval is restricted to that set of document
    ids: we over-fetch and then keep the top NB_PAPERS_LLM that fall within it.
    """

    query = user_query
    if patient_data is not None:
        query = user_query + json.dumps(patient_data)  # combine query and patient data TODO: investigate other options

    if allowed_ids is None:
        return collection.query(query_texts=[query], n_results=NB_PAPERS_LLM)

    allowed = set(allowed_ids)
    try:
        corpus_size = collection.count()
    except Exception:
        corpus_size = 10_000
    over_fetch = min(corpus_size, max(NB_PAPERS_LLM * 40, 1000))
    raw = collection.query(query_texts=[query], n_results=over_fetch)

    keep = [i for i, doc_id in enumerate(raw["ids"][0]) if doc_id in allowed][:NB_PAPERS_LLM]
    keyed = ("ids", "documents", "metadatas", "distances", "embeddings", "uris", "data")
    return {
        k: [[raw[k][0][i] for i in keep]]
        for k in keyed
        if raw.get(k) is not None and raw.get(k)[0] is not None
    }


def call_llm_stream(user_query, title_and_abst, patient_data, guidelines_block="",
                    extra_context=""):
    """Builds a message based on user query, papers, patient characteristics and
    (optionally) official guideline recommendations. Then generates and live
    streams a response using the LLM.

    ``extra_context`` carries the structured GenomicsDB / Variantscape findings
    block assembled by the front-page Personalised Treatment Query; it is empty
    for the ordinary LiteratureDB search.
    """

    #Patient data to string
    patient_string = '\n'.join(f"{key.replace('_', ' ').title()}: {', '.join(value) if isinstance(value, list) else value}" for key, value in patient_data.items())

    system_content = SYSTEM_MSG
    guideline_section = ""
    if guidelines_block:
        system_content = SYSTEM_MSG + " " + GUIDELINE_SYSTEM_MSG
        guideline_section = guidelines_block + "\n\n"

    findings_section = ""
    if extra_context:
        system_content = system_content + " " + INTEGRATED_SYSTEM_MSG
        findings_section = extra_context + "\n\n"

    #LLM
    openai_client = _client()
    messages=[
    {"role": "system", "content": system_content},
    {"role": "user", "content": "The patient has the following characteristics: " + patient_string + "\n\n"
      + findings_section
      + guideline_section
      + "Using the following papers and abstract: " + title_and_abst + "\n\n"
      + "Please answer this question: " + user_query} #PROMPT + user_query + title_and_abst
    ]

    #generate response
    try:
        # Call OpenAI API with streaming enabled
        response = openai_client.chat.completions.create(
            model=MODEL,  # the model you're using
            messages=messages,
            stream=True  # Enable streaming
        )
        # Stream tokens as they arrive
        for chunk in response:
            if len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                yield delta.content

    except Exception as e:
        yield f"Error: {str(e)}"
