from openai import OpenAI
import json

import config

# Retrieval / prompt parameters (see config.py / .env.example to override)
SYSTEM_MSG = config.LLM_SYSTEM_MSG
MODEL = config.LLM_MODEL
NB_PAPERS_LLM = config.LLM_NUM_PAPERS


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


def call_llm_stream(user_query, title_and_abst, patient_data):
    """Builds a message based on user query, papers and patient characteristics.
    Then generates and live streams a response using the LLM"""

    #Patient data to string
    patient_string = '\n'.join(f"{key.replace('_', ' ').title()}: {', '.join(value) if isinstance(value, list) else value}" for key, value in patient_data.items())

    #LLM
    openai_client = OpenAI(
            api_key=config.OPENAI_API_KEY or None,
            base_url=config.OPENAI_BASE_URL,
        )
    messages=[
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": "The patient has the following characteristics: " + patient_string
      + "Using the following papers and abstract: " + title_and_abst
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
