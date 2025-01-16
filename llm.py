#from chromadb.utils import embedding_functions
from openai import OpenAI
import json

CHROMA_DATA_PATH = "./chroma_data2/"
COLLECTION_NAME = "searchable_db_collection"
SYSTEM_MSG  = "You are a medical expert assisting doctors and clinicians in decision making" #"You are a helpful systematic reviewing assistant" #TODO try diff sytsem prompt
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
NB_PAPERS_LLM = 5
API_KEY = ""


def get_relevant_papers(user_query, collection, patient_data=None):
    #get relevant papers from collection

    if patient_data is not None: 
        query = user_query + json.dumps(patient_data)    
         
    query_results = collection.query(
    query_texts=[query],
    n_results=NB_PAPERS_LLM,
    )

    return query_results


def call_llm_stream(user_query, title_and_abst, patient_data):
    
    #Patient Data to string
    patient_string = '\n'.join(f"{key.replace('_', ' ').title()}: {', '.join(value) if isinstance(value, list) else value}" for key, value in patient_data.items())
    
    #LLM
    openai_client = OpenAI(
            api_key = API_KEY,
            base_url="https://api.deepinfra.com/v1/openai",
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

