import time
from chromadb.utils import embedding_functions
from openai import OpenAI

CHROMA_DATA_PATH = "./chroma_data2/"
COLLECTION_NAME = "searchable_db_collection"

SYSTEM_MSG  = "You are a medical expert assisting doctors and clinicians in decision making" #"You are a helpful systematic reviewing assistant" #TODO try diff sytsem prompt
#PROMPT = "Please answer the following question using the following paper titles and abstracts."
MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
NB_PAPERS_LLM = 5


def get_relevant_papers(user_query, collection):
    #get relevant papers from collection
    query_results = collection.query(
    query_texts=[user_query],
    n_results=NB_PAPERS_LLM,
    )

    return query_results


def call_llm_stream(user_query, title_and_abst, patient_data):
    #Patient Data to string
    patient_string = '\n'.join(f"{key.replace('_', ' ').title()}: {', '.join(value) if isinstance(value, list) else value}" for key, value in patient_data.items())
    
    #LLM
    openai_client = OpenAI(
            api_key = "W2oF2Q2NLaTmqj7LGOiJwp9Pdi47Rhhn",
            base_url="https://api.deepinfra.com/v1/openai",
        )
    messages=[
    {"role": "system", "content": SYSTEM_MSG},
    {"role": "user", "content": "Our patient has the following characteristics: " + patient_string
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


"""def call_llm(user_query, collection):


    answer = generateFromPrompt(PROMPT + user_query + title_and_abst)

    return answer"""

"""print("Retrieved from ",query_results["ids"][0], \
    "\n Titles: \n", {query_results['metadatas'][0][i]['titles'] for i in range(NB_PAPERS_LLM)}, \
    "\n \n Answer:",answer,\
    "\n\nTitle and abstract:",query_results['documents'][0])"""


#query = "What are the expected outcomes for a middle-aged man with prostate cancer stage III? What are possible treatments?"


"""def generate_llm_response(prompt, collection):
    #Generator that streams the response word by word or character by character.
    #Replace this with your actual LLM API call logic.
    response_text = call_llm(prompt, collection)  # Call to LLM function
    for word in response_text.split():  # Stream word by word
        yield word + " "
        time.sleep(0.1)  # Simulate a delay (for effect)"""


