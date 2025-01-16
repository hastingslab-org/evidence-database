import os
import sys
import json

# Add the parent directory to sys.path
#parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, Response, session, jsonify
from werkzeug.exceptions import abort, RequestEntityTooLarge
from llm import call_llm_stream, get_relevant_papers
import sqlite3 
import chromadb

# Get the db directory path
DB_PATH = 'chroma_data2'

"""def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_query(query_id):
    conn = get_db_connection()
    query = conn.execute('SELECT * FROM queries WHERE id = ?',
                        (query_id,)).fetchone()
    conn.close()
    if query is None:
        abort(404)
    return query"""

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'


@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(error):
    return "The request is too large!", 413

@app.route('/')
def search_query_page():
    return render_template('search_query_page.html')

@app.route("/answer", methods=['GET'])
def answer_page_get(): #TODO same functiomn as asnwer_page()
    # Retrieve the data stored in session
    query = session.get('query', '')
    patient_data = session.get('patient_data', {})
    query_results = session.get('query_results', {})
    llm_answer = session.get('llm_answer', '')
    return render_template('answer.html', query=query, query_results=query_results, patient_data=patient_data, llm_answer=llm_answer)


@app.route("/answer",  methods=['POST'])
def answer_page():
    form_data = request.form.to_dict(flat=False)  # Converts form data to a dictionary

    # Separate the query from the patient data
    query = form_data.pop('query', [None])[0]  # Get the query field and remove it from the form data
    patient_data = {key: value[0] if len(value) == 1 else value for key, value in form_data.items()}

    # Render the answer.html template with the data
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(name="searchable_db_collection")
    query_results = get_relevant_papers(query, collection, patient_data)

    # Store the answer in session (LLM response could be stored as part of query_results or separately)
    session['query'] = query
    session['patient_data'] = patient_data
    session['query_results'] = query_results

    return render_template('answer.html', query=query, query_results = query_results, patient_data=patient_data, llm_answer=None) #TODO check query_results optimization?


# Route for streaming the LLM response
@app.route('/stream_response', methods=['POST'])
def stream_response():  
    try: 
        #get data
        data = request.get_json()
        query = data.get('query', '')
        papers = data.get('papers', {})
        patient_data = data.get('patient_data', {})
       
        #titles = [paper["titles"] for paper in papers["metadatas"][0]]
        def generate_response():

            # Stream the LLM response
            title_and_abst = ",".join(papers["documents"][0])
            
            chunks = []
            for chunk in call_llm_stream(query, title_and_abst, patient_data):
                if chunk:

                    chunks.append(chunk) #collecting chunks
                    yield chunk.encode('utf-8')
            
            full_response = "".join(chunks)
            session["llm_answer"] = full_response.encode('utf-8')

        return Response(generate_response(), content_type='text/event-stream')
    
    except Exception as e:
        print(f"Error: {e}")
        return Response("An error occurred while streaming the response.", status=500)
    
@app.route('/paper_<int:paper_id>', methods=['POST', 'GET'])
def view_paper(paper_id):
    if request.method == 'POST':
        # Get the paper data from the request
        paper_data = request.get_json()

        # Store paper data in a global variable TODO: or use a better solution like session
        global selected_paper
        selected_paper = paper_data

    if not selected_paper:
        return "No paper data available", 404

    # Render the paper details page
    return render_template('paper.html', paper=selected_paper)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
