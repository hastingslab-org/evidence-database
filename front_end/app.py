import os
import sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from flask import Flask, render_template, request, url_for, flash, redirect, Response, session, jsonify
from werkzeug.exceptions import abort, RequestEntityTooLarge
from llm import call_llm_stream, get_relevant_papers
import sqlite3 
import chromadb
import urllib.parse
import json


# Get the db directory path
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_data2'))

def get_db_connection():
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
    return query

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key' #TODO investigate


@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(error):
    return "The request is too large!", 413

# Route to handle the query form
@app.route('/submit-query', methods=['POST'])
def handle_query():
    data = request.get_json()  # Get the JSON payload

    # Extract query and patient data from the request
    query = data.get('query')
    patient_data = data.get('patient_data')

    # Store the query and patient data in the session
    session['query'] = query
    session['patient_data'] = patient_data

    return jsonify({'message': 'Data received successfully'}), 200


@app.route('/')
def search_query_page():
    return render_template('search_query_page.html')

@app.route("/answer",  methods=['POST'])
def answer_page():
    query = request.form.get('query', '')
    patient_data = request.form.get('patient_data', '')
   
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(name="searchable_db_collection")
    query_results = get_relevant_papers(query, collection)
    return render_template('answer.html', query=query, query_results = query_results, patient_data=patient_data) #TODO check query_results optimization?


# Route for streaming the LLM response
@app.route('/stream_response', methods=['POST'])
def stream_response():  
    try: 
        #get data
        data = request.get_json()
        query = data.get('query', '')
        papers = data.get('papers', {})
        patient_data = session.get('patient_data', {})
       
        titles = [paper["titles"] for paper in papers["metadatas"][0]]
        def generate_response():
            yield "Selected Papers:\n"
            for idx, title in enumerate(titles, 1):
                yield f"{idx}. {title}\n"
            yield "\n---\nResponse:\n\n"

            # Stream the LLM response
            title_and_abst = ",".join(papers["documents"][0])
            for chunk in call_llm_stream(query, title_and_abst, patient_data):
                yield chunk

        return Response(generate_response(), content_type='text/event-stream')
    
    except Exception as e:
        print(f"Error: {e}")
        return Response("An error occurred while streaming the response.", status=500)


"""@app.route('/search', methods=['GET'])
def user_search_page():
    if request.method == 'POST':
        # Get the search query from the form
        content = request.form['query']

        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name="searchable_db_collection")
        llm_response = call_llm(content, collection=collection)
        
        # Insert the query into the database
        conn = get_db_connection()
        conn.execute('INSERT INTO queries (content) VALUES (?)', (content,))
        conn.commit()
        # Fetch only the query that was just inserted
        query = conn.execute('SELECT * FROM queries WHERE content = ? ORDER BY created DESC LIMIT 1', (content,)).fetchone()
        conn.close()
        
        # Pass only this query to the template
        return render_template('search.html', queries=[query],  llm_response=llm_response)
    
    # If the request is not POST, render an empty page or handle GET requests
    return render_template('search.html')"""

if __name__ == "__main__":
    app.run(debug=True)