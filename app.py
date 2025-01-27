import os
import sys
import json
import uuid

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

def get_db_connexion():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'  # Use the filesystem to store session data
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'my_secret_key' #TODO 

def get_response(response_id):
    """Fetch the stored response by its ID."""
    print("RESPONS_ID_get")
    print(session.get('response_id', {}))

    conn = get_db_connexion()
    cursor = conn.cursor()
    cursor.execute("SELECT response FROM qa_data WHERE id = ?", (response_id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()

    if row:
        return row[0]
    else:
        return jsonify({"error": "Response not found"}), 404
    
def get_query(id):
    """Fetch the stored response by its ID."""
    conn = get_db_connexion()
    cursor = conn.cursor()
    cursor.execute("SELECT query FROM qa_data WHERE id = ?", (id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()

    if row:
        print(row)
        return row["query"]
    else:
        return "error: Query not found", 404

def get_papers(id):
    """Fetch the stored response by its ID."""
    conn = get_db_connexion()
    cursor = conn.cursor()
    cursor.execute("SELECT papers FROM qa_data WHERE id = ?", (id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()

    if row:
        print(row)
        return json.loads(row[0])

    else:
        return "error: Query not found", 404
    
def get_patient_data(id): #TODO put 4 get fcts in one
    """Fetch the stored response by its ID."""
    conn = get_db_connexion()
    cursor = conn.cursor()
    cursor.execute("SELECT patient_data FROM qa_data WHERE id = ?", (id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()

    if row:
        print(row)
        return json.loads(row[0])

    else:
        return "error: Query not found", 404

@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(error):
    return "The request is too large!", 413

@app.route('/')
def search_query_page():
    return render_template('search_query_page.html')

@app.route("/answer", methods=['GET'])
def answer_page_get(): #TODO same functiomn as asnwer_page()
    # Retrieve the data stored in session
    response_id = session.get('response_id', {})
    #get query and llm answer from db
    llm_answer = get_response(response_id)
    query = get_query(response_id)
    patient_data = get_patient_data(response_id)
    json_patient_data = json.dumps(patient_data, ensure_ascii=False)

    query_results = get_papers(response_id)

    return render_template('answer.html', query=query, query_results=query_results, patient_data=json_patient_data, llm_answer=llm_answer)


@app.route("/answer",  methods=['POST'])
def answer_page():
    form_data = request.form.to_dict(flat=False)  # Converts form data to a dictionary
    
    print("FORM DATA")
    print(form_data)
    # Separate the query from the patient data
    query = (form_data.pop('query', [None])[0]) # Get the query field and remove it from the form data
    patient_data = {key: value[0] if len(value) == 1 else value for key, value in form_data.items()}
    json_patient_data = json.dumps(patient_data, ensure_ascii=False)

    print("PATIENT DATA")
    print(patient_data)
    
    # Render the answer.html template with the data
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_collection(name="searchable_db_collection")
    query_results = get_relevant_papers(query, collection, patient_data)

    response_id = str(uuid.uuid4())
    session['response_id'] = None  # Clear previous response ID safely
    session['response_id'] = response_id
    session.modified = True 
    print("RESPONSE_ID_post")
    print(response_id)
    print("PATIENT DATA2")
    print(patient_data)

    return render_template('answer.html', query=query, query_results = query_results, \
                           patient_data=json_patient_data, response_id=response_id) #TODO check query_results optimization?


# Route for streaming the LLM response
@app.route('/stream_response', methods=['POST'])
def stream_response():  
    try: 
        #get data
        data = request.get_json()
        query = data.get('query', '')
        papers = data.get('papers', {})
        papers_json = json.dumps(papers)
        patient_data = data.get('patient_data', {})
        patient_data_json = json.dumps(patient_data)
        response_id = data.get('response_id', '')
        print("RESPONS_ID_stream")
        print(response_id)

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

            #store response in database
            print("PATIENT DATAAAAAAAA")
            print(patient_data)
            conn = get_db_connexion()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO qa_data (id, query, patient_data, papers, response) VALUES (?, ?, ?, ?, ?)",
                           (response_id, query, patient_data_json, papers_json, full_response))
            conn.commit()
            conn.close()            

        return Response(generate_response(), content_type='text/event-stream',  headers={"Response-ID": response_id})
    
    except Exception as e:
        print(f"Error: {e}")
        return Response("An error occurred while streaming the response.", status=500)

    
@app.route('/paper_<int:paper_id>', methods=['POST', 'GET'])
def view_paper(paper_id):
    if request.method == 'POST':
        paper_title = request.form.get('title')
        paper_abstract = request.form.get('abstract')
        paper_author = request.form.get('author')
        paper_year = request.form.get('year')
        paper_journal = request.form.get('journal')

        return render_template(
            'paper.html', 
            paper={
                "title": paper_title,
                "abstract": paper_abstract,
                "author": paper_author,
                "year": paper_year,
                "journal": paper_journal
            }
        )

if __name__ == "__main__":
    app.run(host='0.0.0.0')
