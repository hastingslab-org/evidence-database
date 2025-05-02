import json
import uuid
from flask import Flask, render_template, send_from_directory, request, Response, session, jsonify, abort
from werkzeug.exceptions import RequestEntityTooLarge
from llm import call_llm_stream, get_relevant_papers
from init_db import init_db
import sqlite3 
import chromadb
from chromadb.utils import embedding_functions

from genomics.genomics_data import fuzzy_match_gml, get_items_by_name_fuzzy, get_items_from_ids, get_item_by_name, get_item_from_single_id
from variantscape.variantscape import compute_associations, check_variant_in_graph, check_cancer_in_graph
from variantscape.graph_store import G
from variantscape.variantscape import autosuggest_item

# Get db directory path
DB_PATH = 'chroma_data2'

#Variantscape config
TREATMENT_MIN_HIGHLIGHT        = 300   # and require ≥X total weight
CANCER_MIN_HIGHLIGHT           = 80    # and require ≥X total weight
#TODO add percentile and fuzzy ratio (currently in other files). Move to a config file ? 

#app config
app = Flask(__name__)

app.jinja_options = app.jinja_options.copy()

app.config['SESSION_TYPE'] = 'filesystem'  # Use the filesystem to store session data
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'my_secret_key' #TODO 

#db access functions
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_qa_item(item_name, item_id, json_load=False):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT " + item_name + " FROM qa_data WHERE id = ?", (item_id,))
    row = cursor.fetchone()
    conn.commit()
    conn.close()

    if row:
        if json_load:
            return json.loads(row[0])
        else:
            return row[0]
    else:
        return jsonify({"error": "Response not found"}), 404

#large request handling
@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(error):
    return "The request is too large!", 413

########################### app routes ###########################
@app.route('/', methods=['GET'])
def home_page():
    return render_template('home.html')

@app.route('/literature', methods=['GET'])
def search_query_page():
    return render_template('search_query_page.html')

@app.route('/variantscape', methods=['GET', 'POST'])
def variantscape_page():

    if request.method == 'POST':
        form_data = request.form.to_dict(flat=False)  
        gene_search = form_data["gene"][0]
        variant_search = form_data["variant"][0]
        disease_search = form_data["disease"][0]

        EXIST_VARIANT_BOOL = check_variant_in_graph(gene_search, variant_search)
        EXIST_CANCER_BOOL = check_cancer_in_graph(disease_search)

        if not EXIST_VARIANT_BOOL:
            variant_of_interest = (variant_search + "_" + gene_search).lower()
            recommended_variants = [
                " ".join(reversed(variant.split("_"))) for variant in fuzzy_match_gml(variant_of_interest, G.nodes)
            ]
            return render_template('variantscape.html', recommended_variants=recommended_variants, exist_variant_bool = EXIST_VARIANT_BOOL, \
                                   gene=gene_search, variant=variant_search, disease = disease_search)

        if not EXIST_CANCER_BOOL:
            cancer_of_interest = disease_search.strip().lower()
            recommended_cancers = [
                name for name in fuzzy_match_gml(cancer_of_interest, G.nodes)
            ]
            return render_template('variantscape.html', recommended_cancers=recommended_cancers, exist_cancer_bool = EXIST_CANCER_BOOL, \
                                   gene=gene_search, variant=variant_search, disease = disease_search)

        top_sens, top_res, top_var_c, sens_pct, res_pct, cancer_pct, \
            gene_name, variant_name = compute_associations(gene_search, variant_search, disease_search)
        
        return render_template('variantscape.html', top_sens=top_sens, top_res=top_res, top_var_c=top_var_c, \
                               sens_pct=sens_pct, res_pct=res_pct, cancer_pct=cancer_pct, gene=gene_name, \
                                variant=variant_name, disease = disease_search, \
                                    treatment_min_highlight = TREATMENT_MIN_HIGHLIGHT,
                                    cancer_min_highlight = CANCER_MIN_HIGHLIGHT) #TODO add error handling for empty results
    #TODO pass params more elegantly
    if request.method == 'GET':
        return render_template('variantscape.html')


VALID_TYPES = {"gene", "variant", "cancer"}
@app.get("/item-suggestions/<item_type>")
def item_suggestions(item_type):
    
    if item_type.lower() not in VALID_TYPES:
        abort(404)

    q = request.args.get("q", "", type=str)
    gene = request.args.get("gene", "").strip()
    print("gene", gene)
    # make sure we don’t hammer the db for empty strings
    if not q.strip():
        return jsonify([])

    suggestions = autosuggest_item(q, item_type, corresponding_gene=gene)
    
    if item_type.lower() == "variant":
        new_suggestions = []
        for s in suggestions:
            if "_" in s:
                var, g = s.split("_", 1)
                if item_type.lower() == "gene":
                    new_suggestions.append(g)  
                else:  # assume variant
                    if gene and g.lower() != gene.lower():
                        continue
                    new_suggestions.append(var)
            else:
                new_suggestions.append(s)
        suggestions = list(dict.fromkeys(new_suggestions))

    return jsonify(suggestions[:5])

@app.route('/variantscape/networkgraph', methods=['GET'])
def networkgraph_page():
    return send_from_directory('static', 'variantscape_network_graph.html')

@app.route('/genomics', methods=['GET', 'POST'])
def genomics_page():
    if request.method == 'POST':
        form_data = request.form.to_dict(flat=False)  # Converts form data to a dictionary
        gene_name = form_data["gene_name"][0]
        # Get gene info
        genes = get_items_by_name_fuzzy(gene_name, item_name="genes", db_path="database.db")
        gene_data_list = []
        for gene in genes:
            gene_data = {
                "name": gene["name"],
                "description": gene["description"],
                "diseases": get_items_from_ids(gene["diseases"], "diseases", db_path="database.db"),
                "variants": get_items_from_ids(gene["variants"], "variants", db_path="database.db"),
                "molecular_profiles": get_items_from_ids(gene["molecular_profiles"], "molecular_profiles", db_path="database.db")
            }
            gene_data_list.append(gene_data)

        return render_template('genomics.html', genes=gene_data_list)

    if request.method == 'GET':
        return render_template('genomics.html')

@app.route('/genomics/variant/<string:variant_name>')
def variant_page(variant_name):
    variant= get_item_by_name(variant_name, table_name="variants", db_path="database.db")
    #get variant info
    variant_data = {
        "name": variant["name"],
        "description": variant["description"],
        "gene": get_item_from_single_id(variant["gene_id"], "genes", db_path="database.db")[0],
        "molecular_profiles": get_items_from_ids(variant["molecular_profiles"], "molecular_profiles", db_path="database.db"),
        "diseases": get_items_from_ids(variant["diseases"], "diseases", db_path="database.db"),
    }

    return render_template('variant.html', variant_data=variant_data)

@app.route('/genomics/molecular-profile/<string:mp_name>')
def molecular_profile_page(mp_name):
    mp = get_item_by_name(mp_name, table_name="molecular_profiles", db_path="database.db")

    #get mp info
    mp_data = {
        "name": mp["name"],
        "description": mp["description"],
        "variants": get_items_from_ids(mp["variants"], "variants", db_path="database.db"),
        "diseases": get_items_from_ids(mp["disease"], "diseases", db_path="database.db"),
        "score": mp["molecularProfileScore"],   
        }

    return render_template('molecular_profile.html', mp_data=mp_data)

@app.route("/answer",  methods=['POST', 'GET'])
def answer_page():
    if request.method == 'POST':
        form_data = request.form.to_dict(flat=False)  # Converts form data to a dictionary

        # Separate the query from the patient data
        query = (form_data.pop('query', [None])[0]) # Get the query field and remove it from the form data
        patient_data = {key: value[0] if len(value) == 1 else value for key, value in form_data.items()}
        json_patient_data = json.dumps(patient_data, ensure_ascii=False)
        
        # Get papers
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="mixedbread-ai/mxbai-embed-large-v1") #TODO move somewhere else
        collection = chroma_client.get_collection(name="searchable_db_collection",embedding_function=embedding_func)
        query_results = get_relevant_papers(query, collection, patient_data)

        response_id = str(uuid.uuid4())
        session['response_id'] = None  # Clear previous response ID
        session['response_id'] = response_id
        session.modified = True 

        return render_template('answer.html', query=query, query_results = query_results, \
                           patient_data=json_patient_data, response_id=response_id)
    
    if request.method == 'GET':
        # Retrieve the data stored in session
        response_id = session.get('response_id', {})
        #get query and llm answer from db
        llm_answer = get_qa_item("response", response_id)
        query = get_qa_item("query", response_id)
        patient_data = get_qa_item("patient_data", response_id, json_load=True)
        json_patient_data = json.dumps(patient_data, ensure_ascii=False)
        query_results = get_qa_item("papers", response_id, json_load=True)
        return render_template('answer.html', query=query, query_results=query_results, \
                               patient_data=json_patient_data, llm_answer=llm_answer)


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
       
        def generate_response():
            # Stream LLM response live
            title_and_abst = ",".join(papers["documents"][0])
            
            chunks = []
            for chunk in call_llm_stream(query, title_and_abst, patient_data):
                if chunk:
                    chunks.append(chunk) #collecting chunks
                    yield chunk.encode('utf-8')
            full_response = "".join(chunks)

            #store response in db
            print("WRITING TO DB")
            conn = get_db_connection()  # Ensure this function is correctly defined
            cursor = conn.cursor()
            try:
                # Delete all rows from qa_data
                cursor.execute("DELETE FROM qa_data")

                # Insert new data
                cursor.execute(
                    "INSERT INTO qa_data (id, query, patient_data, papers, response) VALUES (?, ?, ?, ?, ?)",
                    (response_id, query, patient_data_json, papers_json, full_response)
                )

                conn.commit()  # Commit the transaction

            except Exception as e:
                conn.rollback()  # Rollback on error to prevent partial updates
                print("Database error:", e)

            finally:
                cursor.close()  # Close cursor
                conn.close()  # Close connection         

        return Response(generate_response(), content_type='text/event-stream',  headers={"Response-ID": response_id})
    
    except Exception as e:
        print(f"Error: {e}")
        return Response("An error occurred while streaming the response.", status=500)

    
@app.route('/paper_<int:paper_id>', methods=['POST'])
def view_paper(paper_id):
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

# Initialize the database
print("Initializing database...")
init_db()
print("Database initialized.")

if __name__ == "__main__": #TODO separate app.py into init, cli, and routes
    app.run(host='0.0.0.0')
