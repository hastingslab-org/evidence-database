import json
import os
import uuid
from flask import Flask, render_template, send_from_directory, request, Response, session, jsonify, abort, current_app
from flask import redirect, url_for
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from llm import call_llm_stream, get_relevant_papers
from init_db import init_db
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
import chromadb
from chromadb.utils import embedding_functions

import config
import literature
import guidelines
from genomics.genomics_data import fuzzy_match_gml, get_items_by_name_fuzzy, get_items_from_ids, get_item_by_name, get_item_from_single_id, check_variant_in_database
from variantscape.variantscape import compute_associations, check_variant_in_graph, check_cancer_in_graph, get_associated_cancer_types_from_variant, get_associated_variants_from_cancer_type
from variantscape.graph_store import G
from variantscape.variantscape import autosuggest_item



# ChromaDB (LiteratureDB RAG) vector store location
DB_PATH = str(config.CHROMA_DB_PATH)

# Variantscape highlight thresholds (see config.py / .env.example to override)
TREATMENT_MIN_HIGHLIGHT = config.TREATMENT_MIN_HIGHLIGHT
CANCER_MIN_HIGHLIGHT = config.CANCER_MIN_HIGHLIGHT

#app config
def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    CORS(app, origins=config.CORS_ORIGINS)  # Enable CORS (configurable allow-list)
    app.config['SESSION_TYPE'] = 'filesystem'  # Use the filesystem to store session data
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['SECRET_KEY'] = config.SECRET_KEY


    @app.context_processor
    def inject_partner_logos():
        # The returned dict is merged into the Jinja context globally
        return {"partner_logos": get_partner_logos()}

    return app

#db access functions
def get_db_connection():
    conn = sqlite3.connect(
        str(config.SQLITE_DB_PATH), timeout=config.SQLITE_BUSY_TIMEOUT
    )
    conn.row_factory = sqlite3.Row
    # WAL lets GenomicsDB reads proceed while an answer is being cached; the
    # busy timeout makes concurrent writers wait instead of raising "locked".
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={int(config.SQLITE_BUSY_TIMEOUT * 1000)}")
    return conn


@lru_cache(maxsize=1)
def get_literature_collection():
    """Build the ChromaDB collection once per process.

    Loading the sentence-transformer embedding model is expensive, so this is
    cached rather than rebuilt on every request.
    """
    client = chromadb.PersistentClient(
        path=DB_PATH,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    return client.get_collection(
        name=config.CHROMA_COLLECTION, embedding_function=embedding_func
    )

def get_qa_row(response_id):
    """Return the cached answer row for ``response_id`` as a dict, or None.

    None means the entry is gone -- a brand-new browser, a session that
    outlived its cache entry, or a pruned row -- and the caller should fall
    back to the search form rather than rendering a half-empty page.
    """
    if not response_id:
        return None
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT id, query, patient_data, papers, response, filters, guidelines "
            "FROM qa_data WHERE id = ?",
            (response_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def _prune_qa_data(cursor):
    """Keep only the most recent ``QA_CACHE_MAX_ROWS`` rows so the cache is bounded."""
    cursor.execute(
        "DELETE FROM qa_data WHERE id NOT IN ("
        "  SELECT id FROM qa_data ORDER BY created_at DESC, rowid DESC LIMIT ?"
        ")",
        (config.QA_CACHE_MAX_ROWS,),
    )


def _store_qa_answer(response_id, query, patient_data_json, papers_json,
                     full_response, filters_raw, guidelines_json):
    """Upsert one answer into the cache and prune it back to its size limit."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO qa_data "
            "(id, query, patient_data, papers, response, filters, guidelines, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (response_id, query, patient_data_json, papers_json, full_response,
             filters_raw, guidelines_json, datetime.now(timezone.utc).isoformat()),
        )
        _prune_qa_data(cursor)
        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Database error:", e)
    finally:
        cursor.close()
        conn.close()

@lru_cache(maxsize=1)
def _partner_links() -> dict:
    """Load the {logo filename: url} map from the configured partners.json."""
    try:
        with open(config.PARTNERS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def get_partner_logos() -> list[tuple[str, str]]:
    """
    Return a sorted list of (filename, url) tuples for the bottom banner.
    """
    partner_dir = os.path.join(current_app.static_folder, "bottom_banner")
    if not os.path.isdir(partner_dir):
        return []

    files = sorted(
        f for f in os.listdir(partner_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))
    )

    # pair each file with its link (or "#" as a benign fallback)
    links = _partner_links()
    return [(f, links.get(f, "#")) for f in files]

app = create_app()
ANGULAR_APP_DIST_DIR = os.path.join(app.static_folder, 'evidence-db-angular')

@app.context_processor
def inject_partner_logos():
    """Automatically inject `partner_logos` into all templates."""
    return dict(partner_logos=get_partner_logos())

########################### app routes ###########################
@app.route('/', methods=['GET'])
def home_page():
    return render_template('home.html')


@app.route('/variantscape', methods=['GET', 'POST'])
def variantscape_page():

    if request.method == 'POST':
        form_data = request.form.to_dict(flat=False)  
        gene_search = form_data["gene"][0]
        variant_search = form_data["variant"][0]
        disease_search = form_data["cancer"][0]

        EXIST_VARIANT_BOOL = check_variant_in_graph(gene_search, variant_search)
        EXIST_CANCER_BOOL = check_cancer_in_graph(disease_search)

        #variant but no cancer type entered
        if EXIST_VARIANT_BOOL and disease_search.strip() == "":
            recommended_cancers = get_associated_cancer_types_from_variant(gene_search, variant_search)
            recommended_cancers = [c.capitalize() for c in recommended_cancers] #cap first letter

            return render_template('variantscape.html', recommended_cancers=recommended_cancers, exist_variant_bool = EXIST_VARIANT_BOOL, \
                                   gene=gene_search.upper(), variant=variant_search.upper())

        #cancer type but no variant entered
        if EXIST_CANCER_BOOL and variant_search.strip() == "" and gene_search.strip() == "":
            recommended_variants = get_associated_variants_from_cancer_type(disease_search)
            recommended_variants = [v.capitalize() for v in recommended_variants][:10]
            recommended_variants = [
                " ".join(reversed(variant.split("_"))).upper() for variant in recommended_variants
            ] #reformat to gene-space-variant format

            return render_template('variantscape.html', recommended_variants=recommended_variants, exist_cancer_bool = EXIST_CANCER_BOOL, \
                                   disease=disease_search.capitalize())

        if not EXIST_VARIANT_BOOL:
            variant_of_interest = (variant_search + "_" + gene_search).lower()
            recommended_variants = [
                " ".join(reversed(variant.split("_"))).upper() for variant in fuzzy_match_gml(variant_of_interest, G.nodes) #TODO integrate fuzzy match fct with autosuggets fcts.
            ]
            return render_template('variantscape.html', recommended_variants=recommended_variants, exist_variant_bool = EXIST_VARIANT_BOOL, \
                                   gene=gene_search, variant=variant_search.upper(), disease = disease_search)

        if not EXIST_CANCER_BOOL:
            cancer_of_interest = disease_search.strip().lower()
            recommended_cancers = autosuggest_item(cancer_of_interest, item_type="Cancer", FUZZY_MATCH = True)
            
            recommended_cancers = [c.capitalize() for c in recommended_cancers] #cap first letter
            return render_template('variantscape.html', recommended_cancers=recommended_cancers, exist_cancer_bool = EXIST_CANCER_BOOL, \
                                   gene=gene_search, variant=variant_search.upper(), disease = disease_search)

        cancer_of_interest, top_sens, top_res, top_var_c, sens_pct, res_pct, cancer_pct, \
            gene_name, variant_name = compute_associations(gene_search, variant_search, disease_search)
        
        return render_template('variantscape.html', top_sens=top_sens, top_res=top_res, top_var_c=top_var_c, \
                               sens_pct=sens_pct, res_pct=res_pct, cancer_pct=cancer_pct, gene=gene_name, \
                                variant=variant_name.upper(), disease = cancer_of_interest, \
                                    treatment_min_highlight = TREATMENT_MIN_HIGHLIGHT,
                                    cancer_min_highlight = CANCER_MIN_HIGHLIGHT, exist_cancer_bool = EXIST_CANCER_BOOL, exist_variant_bool = EXIST_VARIANT_BOOL) #TODO add error handling for empty results
    #TODO pass params more elegantly
    if request.method == 'GET':
        return render_template('variantscape.html')


VALID_TYPES = {"gene", "variant", "cancer"}
@app.get("/item-dictionary/<item_type>")
def item_dictionary(item_type):
    if item_type.lower() not in VALID_TYPES:
        abort(404)
    nodes = [n for n in G.nodes if G.nodes[n]['category'] == item_type.capitalize()]
    if item_type.lower() == "gene":
        nodes = [n for n in G.nodes if G.nodes[n]['category'] == "Variant"]
    elif item_type.lower() == "cancer":
        nodes = [n for n in G.nodes if G.nodes[n]['category'] == item_type.capitalize() and "_" not in n]
    else:
        nodes = [n for n in G.nodes if G.nodes[n]['category'] == item_type.capitalize()]

    gene = request.args.get("gene", "").strip()

    if item_type.lower() == "variant" or item_type.lower() == "gene":
        item_dict = []
        for n in nodes:
            if "_" in n:
                var, g = n.split("_", 1)
                if item_type.lower() == "gene":
                    item_dict.append(g)
                else:  # assume variant
                    if gene and g.lower() != gene.lower():
                        continue
                    item_dict.append(var)
            else:
                item_dict.append(n)
        final_dict = list(dict.fromkeys(item_dict))  # remove duplicates while preserving order

    elif item_type.lower() == "cancer":
        final_dict = list(dict.fromkeys(nodes))
    return jsonify(final_dict)

@app.get("/item-suggestions/<item_type>")
def item_suggestions(item_type):
    
    if item_type.lower() not in VALID_TYPES:
        abort(404)

    q = request.args.get("q", "", type=str)
    gene = request.args.get("gene", "").strip()
    # make sure we don’t hammer the db for empty strings
    if not q.strip():
        return jsonify([])

    suggestions = autosuggest_item(q, item_type, corresponding_gene=gene, FUZZY_MATCH = True)
    
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

@app.route('/variantscape/studydesignclustermap', methods=['GET'])
def clustermap_page():
    return send_from_directory('static', 'Interactive_cluster_map_study_design_type_plot_final.html')


@app.route('/genomics', methods=['GET', 'POST'])
def genomics_page():
    # Accept gene_name from either args or form without branching
    gene_name = request.values.get("gene_name", "").strip()

    if not gene_name:
        # empty page or GET with no query
        return render_template("genomics.html")

    # ----- business logic -----
    genes = get_items_by_name_fuzzy(gene_name, item_name="genes", db_path="database.db")
    gene_data_list = [
        {
            "name": g["name"],
            "description": g["description"],
            "diseases":  get_items_from_ids(g["diseases"],  "diseases",  db_path="database.db"),
            "variants":  get_items_from_ids(g["variants"],  "variants",  db_path="database.db"),
            "molecular_profiles":
                         get_items_from_ids(g["molecular_profiles"], "molecular_profiles",
                                            db_path="database.db"),
        }
        for g in genes
    ]

    # PRG: if the request was POST, redirect to the GET URL
    if request.method == "POST":
        return redirect(url_for("genomics_page", gene_name=gene_name))

    return render_template("genomics.html", genes=gene_data_list)


    
    
@app.route("/genomics/variant/notfound")
def variant_notfound():
    """Display a simple 'variant not found' page."""
    return render_template("variant_notfound.html")

@app.route('/genomics/variant/<string:gene_name>/<string:variant_name>')
def variant_page(gene_name, variant_name):
    if check_variant_in_database(gene_name, variant_name):
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
    
    else:
        # If the variant is not found, redirect to the variantscape page with a message
        return redirect(url_for("variant_notfound"))

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

        # Separate the query and the publication filters from the patient data
        query = (form_data.pop('query', [None])[0]) # Get the query field and remove it from the form data
        filters_raw = form_data.pop('filters', [None])[0]  # active left-panel filters (JSON)
        patient_data = {key: value[0] if len(value) == 1 else value for key, value in form_data.items()}
        json_patient_data = json.dumps(patient_data, ensure_ascii=False)

        # Get papers (retrieval is restricted to the filtered subset when filters are active)
        allowed_ids = literature.matching_ids(filters_raw)
        collection = get_literature_collection()
        query_results = get_relevant_papers(query, collection, patient_data, allowed_ids=allowed_ids)
        overview = literature.overview_stats(collection, ids=allowed_ids)

        # Always anchor the answer with official guideline recommendations
        # (no-op when EVIDENCE_DB_GUIDELINES_ENABLED is false).
        guideline_results = guidelines.get_relevant_guidelines(query, patient_data)

        response_id = str(uuid.uuid4())
        session['response_id'] = None  # Clear previous response ID
        session['response_id'] = response_id
        session.modified = True

        return render_template('answer.html', query=query, query_results=query_results,
                               patient_data=json_patient_data, response_id=response_id,
                               filter_groups=literature.FILTER_GROUPS, overview=overview,
                               active_filters=literature.parse_filters(filters_raw),
                               filters_raw=filters_raw or '',
                               guidelines=guideline_results,
                               guideline_sources=guidelines.all_sources(),
                               guideline_disclaimer=guidelines.disclaimer())

    if request.method == 'GET':
        # Re-render the last answer for THIS browser, keyed by its session id.
        row = get_qa_row(session.get('response_id'))
        if row is None:
            # Nothing cached for this session (new browser / expired / pruned).
            return redirect(url_for('literature_page'))

        query = row['query']
        llm_answer = row['response']
        patient_data = json.loads(row['patient_data'] or '{}')
        json_patient_data = json.dumps(patient_data, ensure_ascii=False)
        query_results = json.loads(row['papers'] or '{}')
        filters_raw = row['filters'] if isinstance(row['filters'], str) else ''
        try:
            guideline_results = json.loads(row['guidelines'] or '[]')
            if not isinstance(guideline_results, list):
                guideline_results = []
        except (ValueError, TypeError):
            guideline_results = []
        allowed_ids = literature.matching_ids(filters_raw)
        overview = literature.overview_stats(get_literature_collection(), ids=allowed_ids)
        return render_template('answer.html', query=query, query_results=query_results,
                               patient_data=json_patient_data, llm_answer=llm_answer,
                               filter_groups=literature.FILTER_GROUPS, overview=overview,
                               active_filters=literature.parse_filters(filters_raw),
                               filters_raw=filters_raw,
                               guidelines=guideline_results,
                               guideline_sources=guidelines.all_sources(),
                               guideline_disclaimer=guidelines.disclaimer())

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
        filters_raw = data.get('filters', '') or ''
        guideline_payload = data.get('guidelines', []) or []
        guidelines_json = json.dumps(guideline_payload)
        guidelines_block = guidelines.format_guidelines_for_prompt(guideline_payload)

        def generate_response():
            # Stream LLM response live
            title_and_abst = ",".join(papers["documents"][0])

            chunks = []
            for chunk in call_llm_stream(query, title_and_abst, patient_data, guidelines_block):
                if chunk:
                    chunks.append(chunk) #collecting chunks
                    yield chunk.encode('utf-8')
            full_response = "".join(chunks)

            # Cache this answer under its own response_id (read back by /answer GET).
            _store_qa_answer(response_id, query, patient_data_json, papers_json,
                             full_response, filters_raw, guidelines_json)

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

#large request handling
@app.errorhandler(RequestEntityTooLarge)
def handle_large_request(error):
    return "The request is too large!", 413


# @app.route('/my-angular-app/')  # Main route for your Angular app
# @app.route('/my-angular-app/<path:path>')  # Catch-all for Angular client-side routing
# def serve_angular_page(path=None):
#     """
#     Serves the host HTML page (angular_host.html) that will bootstrap the Angular app.
#     The 'path' variable captures subpaths for Angular's router but isn't used directly here.
#     """
#     return render_template('angular_host.html')

# @app.route('/my-angular-app-assets/<path:filename>')
# def serve_angular_assets(filename):
#     """
#     Serves static files (JS, CSS, images, etc.) for the Angular application
#     from the ANGULAR_APP_DIST_DIR.
#     """
#     return send_from_directory(ANGULAR_APP_DIST_DIR, filename)

@app.route('/literature', methods=['GET'])
def literature_page():
    """LiteratureDB: filters (left) + patient/query form (centre) + overview (right)."""
    collection = get_literature_collection()
    overview = literature.overview_stats(collection, ids=None)
    return render_template(
        'literature.html',
        filter_groups=literature.FILTER_GROUPS,
        overview=overview,
        guidelines_enabled=config.GUIDELINES_ENABLED,
        guideline_sources=guidelines.all_sources(),
    )


@app.route('/api/literature/overview', methods=['GET'])
def api_literature_overview():
    """Return the overview aggregates + browsable paper list for the active filters."""
    allowed_ids = literature.matching_ids(request.args.get('filters', ''))
    collection = get_literature_collection()
    return jsonify(literature.overview_stats(collection, ids=allowed_ids))


@app.route('/api/guidelines/sources', methods=['GET'])
def api_guideline_sources():
    """List the clinical-guideline issuing bodies used to anchor answers."""
    return jsonify({
        "enabled": config.GUIDELINES_ENABLED,
        "disclaimer": guidelines.disclaimer(),
        "sources": guidelines.all_sources(),
    })


@app.route('/api/guidelines/preview', methods=['GET'])
def api_guideline_preview():
    """Preview the recommendations that would be retrieved for a question."""
    query = request.args.get('q', '', type=str)
    cancer_type = request.args.get('cancer_type', '', type=str) or None
    return jsonify(guidelines.get_relevant_guidelines(query, cancer_type=cancer_type))


# API version of the view database endpoint for Angular
@app.route('/api/view_database', methods=['GET'])
def api_view_database():
    try:
        print("Received request to /api/view_database")
        
        # Connect to the database
        print("Connecting to ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )

        print("Getting collection...")
        collection = chroma_client.get_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=embedding_func
        )
        
        print("Getting items from collection...")
        # Get all items in the collection
        results = collection.get()
        
        print(f"Found {len(results['ids'])} items")
        
        # Format the data for display
        formatted_data = []
        for i in range(len(results["ids"])):
            item = {
                "id": results["ids"][i],
                "document": results["documents"][i]
            }
            
            # Add metadata if available
            if "metadatas" in results and results["metadatas"] and i < len(results["metadatas"]):
                item["metadata"] = results["metadatas"][i]
            
            formatted_data.append(item)
        
        print("Returning JSON response")
        return jsonify({"data": formatted_data})
    
    except Exception as e:
        print(f"Error in /api/view_database: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# API version of the answer endpoint for Angular frontend
@app.route("/api/answer", methods=['POST'])
def api_answer():
    try:
        # Get data from Angular form
        data = request.get_json()
        query = data.get('query', '')
        patient_data = {k: v for k, v in data.items() if k != 'query'}

        # Generate query results
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
        collection = chroma_client.get_collection(name=config.CHROMA_COLLECTION, embedding_function=embedding_func)

        # --- Start Debug Prints ---
        """print(f"--- DEBUG: Attempting to access collection at DB_PATH: {DB_PATH}")
        print(f"--- DEBUG: Collection Name from client: {collection.name}")
        collection_count = collection.count()
        print(f"--- DEBUG: Number of items in collection: {collection_count}")
        if collection_count > 0:
            peeked_item = collection.peek(limit=1)
           

            # --- New Debug: Get item by ID and check its embedding ---
          
        else:
            print(f"--- DEBUG: Collection appears to be empty or could not be loaded correctly.")
        # --- End Debug Prints ---"""

        query_results = get_relevant_papers(query, collection, patient_data)

        guideline_results = guidelines.get_relevant_guidelines(query, patient_data)

        # Generate response ID
        response_id = str(uuid.uuid4())

        # Store session variables
        session['response_id'] = None
        session['response_id'] = response_id
        session.modified = True

        return jsonify({
            "success": True,
            "query": query,
            "queryResults": query_results,
            "guidelines": guideline_results,
            "patientData": patient_data,
            "responseId": response_id
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Route for streaming the LLM response for Angular frontend
@app.route('/api/stream_response', methods=['POST'])
def api_stream_response():  
    try: 
        #get data
        data = request.get_json()
        query = data.get('query', '')
        papers = data.get('papers', {})
        papers_json = json.dumps(papers)
        patient_data = data.get('patient_data', {})
        patient_data_json = json.dumps(patient_data)
        response_id = data.get('response_id', '')
        filters_raw = data.get('filters', '') or ''
        guideline_payload = data.get('guidelines', []) or []
        guidelines_json = json.dumps(guideline_payload)
        guidelines_block = guidelines.format_guidelines_for_prompt(guideline_payload)

        def generate_response():
            # Stream LLM response live
            title_and_abst = ",".join(papers["documents"][0])

            chunks = []
            for chunk in call_llm_stream(query, title_and_abst, patient_data, guidelines_block):
                if chunk:
                    chunks.append(chunk) #collecting chunks
                    yield chunk.encode('utf-8')
            full_response = "".join(chunks)

            # Cache this answer under its own response_id (read back by /answer GET).
            _store_qa_answer(response_id, query, patient_data_json, papers_json,
                             full_response, filters_raw, guidelines_json)

        return Response(generate_response(), content_type='text/event-stream',  headers={"Response-ID": response_id})
    
    except Exception as e:
        print(f"Error: {e}")
        return Response("An error occurred while streaming the response.", status=500)



print("Initializing database...")
init_db()
print("Database initialized.")

if __name__ == "__main__": #TODO separate app.py into init, cli, helpers, and routes
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)