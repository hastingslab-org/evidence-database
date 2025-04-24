import requests
import json
import sqlite3
from rapidfuzz import fuzz
import os


#TODO separate into different files (one for APIs interaction, one for sqlite interaction))

#list of variations to exclude when querying CIVIC
VARIATIONS_TO_EXCLUDE = [
    "activation",
    "allele",
    "alu",
    "alteration",
    "alternative",
    "amplification",
    "and",
    "conserve",
    "deficient",
    "deletion",
    "depletion",
    "demethylation",
    "domain",
    "double",
    "duplication",
    "expression",
    "fas",
    "function",
    "fusion",
    "gain",
    "inactivation",
    "increase",
    "insert",
    "knockdown",
    "loss",
    "local",
    "methylation",
    "mut",
    "overexpression",
    "phosphorylation",
    "polymorphisme",
    "positive",
    "promoter",
    "rearrangement",
    "repeat",
    "transcripts",
    "translocation",
    "underexpression",
    "upregulation",
    "shift",
    "variant",
    "variation",
    "wild",
    "P772_H773insH", 
    "nm0", 
    "fas"]

# Threshold for sqlite query fuzzy matching
MATCHING_RATIO_THRESH = 0.8

""" ***** USEFUL FUNCTIONS TO QUERY SQLITE DATABASE *****"""
def get_item_by_name(name, table_name, db_path): #TODO this is by exact name
    all_items = get_all_items(table_name, db_path)
    # Return variant where the search term is found (case-insensitive)
    matching_items = [item for item in all_items if name.lower() == item['name'].lower()]
    if matching_items:
        matching_item = matching_items[0]
    else:
        matching_item = None 
    return matching_item


def get_all_items(table_name, db_path): #TODO merge with get_genes_by_name
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to retrieve all data from the molecular_profiles table
    cursor.execute("SELECT * FROM " + table_name)

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    column_names = [description[0] for description in cursor.description]

    # Close the connection
    conn.close()

    # Combine column names with rows
    items_with_keys = [dict(zip(column_names, row)) for row in rows]

    return items_with_keys

def fuzzy_match_sql(user_input, all_items):

    #sub-string match
    matching_items = [item for item in all_items if user_input.lower() in item['name'].lower()]

    # Fuzzy matching
    for item in all_items:
        fuzz_ratio = fuzz.ratio(user_input.lower(), item['name'].lower())
        if fuzz_ratio > MATCHING_RATIO_THRESH * 100 and item not in matching_items:
            matching_items.append({**item, 'fuzz_ratio': fuzz_ratio})  #TODO overkill if we never need to access the fuzz_ratio

    return matching_items

def fuzzy_match_gml(user_input, all_names): #TODO find a better strategy and merge with fuzzy_match_sql
    # sub-string match
    matching_names = [name for name in all_names if user_input.lower() in name.lower()]

    # Fuzzy matching
    for name in all_names:
        print("name", name)
        fuzz_ratio = fuzz.ratio(user_input.lower(), name.lower())
        if fuzz_ratio > MATCHING_RATIO_THRESH * 100 and name not in matching_names:
            matching_names.append(name)

    return matching_names

def get_items_by_name_fuzzy(name, item_name, db_path):
    all_items = get_all_items(item_name, db_path)
    # Return genes where the search term is found (case-insensitive)

    matching_items = fuzzy_match_sql(name, all_items)
    return matching_items


def get_item_from_single_id(id, item_name, db_path):
    "IDs are provided as a string with comma separated values. element_name is the name of the table to query (string)" 
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM " + item_name + " WHERE id = ?", (id,))
    item = cursor.fetchone()

    return item

def get_items_from_ids(ids, item_name, db_path):
    "IDs are provided as a string with comma separated values. element_name is the name of the table to query (string)" 
    ids = json.loads(ids)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    items = []
    for item_ids in ids:
        cursor.execute("SELECT name FROM " + item_name + " WHERE id = ?", (item_ids,))
        item = cursor.fetchone()
        if item is not None:
            items.append(item[0])

    conn.close()

    return items


""" ***** Class to interact with cloud databases through API (CIVIC, ClinVar) *****"""
class GenomicsData:
    def __init__(self, url="https://civicdb.org/api/graphql", headers = {"Content-Type": "application/json",}):
        self.url = url
        self.headers = headers

        self.variants = None
        self.genes = None
        self.diseases = None
        self.molecular_profiles = None

    def fetch_all_data(self):
        self.fetch_variants()
        self.fetch_genes()
        self.fetch_disease()
        self.fetch_molecular_profiles()

    def fetch_variants(self):
        #Fetch Variants
        print("Fetching variants...")
        query = """
            query browseVariants($after: String) {
            variants(first: 300, after: $after) {
                nodes {
                    id
                    name
                    feature {
                        id
                    }
                    molecularProfiles {
                        nodes {
                            id
                            name
                            description
                            evidenceItems {
                                nodes {
                                    id
                                    name
                                    disease {
                                        id
                                        name
                                    }
                                }
                            }
                        }
                    }
                }
                pageInfo {
                endCursor
                hasNextPage
                }
                totalCount
            }
        }
        """
        all_variants = []
        variables = {"after": None}
        while True:
            response = requests.post(self.url, json={'query': query, 'variables': variables}, headers=self.headers)
            response_json = response.json()
            
            if 'data' in response_json:
                variants = response_json["data"]["variants"]["nodes"]
                all_variants.extend(variants)
                
                page_info = response_json["data"]["variants"]["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                variables["after"] = page_info["endCursor"]
            else:
                print("Error in response:", response_json.get('errors'))
                break

        print(f"Total variants fetched: {len(all_variants)}")

        filtered_variants = [
            variant for variant in all_variants
            if not any(exclusion in variant['name'].lower() for exclusion in VARIATIONS_TO_EXCLUDE)
        ]

        self.variants = filtered_variants

    def fetch_genes(self):
        print("Fetching genes...")
        query = """
        query browseGenes($after: String) {
            genes(first: 300, after: $after) {
                nodes {
                    id
                    name
                    description
                    variants {
                        nodes {
                            id
                            name
                            molecularProfiles {
                                nodes {
                                    id
                                    name
                                    description
                                    evidenceItems {
                                        nodes {
                                            id
                                            name
                                            disease {
                                                id
                                                name
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                pageInfo {
                    endCursor
                    hasNextPage
                }
                totalCount
            }
        }
        """
        all_genes = []
        variables = {"after": None}

        while True:
            response = requests.post(self.url, json={'query': query, 'variables': variables}, headers=self.headers)
            response_json = response.json()
            
            if 'data' in response_json:
                genes = response_json["data"]["genes"]["nodes"]
                all_genes.extend(genes)
                
                page_info = response_json["data"]["genes"]["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                variables["after"] = page_info["endCursor"]
            else:
                print("Error in response:", response_json.get('errors'))
                break

        print(f"Total genes fetched: {len(all_genes)}")
        self.genes = all_genes

    def fetch_disease(self):
        print("Fetching diseases...")
        query = """
            query browseDiseases($after: String) {
            diseases(first: 300, after: $after) {
                nodes {
                    id
                    name
                }  
                pageInfo {
                endCursor
                hasNextPage
                }
                totalCount
            }
            }
            """

        all_diseases = []
        variables = {"after": None}

        while True:
            response = requests.post(self.url, json={'query': query, 'variables': variables}, headers=self.headers)
            response_json = response.json()
            
            if 'data' in response_json:
                diseases = response_json["data"]["diseases"]["nodes"]
                all_diseases.extend(diseases)
                
                page_info = response_json["data"]["diseases"]["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                variables["after"] = page_info["endCursor"]
            else:
                print("Error in response:", response_json.get('errors'))
                break

        print(f"Total diseases fetched: {len(all_diseases)}")
        self.diseases = all_diseases

    def fetch_molecular_profiles(self):
        print("Fetching molecular profiles...")

        query = """
            query browseMolecularProfiles($after: String) {
            molecularProfiles(first: 300, after: $after) {
                edges {
                node {
                    id
                    name
                    description
                    molecularProfileScore
                    variants {
                    id
                    name
                    feature {
                        id
                        name
                    }
                    }
                    assertions {
                    nodes{
                        id
                        name
                        description
                        disease{
                        id
                        name
                        } 
                    }
                    } 
                }
                }
                pageInfo {
                endCursor
                hasNextPage
                }
                totalCount
            }
        }
        """

        all_molecular_profiles = []
        variables = {"after": None}

        while True:
            response = requests.post(self.url, json={'query': query, 'variables': variables}, headers=self.headers)
            response_json = response.json()
            
            if 'data' in response_json:
                molecular_profiles = response_json["data"]["molecularProfiles"]["edges"]
                all_molecular_profiles.extend(molecular_profiles)
                
                page_info = response_json["data"]["molecularProfiles"]["pageInfo"]
                if not page_info["hasNextPage"]:
                    break
                variables["after"] = page_info["endCursor"]
            else:
                print("Error in response:", response_json.get('errors'))
                break

        print(f"Total profiles fetched: {len(all_molecular_profiles)}")

        #filter out MP with score 0 and exclude variations
        molecular_profiles_filtered = [edge for edge in all_molecular_profiles if edge["node"]["molecularProfileScore"] != 0]

        molecular_profiles_filtered = [
            mp for mp in molecular_profiles_filtered
            if not any(exclusion in mp["node"]['name'].lower() for exclusion in VARIATIONS_TO_EXCLUDE)
        ]

        print(f"Total filtered profiles: {len(molecular_profiles_filtered)}")

        self.molecular_profiles = molecular_profiles_filtered

    def save_to_sqlite(self, db_path="../database.db"):
        # Connect to the SQLite database 
        conn = sqlite3.connect(db_path)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sql_file_path = os.path.join(script_dir, '..', 'genomics.sql')
        with open(sql_file_path) as f:
                conn.executescript(f.read())

        cursor = conn.cursor()

        # Store variants
        for variant in self.variants:
            mp_ids      = []
            disease_ids = []

            for profile in variant["molecularProfiles"]["nodes"]:
                mp_ids.append(profile["id"])
                for evidenceItem in profile["evidenceItems"]["nodes"]:
                    if evidenceItem["disease"] and evidenceItem["disease"]["id"]:
                        disease_ids.append(evidenceItem["disease"]["id"])
            
            disease_ids = json.dumps(list(set(disease_ids)))  # Remove duplicates and convert to JSON format
            mp_ids = json.dumps(list(set(mp_ids)))  # Remove duplicates and convert to JSON format
            cursor.execute('''
            INSERT OR REPLACE INTO variants (id, name, gene_id, molecular_profiles, diseases, db_source)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (variant['id'], variant['name'], variant["feature"]["id"], mp_ids, disease_ids, "civic"))

        # Commit the transaction and close the connection
        conn.commit()  

        #store genes
        for node in self.genes:
            disease_ids = []
            variant_ids = []
            mp_ids      = []
            for variant in node["variants"]["nodes"]:
                variant_ids.append(variant["id"])
                for profile in variant["molecularProfiles"]["nodes"]:
                    mp_ids.append(profile["id"])
                    for evidenceItem in profile["evidenceItems"]["nodes"]:
                        if evidenceItem["disease"] and evidenceItem["disease"]["id"]:
                            disease_ids.append(evidenceItem["disease"]["id"])
            disease_ids = json.dumps(list(set(disease_ids)))  # Remove duplicates and convert to JSON format
            variant_ids = json.dumps(list(set(variant_ids)))  # Remove duplicates and convert to JSON format
            mp_ids = json.dumps(list(set(mp_ids)))  # Remove duplicates and convert to JSON format

            cursor.execute('''
            INSERT OR REPLACE INTO genes (id, name, description, molecular_profiles, variants, diseases, db_source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (node['id'], node['name'], node['description'], mp_ids, variant_ids, disease_ids, "civic"))

        # Commit the transaction and close the connection
        conn.commit() 


        #store diseases
        for disease in self.diseases:
            cursor.execute('''
            INSERT OR REPLACE INTO diseases (id, name, db_source)
            VALUES (?, ?, ?)
            ''', (disease['id'], disease['name'], "civic"))

        # Commit the transaction and close the connection
        conn.commit()         


        #store molecular profiles
        for profile in self.molecular_profiles:
            disease_ids = []
            variant_ids = []
            node = profile['node']
            for variant in node["variants"]:
                variant_ids.append(variant["id"])
            
            for assertion in node["assertions"]["nodes"]:
                if assertion["disease"] and assertion["disease"]["id"]:
                    disease_ids.append(assertion["disease"]["id"])
            disease_ids = json.dumps(list(set(disease_ids)))  # Remove duplicates
            variant_ids = json.dumps(list(set(variant_ids)))  # Remove duplicates and convert to JSON format
            cursor.execute('''
            INSERT OR REPLACE INTO molecular_profiles (id, name, description, variants, disease, molecularProfileScore, db_source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (node['id'], node['name'], node['description'], variant_ids, disease_ids,  node['molecularProfileScore'], "civic"))

        # Commit the transaction and close the connection
        conn.commit()
        conn.close()