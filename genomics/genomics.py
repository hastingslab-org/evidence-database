
import sqlite3
import json

def get_all_genes(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to retrieve all data from the molecular_profiles table
    cursor.execute("SELECT * FROM genes")

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    column_names = [description[0] for description in cursor.description]

    # Close the connection
    conn.close()

    # Combine column names with rows
    genes_with_keys = [dict(zip(column_names, row)) for row in rows]

    return genes_with_keys


def get_gene_by_name(name, db_path):
    all_genes = get_all_genes(db_path)

    for gene in all_genes:
        if gene['name'] == name:
            return gene
    return None


def get_diseases_fom_ids(ids, db_path):
    "IDs are provided as a string with comma separated values"
    ids = json.loads(ids)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    diseases = []
    for disease_id in ids:
        cursor.execute("SELECT name FROM diseases WHERE id = ?", (disease_id,))
        disease = cursor.fetchone()
        diseases.append(disease[0])

    conn.close()

    return diseases