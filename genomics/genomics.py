
import sqlite3
import json
from rapidfuzz import fuzz

MATCHING_RATIO_THRESH = 0.7

def get_item_by_name(name, table_name, db_path): #TODO this is by exact name
    all_items = get_all_items(table_name, db_path)
    # Return variant where the search term is found (case-insensitive)
    matching_item = [item  for item in all_items if name.lower() == item['name'].lower()][0]
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


def get_genes_by_name(name, db_path):
    all_genes = get_all_items("genes", db_path)
    # Return genes where the search term is found (case-insensitive)
    matching_genes = [gene for gene in all_genes if name.lower() in gene['name'].lower()]

    # Fuzzy matching
    for gene in all_genes:
        fuzz_ratio = fuzz.ratio(name.lower(), gene['name'].lower())
        if fuzz_ratio > MATCHING_RATIO_THRESH * 100 and gene not in matching_genes:
            matching_genes.append({**gene, 'fuzz_ratio': fuzz_ratio})

    return matching_genes


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