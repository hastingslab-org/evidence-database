
import sqlite3

def get_all_molecular_profiles():
    # Connect to the SQLite database
    conn = sqlite3.connect('../database.db')
    cursor = conn.cursor()

    # Execute a query to retrieve all data from the molecular_profiles table
    cursor.execute("SELECT * FROM molecular_profiles")

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Close the connection
    conn.close()

    return rows


def get_profile_by_name(name):

    all_molecular_profiles = get_all_molecular_profiles()

    for profile in all_molecular_profiles:
        if profile['node']['name'] == name:
            return profile['node']
    return None