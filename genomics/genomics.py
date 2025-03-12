
import sqlite3

def get_all_molecular_profiles(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute a query to retrieve all data from the molecular_profiles table
    cursor.execute("SELECT * FROM molecular_profiles")

    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    # Get the column names from the cursor description
    column_names = [description[0] for description in cursor.description]

    # Close the connection
    conn.close()

    # Combine column names with rows
    profiles_with_keys = [dict(zip(column_names, row)) for row in rows]

    return profiles_with_keys



def get_profile_by_name(name, db_path):
    all_molecular_profiles = get_all_molecular_profiles(db_path)

    for profile in all_molecular_profiles:
        if profile['name'] == name:
            return profile
    return None