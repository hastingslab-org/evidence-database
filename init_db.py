import sqlite3

connection = sqlite3.connect('database.db')

with open('responses.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()
cur.execute('''
        CREATE TABLE IF NOT EXISTS qa_data (
            id TEXT PRIMARY KEY,
            query TEXT,
            patient_data TEXT,
            papers TEXT,
            response TEXT
        )
    ''')
connection.commit()
connection.close()