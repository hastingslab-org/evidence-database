import sqlite3

connection = sqlite3.connect('database.db')

with open('queries.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()
cur.execute('''
        CREATE TABLE IF NOT EXISTS llm_responses (
            id TEXT PRIMARY KEY,
            query TEXT,
            response TEXT
        )
    ''')
connection.commit()
connection.close()