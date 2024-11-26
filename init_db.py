import sqlite3

connection = sqlite3.connect('database.db')


with open('queries.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO queries (content) VALUES (?)",
            ('Content for the first query',)
            )

cur.execute("INSERT INTO queries (content) VALUES (?)",
            ('Content for the second query',)
            )

connection.commit()
connection.close()