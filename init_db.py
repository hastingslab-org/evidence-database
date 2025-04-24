import sqlite3

def init_db():

    connection = sqlite3.connect('database.db')

    with open('responses.sql') as f:
        connection.executescript(f.read())

if __name__ == "__main__":
    init_db()