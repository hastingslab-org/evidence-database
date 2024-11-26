from flask import Flask, render_template
from werkzeug.exceptions import abort
import sqlite3 

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_query(query_id):
    conn = get_db_connection()
    query = conn.execute('SELECT * FROM queries WHERE id = ?',
                        (query_id,)).fetchone()
    conn.close()
    if query is None:
        abort(404)
    return query


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'


@app.route('/')
def search_query_page():
    return render_template('search_query_page.html')

@app.route('/<int:query_id>')
def post(query_id):
    query = get_query(query_id)
    return render_template('query.html', query=query)