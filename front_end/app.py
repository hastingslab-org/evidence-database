from flask import Flask, render_template, request, url_for, flash, redirect
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


@app.route('/search', methods=('GET', 'POST'))
def user_search_page():
    if request.method == 'POST':
        # Get the search query from the form
        content = request.form['query']
        
        # Insert the query into the database
        conn = get_db_connection()
        conn.execute('INSERT INTO queries (content) VALUES (?)', (content,))
        conn.commit()
        
        # Fetch only the query that was just inserted
        query = conn.execute('SELECT * FROM queries WHERE content = ? ORDER BY created DESC LIMIT 1', (content,)).fetchone()
        conn.close()
        
        # Pass only this query to the template
        return render_template('search.html', queries=[query])
    
    # If the request is not POST, render an empty page or handle GET requests
    return render_template('search.html', queries=[])
