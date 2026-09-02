import sqlite3

import config

# Columns that newer app versions expect on qa_data. Databases created by an
# older schema are upgraded in place (no data loss) rather than dropped.
_REQUIRED_QA_COLUMNS = {
    "guidelines": "TEXT",
    "created_at": "TEXT",
}


def init_db():
    connection = sqlite3.connect(str(config.SQLITE_DB_PATH))
    try:
        with open(config.RESPONSES_SQL_PATH) as f:
            connection.executescript(f.read())
        _migrate_qa_data(connection)
        connection.commit()
    finally:
        connection.close()


def _migrate_qa_data(connection):
    """Add any missing columns to a pre-existing qa_data table, then its index."""
    existing = {row[1] for row in connection.execute("PRAGMA table_info(qa_data)")}
    for column, decl in _REQUIRED_QA_COLUMNS.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE qa_data ADD COLUMN {column} {decl}")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_qa_data_created_at ON qa_data (created_at)"
    )


if __name__ == "__main__":
    init_db()
