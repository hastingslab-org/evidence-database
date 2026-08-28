import sqlite3

import config


def init_db():

    connection = sqlite3.connect(str(config.SQLITE_DB_PATH))

    with open(config.RESPONSES_SQL_PATH) as f:
        connection.executescript(f.read())


if __name__ == "__main__":
    init_db()
