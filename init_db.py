import sqlite3
import networkx as nx
import pandas as pd

def init_db():

    connection = sqlite3.connect('database.db')

    with open('responses.sql') as f:
        connection.executescript(f.read())

    global G, df_consensus #TODO are gloobal variables the best approach? Check session variables
    # Load datasets needed for network query (variantscape)
    print("Loading the network graph...")
    G = nx.read_gml("network_graph_weighted.gml")
    print("Loading the consensus file...")
    df_consensus = pd.read_csv('final_variant_treatment_consensus.csv')


if __name__ == "__main__":
    init_db()