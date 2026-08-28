import os
import sys

import networkx as nx
import pandas as pd

# Make the project root importable regardless of the process working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

 # Load datasets needed for network query (variantscape)
print("[variantscape] Loading the network graph...")
G = nx.read_gml(str(config.VARIANTSCAPE_GRAPH_PATH))
print("[variantscape] Loading the consensus file...")
df_consensus = pd.read_csv(config.VARIANTSCAPE_CONSENSUS_PATH)
# === Prepare consensus lookup ===
df_consensus["Variant_Treatment_Pair"] = (
df_consensus["Variant_Treatment_Pair"]
.str.strip()
.str.lower()
)
consensus_dict = dict(
    zip(df_consensus["Variant_Treatment_Pair"], df_consensus["Resolved_Prediction"])
)
metadata_mapping = pd.read_csv(config.VARIANTSCAPE_METADATA_MAPPING_PATH, low_memory=False)
