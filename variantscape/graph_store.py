import networkx as nx
import pandas as pd

 # Load datasets needed for network query (variantscape)
print("[variantscape] Loading the network graph...")
G = nx.read_gml("./variantscape/network_graph_weighted.gml") #TODO fix path 
print("[variantscape] Loading the consensus file...")
df_consensus = pd.read_csv('./variantscape/final_variant_treatment_consensus.csv')
# === Prepare consensus lookup ===
df_consensus["Variant_Treatment_Pair"] = (
df_consensus["Variant_Treatment_Pair"]
.str.strip()
.str.lower()
)   
consensus_dict = dict(
    zip(df_consensus["Variant_Treatment_Pair"], df_consensus["Resolved_Prediction"])
)
