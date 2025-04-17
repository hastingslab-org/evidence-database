import networkx as nx
import pandas as pd


# Load datasets needed for network query
G_w = nx.read_gml("network_graph_weighted.gml")
G = G_w.copy()


print("Loading the consensus file...")
df_consensus_path = variantscape_LLM_coas_directory + '/final_variant_treatment_consensus.csv'
    df_consensus = pd.read_csv(df_consensus_path)

    

    
# Define variant and cancer of interest (with aliasing)
user_input_cancer = "NSCLC"
#variant_of_interest = "v600e_BRAF"

############## Variants of interest for paper ##############
variant_of_interest = 'l858r_EGFR' #as durggable usecase
#variant_of_interest = 't790m_EGFR' #as resistant usecase

######################## RARE VARIANTS #############################
#variant_of_interest = 'g469v_BRAF'
#variant_of_interest = 'g719x_EGFR' # does not exist in network
#variant_of_interest = 's768i_EGFR'
#variant_of_interest = 'l861q_EGFR'
#variant_of_interest = 'l747p_EGFR' # no information in the network


# Define alias mapping for cancer names
cancer_alias_map = {
    "nsclc": "lung cancer",
    "non-small cell lung cancer": "lung cancer",
    "tnbc": "breast cancer",
    "her2+ breast cancer": "breast cancer"
}

# Normalize user input
clean_input = user_input_cancer.strip().lower()

# Use alias if it exists, otherwise just use the same name
cancer_of_interest = cancer_alias_map.get(clean_input, clean_input)
display_cancer_name = user_input_cancer  # Always show the original user input

print(f"\n\n\033[1mCancer of interest set to:\033[0m {display_cancer_name} (cancer type:'{cancer_of_interest}')")
print(f"\033[1mVariant of interest set to:\033[0m {variant_of_interest}")