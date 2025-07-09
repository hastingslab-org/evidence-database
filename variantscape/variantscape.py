import numpy as np
import sys
import os
import pandas as pd
from .graph_store import metadata_mapping
from rapidfuzz import fuzz
import ast
import re


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_store import G, consensus_dict

# === Adjustable thresholds === #
TREATMENT_THRESHOLD_PERCENTILE = 80    # highlight top X% of treatment weights
CANCER_THRESHOLD_PERCENTILE    = 80    # highlight top X% of cancer–variant weights

MATCHING_RATIO_THRESH = 0.8 # threshold for fuzzy matching TODO in config/common file

EXCLUDED_TREATMENTS = {
    'chemotherapy', 'tyrosine kinase inhibitor', 'radiotherapy', 'hormone therapy',
    'adjuvant chemotherapy', 'immunotherapy', 'immune checkpoint inhibitor',
    'mrna vaccine', 'mtor inhibitor', 'radiation ionizing radiotherapy'
}

CANCER_SYNONYM_MAP = {
    "carcinoma": "cancer",
    "sarcoma": "cancer",
    "tumor": "cancer",
    "tumour": "cancer",
    "neoplasm": "cancer",
    "malignancy": "cancer",
    "lesion": "cancer"
}

def compute_cancer_alias_map():
    # Load and normalize the synonym dataframe
    CIVIC_cancer_synonyms_df = pd.read_csv("static/Network_cancer_synonyms.csv")
    df_syn = CIVIC_cancer_synonyms_df.copy()
    df_syn["name"] = df_syn["name"].str.strip().str.lower()

    def safe_parse_synonyms(x):
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return [s.strip().lower() for s in parsed if isinstance(s, str)]
            except Exception as e:
                print(f"Warning: could not parse synonym entry '{x}': {e}")
        return []

    df_syn["synonyms"] = df_syn["synonyms"].apply(safe_parse_synonyms)

    # Build the alias map: synonym → canonical cancer name
    cancer_alias_map = {}
    for _, row in df_syn.iterrows():
        canonical = row["name"]
        for synonym in row["synonyms"]:
            cancer_alias_map[synonym] = canonical

        #get cancer synonyms in right format: 
    synonyms = CIVIC_cancer_synonyms_df["synonyms"].to_numpy()
    syn_series = pd.Series(synonyms)        
    syn_series = syn_series.dropna()
    syn_series = syn_series.map(
        lambda x:                                            # x is one cell
            ast.literal_eval(x) if isinstance(x, str) and x.lstrip().startswith('[')
            else [x]                                         # keep singletons like '0'
    )
    synonyms_flat = syn_series.explode().astype(str)         
    synonyms_array = synonyms_flat.to_numpy()                
    synonyms_array_unique = pd.unique(synonyms_array)
    print("aliases", cancer_alias_map)

    return cancer_alias_map, synonyms_array_unique

#TODO define as object with methods
CANCER_ALIAS_MAP, SYNONYM_ARRAY = compute_cancer_alias_map()

def normalise_cancer_name(raw: str) -> str:
    """
    → Lower-case, trim, swap synonyms for 'cancer',
      then apply the alias map if present.
    """
    # basic cleanup
    name = raw.strip().lower()

    direct_alias = CANCER_ALIAS_MAP.get(name)
    print("DIRECT ALIAS", direct_alias)
    if direct_alias is not None:
        return direct_alias

    # synonym-to-'cancer' pass  (word boundaries ⇒ no 'adenocarcinoma' → 'adenocancer')
    for syn, replacement in CANCER_SYNONYM_MAP.items():
        pattern = rf"\b{re.escape(syn)}\b"
        name = re.sub(pattern, replacement, name)
    name = " ".join(name.split())

    print("NAME", name)

    return CANCER_ALIAS_MAP.get(name, name)

#TODO create a Graph class that contains the following methods
def check_variant_in_graph(user_gene_name, user_variant_name):
    """Check if the variant exists in the graph."""
    variant_of_interest = (user_variant_name + "_" + user_gene_name).lower()
    lowercase_mapping = {key.lower(): key for key in G.nodes}  # Create a temporary mapping of lowercase keys to original keys
    variant_of_interest = lowercase_mapping.get(variant_of_interest)
    return variant_of_interest in G.nodes

def check_cancer_in_graph(user_cancer_name):
    """Check if the cancer exists in the graph."""
    cancer_of_interest = normalise_cancer_name(user_cancer_name)
    return cancer_of_interest in G.nodes


def get_associated_cancer_types_from_variant(gene_name, variant_name):
    """Get associated cancer types from the variant."""
    variant_of_interest = (variant_name + "_" + gene_name).lower()
    lowercase_mapping = {key.lower(): key for key in G.nodes}  # Create a temporary mapping of lowercase keys to original keys
    variant_of_interest = lowercase_mapping.get(variant_of_interest) #TODO do this block only once in app.py ?

    var_nei = set(G.neighbors(variant_of_interest))
    var_cancers = [
        n for n in var_nei
        if G.nodes[n]['category']=='Cancer'
        and n != variant_of_interest
    ]

    vc_weights = {}
    for c in var_cancers:
        w_v = G[variant_of_interest][c]['weight']
        vc_weights[c] = w_v
    sorted_var_c = [key.capitalize() for key, _ in sorted(vc_weights.items(), key=lambda x: x[1], reverse=True)]
    return sorted_var_c

def get_associated_variants_from_cancer_type(cancer_name):
    """Get associated variants from cancer type."""
    cancer_of_interest = normalise_cancer_name(cancer_name)

    cancer_nei = set(G.neighbors(cancer_of_interest))
    associated_variants = [
        n for n in cancer_nei
        if G.nodes[n]['category']=='Variant'
    ]

    vc_weights = {}
    for v in associated_variants:
        w = G[cancer_of_interest][v]['weight']
        vc_weights[v] = w

    sorted_var = [key.capitalize() for key, _ in sorted(vc_weights.items(), key=lambda x: x[1], reverse=True)]
    return sorted_var

def compute_associations(gene_name, variant_name, cancer_name):  
    top_sens, top_res, top_var_c, sens_pct, res_pct, cancer_pct = \
        None, None, None, None, None, None

    variant_of_interest = (variant_name + "_" + gene_name).lower()
    lowercase_mapping = {key.lower(): key for key in G.nodes}  # Create a temporary mapping of lowercase keys to original keys
    variant_of_interest = lowercase_mapping.get(variant_of_interest) #TODO do this block only once in app.py ?
    
    cancer_of_interest = normalise_cancer_name(cancer_name)

    # === Step 1: Cancer‐only treatments ===
    canc_nei = set(G.neighbors(cancer_of_interest))
    treatments = [
        n for n in canc_nei
        if G.nodes[n]['category']=='Treatment'
        and n.lower() not in EXCLUDED_TREATMENTS
    ]

    # === Step 2: Variant + cancer associations ===
    sensitive, resistant = [], []
    for t in treatments:
        try:
            w = G[cancer_of_interest][t]['weight'] + G[variant_of_interest][t]['weight']
            pred = consensus_dict.get(f"{variant_of_interest} + {t}".lower())
            if pred == "Sensitive":
                sensitive.append((t, w))
            elif pred == "Resistant":
                resistant.append((t, w))
        except KeyError:
            continue

    top_sens = sorted(sensitive, key=lambda x: x[1], reverse=True)[:6]
    top_res  = sorted(resistant, key=lambda x: x[1], reverse=True)[:6]
    sens_w = [w for _, w in sensitive]
    res_w  = [w for _, w in resistant]
    sens_pct = np.percentile(sens_w, TREATMENT_THRESHOLD_PERCENTILE) if sens_w else 0
    res_pct  = np.percentile(res_w,   TREATMENT_THRESHOLD_PERCENTILE) if res_w else 0

    # === Step 3: Other cancers for variant ===
    var_nei = set(G.neighbors(variant_of_interest))
    var_cancers = [
        n for n in var_nei
        if G.nodes[n]['category']=='Cancer'
        and n != cancer_of_interest
    ]
    vc_weights = {}
    for c in var_cancers:
        w_v = G[variant_of_interest][c]['weight']
        w_c = G[cancer_of_interest][c]['weight'] if G.has_edge(cancer_of_interest, c) else 0
        vc_weights[c] = w_v + w_c

        top_var_c = [(key.capitalize(), value) for key, value in sorted(vc_weights.items(), key=lambda x: x[1], reverse=True)[:6]]
        vc_w = list(vc_weights.values())
        cancer_pct = np.percentile(vc_w, CANCER_THRESHOLD_PERCENTILE) if vc_w else 0

    # get gene and variant name from the variant of interest
    variant_name, gene_name = variant_of_interest.split("_")

    return cancer_of_interest, top_sens, top_res, top_var_c, sens_pct, res_pct, cancer_pct, gene_name, variant_name


def autosuggest_item(user_input: str, item_type: str, corresponding_gene = None, FUZZY_MATCH = False) -> list:
    """
    Simple contains-filter.
    - Case-insensitive
    - Suggestions whose *start* matches rank higher
    - Return unique, presorted list
    """
    out = []
    q = user_input.strip().lower()

    if item_type == 'Gene':
        entities = metadata_mapping[metadata_mapping['Category'] == "Variant"]['Entity'] #no genes in the mapping, but only concatenated varinates and genes
    else:
        entities = metadata_mapping[metadata_mapping['Category'] == item_type]['Entity']
    
    if item_type in ['Variant', 'Gene']:
        parsed = [ent.split("_", 1) for ent in entities if "_" in ent]
        if item_type == 'Variant':
            if any(g.lower() == corresponding_gene.lower() for _, g in parsed):
                filtered = [(var, g) for var, g in parsed if g.lower() == corresponding_gene.lower()]
            else:
                filtered = parsed
            out = pd.Series([var.upper() for var, _ in filtered])
        else:
            out = pd.Series([g for _, g in parsed])
    else:
        out = pd.concat([entities, pd.Series(SYNONYM_ARRAY, index=range(len(entities), len(entities)+len(SYNONYM_ARRAY)))],
                ignore_index=True
        ).map(lambda x: str(x).title())

    matching_names = []
    if FUZZY_MATCH:
        for name in entities:
            fuzz_ratio = fuzz.ratio(user_input.lower(), name.lower())
            if fuzz_ratio > MATCHING_RATIO_THRESH * 100 and name.lower() not in out.str.lower():
                matching_names.append(name)

    # two-pass ranking with duplicates removed
    starts = out[out.str.lower().str.startswith(q)]
    contains = out[out.str.lower().str.contains(q) & ~out.isin(starts)]
    suggestions = list(dict.fromkeys(list(starts) + list(contains) + list(matching_names))) 
    
    return suggestions