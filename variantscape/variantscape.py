import networkx as nx
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_store import G, consensus_dict



# === Adjustable thresholds ===
TREATMENT_THRESHOLD_PERCENTILE = 80    # highlight top X% of treatment weights
TREATMENT_MIN_HIGHLIGHT        = 300   # and require ≥X total weight
CANCER_THRESHOLD_PERCENTILE    = 80    # highlight top X% of cancer–variant weights
CANCER_MIN_HIGHLIGHT           = 80    # and require ≥X total weight

CANCER_ALIAS_MAP = {
    "nsclc": "lung cancer",
    "non-small cell lung cancer": "lung cancer",
    "tnbc": "breast cancer",
    "her2+ breast cancer": "breast cancer"
}

EXCLUDED_TREATMENTS = {
    'chemotherapy', 'tyrosine kinase inhibitor', 'radiotherapy', 'hormone therapy',
    'adjuvant chemotherapy', 'immunotherapy', 'immune checkpoint inhibitor',
    'mrna vaccine', 'mtor inhibitor', 'radiation ionizing radiotherapy'
}


def compute_associations(gene_name, variant_name, cancer_name):
    
    variant_of_interest = (variant_name + "_" + gene_name).lower()
    print("variant of interest", variant_of_interest)
    clean_input = cancer_name.strip().lower()
    cancer_of_interest = CANCER_ALIAS_MAP.get(clean_input, clean_input)

    # === Step 1: Cancer‐only treatments ===
    canc_nei = set(G.neighbors(cancer_of_interest))
    treatments = [
        n for n in canc_nei
        if G.nodes[n]['category']=='Treatment'
        and n.lower() not in EXCLUDED_TREATMENTS
    ]
    t_weights = {t: G[cancer_of_interest][t]['weight'] for t in treatments}
    top_cancer_treats = sorted(t_weights.items(), key=lambda x: x[1], reverse=True)[:6]
    c_w = list(t_weights.values())
    treat_pct = np.percentile(c_w, TREATMENT_THRESHOLD_PERCENTILE) if c_w else 0


    # === Step 2: Variant + cancer associations ===
    lowercase_mapping = {key.lower(): key for key in G.nodes}     # Create a temporary mapping of lowercase keys to original keys
    variant_of_interest = lowercase_mapping.get(variant_of_interest) #TODO deal with case where key does not exist 

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

    
    print(f"\n\033[1mSensitive treatments for variant '{variant_of_interest}' "
        f"(≥{TREATMENT_THRESHOLD_PERCENTILE}th pct & ≥{TREATMENT_MIN_HIGHLIGHT}):\033[0m")
    for t, w in top_sens:
        if w >= sens_pct and w >= TREATMENT_MIN_HIGHLIGHT:
            print(f"\033[1;32m{t}: {w:.0f}\033[0m")
        else:
            print(f"\033[2;37m{t}: {w:.0f}\033[0m")

    print(f"\n\033[1mResistant treatments for variant '{variant_of_interest}' "
        f"(≥{TREATMENT_THRESHOLD_PERCENTILE}th pct & ≥{TREATMENT_MIN_HIGHLIGHT}):\033[0m")
    for t, w in top_res:
        if w >= res_pct and w >= TREATMENT_MIN_HIGHLIGHT:
            print(f"\033[1;31m{t}: {w:.0f}\033[0m")
        else:
            print(f"\033[2;37m{t}: {w:.0f}\033[0m")

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

        top_var_c = sorted(vc_weights.items(), key=lambda x: x[1], reverse=True)[:6]
        vc_w = list(vc_weights.values())
        cancer_pct = np.percentile(vc_w, CANCER_THRESHOLD_PERCENTILE) if vc_w else 0

        print(f"\n\033[1mOther cancers for variant '{variant_of_interest}' "
            f"(≥{CANCER_THRESHOLD_PERCENTILE}th pct & ≥{CANCER_MIN_HIGHLIGHT}):\033[0m")
        for c, w in top_var_c:
            if w >= cancer_pct and w >= CANCER_MIN_HIGHLIGHT:
                print(f"\033[1;34m{c}: {w:.0f}\033[0m")
            else:
                print(f"\033[2;37m{c}: {w:.0f}\033[0m")

        # === Assemble & save ===
        results = []
        for t, w in top_cancer_treats:
            results.append({
                "Cancer": cancer_name, "Variant": None,
                "Treatment": t, "Association_Type": "Cancer-Only",
                "Prediction": "NA", "Combined_Weight": w
            })
        for t, w in top_sens:
            results.append({
                "Cancer": cancer_name, "Variant": variant_of_interest,
                "Treatment": t, "Association_Type": "Variant-Cancer",
                "Prediction": "Sensitive", "Combined_Weight": w
            })
        for t, w in top_res:
            results.append({
                "Cancer": cancer_name, "Variant": variant_of_interest,
                "Treatment": t, "Association_Type": "Variant-Cancer",
                "Prediction": "Resistant", "Combined_Weight": w
            })
        for c, w in top_var_c:
            results.append({
                "Cancer": c, "Variant": variant_of_interest,
                "Treatment": None, "Association_Type": "Cross-Cancer",
                "Prediction": "NA", "Combined_Weight": w
            })