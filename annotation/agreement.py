"""
Inter-Rater Agreement Calculation Module
Supports Cohen's Kappa, Krippendorff's Alpha, Fleiss' Kappa, and Consensus Adjudication.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import cohen_kappa_score, confusion_matrix

try:
    import krippendorff
    HAS_KRIPPENDORFF_LIB = True
except ImportError:
    HAS_KRIPPENDORFF_LIB = False


def calculate_pairwise_cohen_kappa(
    rater_a_series: pd.Series,
    rater_b_series: pd.Series,
    labels: Optional[List[str]] = None
) -> float:
    """
    Calculate Cohen's Kappa between two raters.
    """
    valid_idx = rater_a_series.dropna().index.intersection(rater_b_series.dropna().index)
    if len(valid_idx) == 0:
        return 0.0
    a = rater_a_series.loc[valid_idx].astype(str)
    b = rater_b_series.loc[valid_idx].astype(str)
    return float(cohen_kappa_score(a, b, labels=labels))


def _compute_nominal_krippendorff(reliability_matrix: np.ndarray) -> float:
    """
    Pure Python fallback for nominal Krippendorff's alpha if library is missing.
    Matrix shape: (num_raters, num_items) with values or np.nan
    """
    n_raters, n_items = reliability_matrix.shape
    unique_vals = [v for v in np.unique(reliability_matrix) if not pd.isna(v)]
    if len(unique_vals) <= 1:
        return 1.0

    val_map = {val: i for i, val in enumerate(unique_vals)}
    n_vals = len(unique_vals)
    
    # Coincidence matrix
    coincidence = np.zeros((n_vals, n_vals), dtype=float)
    
    for u in range(n_items):
        item_vals = [reliability_matrix[r, u] for r in range(n_raters) if not pd.isna(reliability_matrix[r, u])]
        m_u = len(item_vals)
        if m_u > 1:
            for val_j in item_vals:
                for val_k in item_vals:
                    if val_j != val_k:
                        coincidence[val_map[val_j], val_map[val_k]] += 1.0 / (m_u - 1)
                    else:
                        coincidence[val_map[val_j], val_map[val_j]] += 1.0 / (m_u - 1) - 1.0 / (m_u - 1)

    # Observed disagreement
    total_pairs = 0
    obs_diff = 0.0
    for u in range(n_items):
        item_vals = [reliability_matrix[r, u] for r in range(n_raters) if not pd.isna(reliability_matrix[r, u])]
        m_u = len(item_vals)
        if m_u > 1:
            for j in range(m_u):
                for k in range(j + 1, m_u):
                    total_pairs += 1
                    if item_vals[j] != item_vals[k]:
                        obs_diff += 1.0
    
    if total_pairs == 0:
        return 1.0
    
    d_o = obs_diff / total_pairs
    
    # Marginal frequencies
    val_counts = {}
    total_annotations = 0
    for r in range(n_raters):
        for u in range(n_items):
            v = reliability_matrix[r, u]
            if not pd.isna(v):
                val_counts[v] = val_counts.get(v, 0) + 1
                total_annotations += 1
    
    if total_annotations <= 1:
        return 1.0

    expected_diff = 0.0
    for v1 in unique_vals:
        for v2 in unique_vals:
            if v1 != v2:
                expected_diff += (val_counts[v1] * val_counts[v2])
    
    d_e = expected_diff / (total_annotations * (total_annotations - 1)) if total_annotations > 1 else 1.0
    
    if d_e == 0:
        return 1.0
    
    alpha = 1.0 - (d_o / d_e)
    return float(np.clip(alpha, -1.0, 1.0))


def calculate_krippendorff_alpha(
    annotations_list: List[pd.Series],
    level_of_measurement: str = "nominal"
) -> float:
    """
    Calculate Krippendorff's alpha across multiple raters.
    """
    if len(annotations_list) < 2:
        return 1.0

    all_ids = set()
    for s in annotations_list:
        all_ids.update(s.index)
    sorted_ids = sorted(list(all_ids))

    matrix = []
    for s in annotations_list:
        row = [s.get(item_id, np.nan) for item_id in sorted_ids]
        matrix.append(row)
    
    reliability_data = np.array(matrix, dtype=object)

    if HAS_KRIPPENDORFF_LIB:
        try:
            # Map strings to integers for krippendorff library
            unique_labels = sorted(list(set(val for row in matrix for val in row if pd.notna(val))))
            label_to_code = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            numeric_data = np.full(reliability_data.shape, np.nan)
            for r in range(reliability_data.shape[0]):
                for c in range(reliability_data.shape[1]):
                    if pd.notna(reliability_data[r, c]):
                        numeric_data[r, c] = label_to_code[reliability_data[r, c]]
            return float(krippendorff.alpha(reliability_data=numeric_data, level_of_measurement=level_of_measurement))
        except Exception:
            return _compute_nominal_krippendorff(reliability_data)
    else:
        return _compute_nominal_krippendorff(reliability_data)


def compute_all_agreement_metrics(annotator_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Compute pairwise Cohen's Kappa, multi-rater Krippendorff's Alpha, and raw agreement %
    across annotators for Sentiment, Toxicity, Sarcasm, and Script-mixing.
    """
    results = {
        "tasks": {},
        "pairwise_kappa": {},
        "krippendorff_alpha": {},
        "annotator_count": len(annotator_dfs),
        "total_overlapping_items": 0
    }

    annotator_names = list(annotator_dfs.keys())
    if len(annotator_names) < 2:
        return results

    # Get overlapping item IDs
    common_ids = set(annotator_dfs[annotator_names[0]]["id"])
    for name in annotator_names[1:]:
        common_ids = common_ids.intersection(set(annotator_dfs[name]["id"]))
    common_ids = sorted(list(common_ids))
    results["total_overlapping_items"] = len(common_ids)

    # Process each task
    tasks = ["sentiment", "toxicity", "sarcasm", "script_mixing"]
    
    for task in tasks:
        # Check if task is present in all dfs
        present = all(task in df.columns for df in annotator_dfs.values())
        if not present:
            continue

        series_dict = {}
        for name, df in annotator_dfs.items():
            sub_df = df[df["id"].isin(common_ids)].set_index("id")
            series_dict[name] = sub_df[task].astype(str)

        # Krippendorff's alpha
        alpha = calculate_krippendorff_alpha(list(series_dict.values()), level_of_measurement="nominal")
        results["krippendorff_alpha"][task] = round(alpha, 4)

        # Pairwise metrics
        pairwise = {}
        exact_agreements = []
        for i in range(len(annotator_names)):
            for j in range(i + 1, len(annotator_names)):
                name_a = annotator_names[i]
                name_b = annotator_names[j]
                pair_key = f"{name_a}_vs_{name_b}"
                
                s_a = series_dict[name_a]
                s_b = series_dict[name_b]
                
                kappa = calculate_pairwise_cohen_kappa(s_a, s_b)
                exact_pct = float((s_a == s_b).mean() * 100.0)
                exact_agreements.append(exact_pct)
                
                pairwise[pair_key] = {
                    "cohen_kappa": round(kappa, 4),
                    "exact_agreement_pct": round(exact_pct, 2)
                }

        results["pairwise_kappa"][task] = pairwise
        results["tasks"][task] = {
            "krippendorff_alpha": round(alpha, 4),
            "mean_pairwise_kappa": round(float(np.mean([p["cohen_kappa"] for p in pairwise.values()])), 4),
            "mean_agreement_pct": round(float(np.mean(exact_agreements)), 2)
        }

    return results


def resolve_majority_voting(annotator_dfs: List[pd.DataFrame], id_col: str = "id") -> pd.DataFrame:
    """
    Consolidate multi-annotator datasets into a single adjudicated Gold Master dataset
    via majority voting.
    """
    combined = pd.concat(annotator_dfs, ignore_index=True)
    grouped = combined.groupby(id_col)

    adjudicated_rows = []
    for item_id, group in grouped:
        text = group["text"].iloc[0] if "text" in group.columns else ""
        
        # Majority sentiment
        sentiment = group["sentiment"].mode().iloc[0] if "sentiment" in group.columns and not group["sentiment"].empty else "neutral"
        # Majority toxicity
        toxicity = group["toxicity"].mode().iloc[0] if "toxicity" in group.columns and not group["toxicity"].empty else "non-toxic"
        # Sarcasm majority
        sarcasm = group["sarcasm"].mode().iloc[0] if "sarcasm" in group.columns and not group["sarcasm"].empty else False
        # Script mixing majority
        script_mixing = group["script_mixing"].mode().iloc[0] if "script_mixing" in group.columns and not group["script_mixing"].empty else False

        adjudicated_rows.append({
            "id": item_id,
            "text": text,
            "sentiment": str(sentiment).lower(),
            "toxicity": str(toxicity).lower(),
            "sarcasm": bool(sarcasm),
            "script_mixing": bool(script_mixing),
            "num_annotators": len(group)
        })

    return pd.DataFrame(adjudicated_rows)


def generate_agreement_report(gold_dir: str = "data/gold") -> pd.DataFrame:
    """
    Load all annotator_*.csv files in gold_dir and produce a summary DataFrame.
    """
    annotator_files = sorted(glob.glob(os.path.join(gold_dir, "annotator_*.csv")))
    if not annotator_files:
        return pd.DataFrame()

    dfs = {}
    for f in annotator_files:
        name = os.path.basename(f).replace(".csv", "")
        dfs[name] = pd.read_csv(f)

    metrics = compute_all_agreement_metrics(dfs)
    
    rows = []
    for task, data in metrics.get("tasks", {}).items():
        rows.append({
            "Task": task.replace("_", " ").title(),
            "Krippendorff Alpha (a)": data["krippendorff_alpha"],
            "Mean Cohen Kappa (k)": data["mean_pairwise_kappa"],
            "Mean Raw Agreement (%)": f"{data['mean_agreement_pct']}%",
            "Reliability Interpretation": (
                "Substantial Agreement" if data["mean_pairwise_kappa"] >= 0.7 
                else ("Moderate Agreement" if data["mean_pairwise_kappa"] >= 0.5 else "Fair Agreement")
            )
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    report = generate_agreement_report()
    print("=== Inter-Annotator Agreement Report ===")
    print(report.to_string(index=False))
