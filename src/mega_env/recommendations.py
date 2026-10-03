"""Recommendation layer: map clusters to varieties/practices."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

from mega_env.config import Config


def load_recommendation_data(
    variety_file: str = None,
    practice_file: str = None
) -> Dict[str, pd.DataFrame]:
    """Load variety and practice recommendation data."""
    data = {}

    if variety_file and Path(variety_file).exists():
        data["varieties"] = pd.read_csv(variety_file)
        print(f"Loaded varieties: {len(data['varieties'])} records")

    if practice_file and Path(practice_file).exists():
        data["practices"] = pd.read_csv(practice_file)
        print(f"Loaded practices: {len(data['practices'])} records")

    return data


def create_recommendation_template(
    district_clusters: pd.DataFrame,
    district_col: str,
    output_path: str
) -> pd.DataFrame:
    """Create a template for manual recommendation entry."""
    clusters = sorted(district_clusters["cluster"].unique())

    # Variety template
    variety_template = []
    for cluster in clusters:
        districts_in_cluster = district_clusters[district_clusters["cluster"] == cluster][district_col].tolist()
        variety_template.append({
            "cluster": cluster,
            "districts": "; ".join(districts_in_cluster),
            "recommended_varieties": "",  # Fill manually
            "yield_potential_ton_ha": "",
            "notes": ""
        })

    # Practice template
    practice_template = []
    for cluster in clusters:
        districts_in_cluster = district_clusters[district_clusters["cluster"] == cluster][district_col].tolist()
        practice_template.append({
            "cluster": cluster,
            "districts": "; ".join(districts_in_cluster),
            "sowing_window": "",
            "seed_rate_kg_ha": "",
            "fertilizer_npk_kg_ha": "",
            "irrigation_schedule": "",
            "key_pests_diseases": "",
            "notes": ""
        })

    variety_df = pd.DataFrame(variety_template)
    practice_df = pd.DataFrame(practice_template)

    # Save both
    variety_path = output_path.replace(".csv", "_varieties.csv")
    practice_path = output_path.replace(".csv", "_practices.csv")

    variety_df.to_csv(variety_path, index=False)
    practice_df.to_csv(practice_path, index=False)

    print(f"Created templates: {variety_path}, {practice_path}")
    return variety_df, practice_df


def build_recommendation_lookup(
    variety_file: str,
    practice_file: str
) -> Dict[int, Dict]:
    """Build cluster -> recommendations lookup."""
    lookup = {}

    varieties = pd.read_csv(variety_file)
    practices = pd.read_csv(practice_file)

    for _, row in varieties.iterrows():
        cluster = int(row["cluster"])
        if cluster not in lookup:
            lookup[cluster] = {"varieties": [], "practices": {}}
        if pd.notna(row["recommended_varieties"]) and row["recommended_varieties"]:
            lookup[cluster]["varieties"].append({
                "name": row["recommended_varieties"],
                "yield_potential": row.get("yield_potential_ton_ha", ""),
                "notes": row.get("notes", "")
            })

    for _, row in practices.iterrows():
        cluster = int(row["cluster"])
        if cluster not in lookup:
            lookup[cluster] = {"varieties": [], "practices": {}}
        lookup[cluster]["practices"] = {
            "sowing_window": row.get("sowing_window", ""),
            "seed_rate_kg_ha": row.get("seed_rate_kg_ha", ""),
            "fertilizer_npk_kg_ha": row.get("fertilizer_npk_kg_ha", ""),
            "irrigation_schedule": row.get("irrigation_schedule", ""),
            "key_pests_diseases": row.get("key_pests_diseases", ""),
            "notes": row.get("notes", "")
        }

    return lookup


def get_recommendations(
    cluster: int,
    lookup: Dict[int, Dict]
) -> Dict:
    """Get recommendations for a cluster."""
    return lookup.get(cluster, {"varieties": [], "practices": {}})


def save_recommendation_lookup(lookup: Dict[int, Dict], output_path: str) -> None:
    """Save recommendation lookup as JSON."""
    with open(output_path, "w") as f:
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(lookup), f, indent=2)
    print(f"Saved recommendation lookup to {output_path}")