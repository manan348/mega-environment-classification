"""Main pipeline for mega-environment classification."""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mega_env.config import load_config
from mega_env.gee_utils import run_full_extraction, load_district_boundaries
from mega_env.processing import (
    prepare_covariate_matrix,
    standardize_and_pca,
    find_optimal_k,
    fit_clustering,
    aggregate_to_district_level,
    cross_validate_clustering,
    save_models,
    load_models
)
from mega_env.recommendations import create_recommendation_template, build_recommendation_lookup, save_recommendation_lookup


def run_phase1_extraction(config) -> pd.DataFrame:
    """Phase 1: Covariate extraction from GEE."""
    print("\n" + "="*60)
    print("PHASE 1: COVARIATE EXTRACTION")
    print("="*60)

    # Load district boundaries
    districts_gdf = load_district_boundaries(config.districts_boundary_file)

    # Run extraction
    covariate_df = run_full_extraction(config, districts_gdf)

    # Save raw covariate table
    output_path = config.output_paths["covariate_table"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    covariate_df.to_parquet(output_path, index=False)
    print(f"Saved covariate table to {output_path}")

    return covariate_df


def run_phase2_pca(config, covariate_df: pd.DataFrame):
    """Phase 2: Dimensionality reduction with PCA."""
    print("\n" + "="*60)
    print("PHASE 2: DIMENSIONALITY REDUCTION (PCA)")
    print("="*60)

    # Prepare matrix
    df_clean, covariate_cols, district_col = prepare_covariate_matrix(covariate_df, config)
    X = df_clean[covariate_cols].values

    # Standardize and PCA
    X_pca, pca, scaler, component_names = standardize_and_pca(df_clean[covariate_cols], config)

    # Save feature names for later inference
    feature_info = {
        "feature_names": covariate_cols,
        "component_names": component_names,
        "n_components": len(component_names),
        "explained_variance_ratio": pca.explained_variance_ratio_[:len(component_names)].tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)[:len(component_names)].tolist()
    }

    import json
    with open("data/processed/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    return df_clean, X_pca, pca, scaler, covariate_cols, district_col, component_names


def run_phase3_clustering(config, X_pca: np.ndarray, df_clean: pd.DataFrame, district_col: str):
    """Phase 3: Clustering."""
    print("\n" + "="*60)
    print("PHASE 3: CLUSTERING")
    print("="*60)

    # Find optimal k
    print("\nFinding optimal number of clusters...")
    k_results = find_optimal_k(X_pca, config.clustering_k_range, config.clustering_random_state, config.clustering_n_init)

    # Select best k (highest silhouette)
    best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
    print(f"\nBest k = {best_k} (silhouette = {k_results[best_k]['silhouette']:.4f})")

    # Fit final model
    print(f"\nFitting {config.clustering_algorithm} with k={best_k}...")
    cluster_model, labels = fit_clustering(
        X_pca, best_k, config.clustering_algorithm,
        config.clustering_random_state, config.clustering_n_init
    )

    # Aggregate to district level
    district_clusters = aggregate_to_district_level(df_clean, labels, district_col)

    # Save district clusters
    output_path = config.output_paths["district_clusters"]
    district_clusters.to_csv(output_path, index=False)
    print(f"Saved district clusters to {output_path}")

    return cluster_model, labels, district_clusters, best_k, k_results


def run_phase4_validation(config, X_pca: np.ndarray, df_clean: pd.DataFrame, district_col: str, best_k: int):
    """Phase 4: Cross-validation."""
    print("\n" + "="*60)
    print("PHASE 4: CROSS-VALIDATION")
    print("="*60)

    cv_results = cross_validate_clustering(X_pca, df_clean, district_col, best_k, config)

    # Save validation results
    output_path = config.output_paths["validation_results"]
    with open(output_path, "w") as f:
        import json
        json.dump(cv_results, f, indent=2)

    print(f"\nValidation complete: mean agreement = {cv_results['mean_agreement']:.3f} +- {cv_results['std_agreement']:.3f}")
    print(f"Saved validation results to {output_path}")

    return cv_results


def run_phase5_recommendations(config, district_clusters: pd.DataFrame, district_col: str):
    """Phase 5: Create recommendation templates."""
    print("\n" + "="*60)
    print("PHASE 5: RECOMMENDATION TEMPLATES")
    print("="*60)

    variety_df, practice_df = create_recommendation_template(
        district_clusters, district_col, config.output_paths["recommendations"]
    )

    print("\nNEXT STEPS:")
    print(f"  1. Edit {config.output_paths['recommendations'].replace('.csv', '_varieties.csv')}")
    print(f"  2. Edit {config.output_paths['recommendations'].replace('.csv', '_practices.csv')}")
    print("  3. Run: python -m mega_env.pipeline --build-recommendations")

    return variety_df, practice_df


def build_final_recommendations(config):
    """Build final recommendation lookup from filled templates."""
    variety_file = config.output_paths["recommendations"].replace(".csv", "_varieties.csv")
    practice_file = config.output_paths["recommendations"].replace(".csv", "_practices.csv")

    lookup = build_recommendation_lookup(variety_file, practice_file)
    save_recommendation_lookup(lookup, config.output_paths["recommendations"])

    return lookup


def main():
    parser = argparse.ArgumentParser(description="Mega-Environment Classification Pipeline")
    parser.add_argument("--phase", choices=["1", "2", "3", "4", "5", "all", "build-recommendations"],
                        default="all", help="Pipeline phase to run")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file path")
    parser.add_argument("--k", type=int, help="Override optimal k for clustering")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Check GEE project ID
    if config.gee_project_id == "your-gee-project-id":
        print("⚠️  Please set your GEE project ID in config/settings.yaml")
        return

    # Run phases
    if args.phase in ["1", "all"]:
        covariate_df = run_phase1_extraction(config)
    else:
        covariate_df = pd.read_parquet(config.output_paths["covariate_table"])

    if args.phase in ["2", "all"]:
        df_clean, X_pca, pca, scaler, covariate_cols, district_col, component_names = run_phase2_pca(config, covariate_df)
    else:
        # Load from saved
        import joblib
        scaler = joblib.load(config.output_paths["scaler"])
        pca = joblib.load(config.output_paths["pca_model"])
        df_clean = pd.read_parquet(config.output_paths["covariate_table"])
        # Recompute X_pca
        _, covariate_cols, district_col = prepare_covariate_matrix(df_clean, config)
        X_scaled = scaler.transform(df_clean[covariate_cols])
        X_pca = pca.transform(X_scaled)
        with open("data/processed/feature_info.json") as f:
            import json
            feature_info = json.load(f)
        component_names = feature_info["component_names"]
        X_pca = X_pca[:, :len(component_names)]

    if args.phase in ["3", "all"]:
        k = args.k if args.k else None
        cluster_model, labels, district_clusters, best_k, k_results = run_phase3_clustering(config, X_pca, df_clean, district_col)
    else:
        import joblib
        cluster_model = joblib.load(config.output_paths["cluster_model"])
        district_clusters = pd.read_csv(config.output_paths["district_clusters"])
        best_k = district_clusters["cluster"].nunique()

    if args.phase in ["4", "all"]:
        cv_results = run_phase4_validation(config, X_pca, df_clean, district_col, best_k)

    if args.phase in ["5", "all"]:
        run_phase5_recommendations(config, district_clusters, district_col)

    if args.phase == "build-recommendations":
        build_final_recommendations(config)

    # Save models
    if args.phase in ["3", "all"]:
        save_models(scaler, pca, cluster_model, config)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()