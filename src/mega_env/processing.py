"""Data processing: PCA, clustering, and validation."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings("ignore")

from mega_env.config import Config


def prepare_covariate_matrix(
    df: pd.DataFrame,
    config: Config,
    district_col: str = None
) -> Tuple[pd.DataFrame, List[str], str]:
    """Prepare covariate matrix for PCA/clustering."""
    # Identify district column
    if district_col is None:
        candidates = [c for c in df.columns if "NAME" in c.upper() or "district" in c.lower()]
        district_col = candidates[0] if candidates else df.columns[0]

    # Identify covariate columns (numeric, non-identifier)
    exclude_cols = {district_col, "year", "system:index", ".geo"}
    covariate_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Drop rows with missing covariates
    initial_len = len(df)
    df_clean = df.dropna(subset=covariate_cols).copy()
    if len(df_clean) < initial_len:
        print(f"  Dropped {initial_len - len(df_clean)} rows with missing covariates")

    X = df_clean[covariate_cols].copy()
    return df_clean, covariate_cols, district_col


def standardize_and_pca(
    X: pd.DataFrame,
    config: Config
) -> Tuple[np.ndarray, PCA, StandardScaler, List[str]]:
    """Standardize and run PCA."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=config.pca_n_components_max, random_state=config.clustering_random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Determine n_components for variance threshold
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= config.pca_variance_threshold) + 1
    n_components = min(n_components, config.pca_n_components_max)

    print(f"  PCA: {X.shape[1]} features -> {n_components} components ({cumsum_var[n_components-1]:.1%} variance)")

    # Return truncated PCA space
    X_pca_truncated = X_pca[:, :n_components]
    component_names = [f"PC{i+1}" for i in range(n_components)]

    return X_pca_truncated, pca, scaler, component_names


def find_optimal_k(
    X: np.ndarray,
    k_range: List[int],
    random_state: int = 42,
    n_init: int = 20
) -> Dict[int, Dict[str, float]]:
    """Find optimal number of clusters using silhouette and inertia."""
    results = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = kmeans.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        inertia = kmeans.inertia_

        results[k] = {
            "silhouette": silhouette,
            "inertia": inertia
        }
        print(f"  k={k}: silhouette={silhouette:.4f}, inertia={inertia:.2f}")

    return results


def fit_clustering(
    X: np.ndarray,
    k: int,
    algorithm: str = "kmeans",
    random_state: int = 42,
    n_init: int = 20
) -> Tuple[Any, np.ndarray]:
    """Fit clustering model."""
    if algorithm == "kmeans":
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    elif algorithm == "hierarchical":
        model = AgglomerativeClustering(n_clusters=k)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    labels = model.fit_predict(X)
    return model, labels


def aggregate_to_district_level(
    df: pd.DataFrame,
    labels: np.ndarray,
    district_col: str,
    year_col: str = "year"
) -> pd.DataFrame:
    """Aggregate district-year clusters to district-level (mode)."""
    df = df.copy()
    df["cluster"] = labels

    # Mode cluster per district
    district_clusters = df.groupby(district_col)["cluster"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else -1
    ).reset_index()
    district_clusters.columns = [district_col, "cluster"]

    # Also compute cluster distribution per district
    cluster_dist = df.groupby([district_col, "cluster"]).size().unstack(fill_value=0)
    cluster_dist.columns = [f"cluster_{c}_count" for c in cluster_dist.columns]
    cluster_dist = cluster_dist.reset_index()

    district_clusters = district_clusters.merge(cluster_dist, on=district_col)
    return district_clusters


def cross_validate_clustering(
    X: np.ndarray,
    df: pd.DataFrame,
    district_col: str,
    k: int,
    config: Config
) -> Dict[str, Any]:
    """Leave-districts-out cross-validation."""
    districts = df[district_col].unique()
    n_districts = len(districts)

    kf = KFold(n_splits=config.validation_n_splits, shuffle=True, random_state=config.validation_random_state)

    fold_results = []
    all_agreements = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(districts)):
        train_districts = districts[train_idx]
        test_districts = districts[test_idx]

        # Training data
        train_mask = df[district_col].isin(train_districts)
        X_train = X[train_mask]

        # Fit PCA on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pca = PCA(n_components=X.shape[1], random_state=config.clustering_random_state)
        X_train_pca = pca.fit_transform(X_train_scaled)

        # Determine n_components from training
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= config.pca_variance_threshold) + 1
        X_train_pca = X_train_pca[:, :n_components]

        # Fit clustering on training
        kmeans = KMeans(n_clusters=k, random_state=config.clustering_random_state, n_init=config.clustering_n_init)
        train_labels = kmeans.fit_predict(X_train_pca)
        centroids = kmeans.cluster_centers_

        # Project test data using training PCA/scaler
        test_mask = df[district_col].isin(test_districts)
        X_test = X[test_mask]
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)[:, :n_components]

        # Assign test districts to nearest centroid
        test_labels = []
        for x in X_test_pca:
            distances = np.linalg.norm(centroids - x, axis=1)
            test_labels.append(np.argmin(distances))
        test_labels = np.array(test_labels)

        # Compare with "full" clustering labels for test districts
        # (This requires having run full clustering first - we'll compute agreement differently)
        # For now, compute internal agreement: how consistent are test district-year assignments?
        test_df = df[test_mask].copy()
        test_df["cv_cluster"] = test_labels

        # Agreement: proportion of district-years in each test district that get same cluster
        agreements = []
        for dist in test_districts:
            dist_labels = test_df[test_df[district_col] == dist]["cv_cluster"]
            if len(dist_labels) > 0:
                mode_label = dist_labels.mode().iloc[0]
                agreement = (dist_labels == mode_label).mean()
                agreements.append(agreement)

        fold_agreement = np.mean(agreements) if agreements else 0
        all_agreements.append(fold_agreement)

        fold_results.append({
            "fold": fold,
            "train_districts": len(train_districts),
            "test_districts": len(test_districts),
            "agreement": fold_agreement
        })
        print(f"  Fold {fold+1}: agreement = {fold_agreement:.3f}")

    return {
        "fold_results": fold_results,
        "mean_agreement": np.mean(all_agreements),
        "std_agreement": np.std(all_agreements)
    }


def save_models(
    scaler: StandardScaler,
    pca: PCA,
    cluster_model: Any,
    config: Config
) -> None:
    """Save fitted models."""
    output = config.output_paths
    joblib.dump(scaler, output["scaler"])
    joblib.dump(pca, output["pca_model"])
    joblib.dump(cluster_model, output["cluster_model"])
    print("Models saved")


def load_models(config: Config) -> Tuple[StandardScaler, PCA, Any]:
    """Load fitted models."""
    output = config.output_paths
    scaler = joblib.load(output["scaler"])
    pca = joblib.load(output["pca_model"])
    cluster_model = joblib.load(output["cluster_model"])
    return scaler, pca, cluster_model


def classify_new_location(
    covariate_dict: Dict[str, float],
    scaler: StandardScaler,
    pca: PCA,
    cluster_model: Any,
    feature_names: List[str],
    n_components: int
) -> int:
    """Classify a new location into a mega-environment cluster."""
    # Create feature vector in correct order
    x = np.array([[covariate_dict.get(f, 0) for f in feature_names]])

    # Transform
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)[:, :n_components]

    # Predict
    cluster = cluster_model.predict(x_pca)[0]
    return int(cluster)