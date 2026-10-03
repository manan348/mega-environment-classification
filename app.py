"""Streamlit app for mega-environment classification and recommendations."""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mega_env.config import load_config
from mega_env.processing import classify_new_location

st.set_page_config(
    page_title="Mega-Environment Classifier - Punjab Wheat Zones",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_models_and_config():
    config = load_config("config/settings.yaml")
    scaler = joblib.load(config.output_paths["scaler"])
    pca = joblib.load(config.output_paths["pca_model"])
    cluster_model = joblib.load(config.output_paths["cluster_model"])
    
    with open("data/processed/feature_info.json") as f:
        feature_info = json.load(f)
    
    with open(config.output_paths["recommendations"]) as f:
        recommendations = json.load(f)
    
    district_clusters = pd.read_csv(config.output_paths["district_clusters"])
    district_col = [c for c in district_clusters.columns if 'NAME' in c.upper() or 'district' in c.lower()][0]
    
    return config, scaler, pca, cluster_model, feature_info, recommendations, district_clusters, district_col

def main():
    st.title("Mega-Environment Classifier for Punjab Wheat Zones")
    st.markdown("""
    Classify any location in Punjab into a mega-environment cluster and get variety/practice recommendations.
    Based on soil, weather, and topography covariates (no genomic or yield data required).
    """)
    
    # Load models
    try:
        config, scaler, pca, cluster_model, feature_info, recommendations, district_clusters, district_col = load_models_and_config()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Run the pipeline first: `python -m mega_env.pipeline --phase all`")
        return
    
    feature_names = feature_info["feature_names"]
    n_components = feature_info["n_components"]
    
    # Sidebar: Input method
    st.sidebar.header("Input Method")
    input_mode = st.sidebar.radio(
        "Choose input method:",
        ["Select District", "Manual Covariates", "Upload CSV"]
    )
    
    if input_mode == "Select District":
        # District selector
        districts = sorted(district_clusters[district_col].unique())
        selected_district = st.sidebar.selectbox("Select District", districts)
        
        if selected_district:
            # Get district's covariate profile (most recent year)
            district_data = district_clusters[district_clusters[district_col] == selected_district]
            cluster = int(district_data['cluster'].iloc[0])
            
            st.subheader(f"District: {selected_district}")
            st.metric("Mega-Environment Cluster", f"Cluster {cluster}")
            
            # Show recommendations
            show_recommendations(cluster, recommendations)
            
            # Show district cluster composition
            st.subheader("Cluster Composition")
            cluster_districts = district_clusters[district_clusters['cluster'] == cluster][district_col].tolist()
            st.write(f"**Districts in Cluster {cluster}:** {', '.join(cluster_districts)}")
    
    elif input_mode == "Manual Covariates":
        st.subheader("Enter Covariate Values")
        st.markdown("Enter values for each covariate (leave blank to use district average)")
        
        # Create input form
        covariates = {}
        
        # Group covariates by type
        weather_cols = [c for c in feature_names if any(x in c.lower() for x in ['temp', 'precip', 'solar', 'vpd'])]
        soil_cols = [c for c in feature_names if any(x in c.lower() for x in ['clay', 'sand', 'silt', 'ocd', 'ph'])]
        topo_cols = [c for c in feature_names if any(x in c.lower() for x in ['elev', 'slope'])]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Weather**")
            for feat in weather_cols:
                covariates[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"w_{feat}")
        
        with col2:
            st.markdown("**Soil**")
            for feat in soil_cols:
                covariates[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"s_{feat}")
        
        with col3:
            st.markdown("**Topography**")
            for feat in topo_cols:
                covariates[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"t_{feat}")
        
        if st.button("Classify Location"):
            # Filter non-zero values
            input_covariates = {k: v for k, v in covariates.items() if v != 0.0}
            
            if len(input_covariates) < len(feature_names) * 0.5:
                st.warning("Please provide at least 50% of covariates for reliable classification")
            
            cluster = classify_new_location(
                input_covariates, scaler, pca, cluster_model, feature_names, n_components
            )
            
            st.success(f"Classified as **Cluster {cluster}**")
            show_recommendations(cluster, recommendations)
    
    elif input_mode == "Upload CSV":
        st.subheader("Upload Covariate CSV")
        st.markdown("CSV must contain columns matching the feature names")
        
        uploaded = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Check columns
            missing = set(feature_names) - set(df.columns)
            if missing:
                st.warning(f"Missing columns: {missing}")
            
            if st.button("Classify All Rows"):
                results = []
                for _, row in df.iterrows():
                    covariates = row.to_dict()
                    cluster = classify_new_location(
                        covariates, scaler, pca, cluster_model, feature_names, n_components
                    )
                    results.append({"cluster": cluster, **covariates})
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button("Download Results", csv, "classifications.csv", "text/csv")

def show_recommendations(cluster: int, recommendations: dict):
    """Display recommendations for a cluster."""
    cluster_key = str(cluster)
    recs = recommendations.get(cluster_key, {"varieties": [], "practices": {}})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Recommended Varieties")
        if recs["varieties"]:
            for v in recs["varieties"]:
                st.markdown(f"- **{v['name']}** — Yield: {v.get('yield_potential', 'N/A')} t/ha")
                if v.get('notes'):
                    st.caption(v['notes'])
        else:
            st.info("No variety data available. Edit recommendations_varieties.csv")
    
    with col2:
        st.markdown("### Recommended Practices")
        practices = recs["practices"]
        if practices:
            for key, value in practices.items():
                if value:
                    label = key.replace('_', ' ').title()
                    st.markdown(f"- **{label}:** {value}")
        else:
            st.info("No practice data available. Edit recommendations_practices.csv")

if __name__ == "__main__":
    main()