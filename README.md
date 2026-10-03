# Mega-Environment Classification for Pakistani Crop Zones

Classifies Punjab districts into "mega-environments" using only soil, weather, and topography covariates — no genomic or yield data required. Then recommends wheat varieties and practices known to succeed in each cluster.

## Project Structure

```
mega-environment-classification/
├── config/
│   └── settings.yaml          # All configuration
├── data/
│   ├── raw/                   # Input data (district boundaries)
│   └── processed/             # Output data
├── notebooks/
│   ├── 00_setup_gee.ipynb     # GEE authentication
│   ├── 01_extract_covariates.ipynb  # Phase 1: Data extraction
│   ├── 02_pca_clustering_validation.ipynb  # Phases 2-4
│   └── 03_recommendations.ipynb  # Phase 5: Recommendations
├── src/
│   └── mega_env/
│       ├── __init__.py
│       ├── config.py          # Config loader
│       ├── gee_utils.py       # GEE data extraction
│       ├── processing.py      # PCA, clustering, validation
│       ├── recommendations.py # Recommendation layer
│       └── pipeline.py        # Main pipeline CLI
├── app.py                     # Streamlit inference app
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure
Edit `config/settings.yaml`:
- Set your GEE project ID
- Verify district boundary file path
- Adjust phenological windows if needed
- Set extraction years

### 3. Prepare District Boundaries
Place your Punjab district boundaries (GeoJSON/Shapefile) at:
```
data/raw/punjab_districts.geojson
```
The file should have a column with district names (e.g., `NAME_2`).

### 4. Run Pipeline

**Option A: Step-by-step (notebooks)**
```bash
jupyter notebook notebooks/
```
Run notebooks in order: 00 → 01 → 02 → 03

**Option B: Command line**
```bash
# Full pipeline
python -m mega_env.pipeline --phase all

# Individual phases
python -m mega_env.pipeline --phase 1   # Extraction
python -m mega_env.pipeline --phase 2   # PCA
python -m mega_env.pipeline --phase 3   # Clustering
python -m mega_env.pipeline --phase 4   # Validation
python -m mega_env.pipeline --phase 5   # Recommendation templates
```

### 5. Fill Recommendation Templates
After Phase 3, edit:
- `data/processed/recommendations_varieties.csv`
- `data/processed/recommendations_practices.csv`

Then build final lookup:
```bash
python -m mega_env.pipeline --phase build-recommendations
```

### 6. Launch Inference App
```bash
streamlit run app.py
```

## Configuration

Key settings in `config/settings.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `gee` | `project_id` |
| `districts` | `boundary_file`, `names` |
| `phenology` | 5 wheat growth stages (calendar dates) |
| `extraction` | `years`, `era5_dataset`, `soilgrids_dataset` |
| `pca` | `variance_threshold` (default 0.85) |
| `clustering` | `algorithm`, `k_range`, `random_state` |
| `validation` | `n_splits`, `test_size` |

## Phases Overview

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Extract ERA5 weather, SoilGrids soil, SRTM topography | `covariate_table.parquet` |
| 2 | Standardize → PCA (85% variance) | `pca_model.pkl`, `scaler.pkl` |
| 3 | K-means clustering, optimize k via silhouette | `cluster_model.pkl`, `district_clusters.csv` |
| 4 | Leave-districts-out CV (5-fold) | `validation_results.json` |
| 5 | Map clusters → varieties/practices | `recommendations.json` |

## Validation

The cross-validation holds out 25% of districts, re-clusters on the rest, and classifies held-out districts by nearest centroid. Reports mean agreement across 5 folds.

## Recommendation Data Sources

Fill templates from:
- Punjab Agriculture Department zone recommendations
- PARC/AARI wheat variety trial reports
- Extension service records
- Published literature on Punjab agro-ecological zones

## Inference App

The Streamlit app (`app.py`) allows:
- Select any Punjab district → see its cluster + recommendations
- Manual covariate entry for new locations
- Batch CSV upload for multiple locations

## Scientific Basis

This tests the hypothesis: **Environmental similarity alone can predict which known-successful practices transfer to new locations** — without genomic data or field trials at the target site.

## License

MIT