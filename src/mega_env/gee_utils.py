"""Google Earth Engine utilities for data extraction."""
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from mega_env.config import Config


def initialize_gee(project_id: Optional[str] = None) -> None:
    """Initialize Google Earth Engine."""
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        print("GEE initialized successfully")
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        print("Run ee.Authenticate() first if not already authenticated")
        raise


def load_district_boundaries(boundary_file: str) -> gpd.GeoDataFrame:
    """Load Punjab district boundaries from GeoJSON/Shapefile."""
    gdf = gpd.read_file(boundary_file)
    print(f"Loaded {len(gdf)} districts from {boundary_file}")
    return gdf


def district_to_ee_feature(district_row: gpd.GeoSeries, district_name_col: str = "NAME_2") -> ee.Feature:
    """Convert a GeoPandas row to EE Feature."""
    geom = district_row.geometry
    if geom.geom_type == "Polygon":
        coords = list(geom.exterior.coords)
        ee_geom = ee.Geometry.Polygon(coords)
    elif geom.geom_type == "MultiPolygon":
        coords = [[list(p.exterior.coords) for p in geom.geoms]]
        ee_geom = ee.Geometry.MultiPolygon(coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")

    props = {district_name_col: district_row[district_name_col]}
    return ee.Feature(ee_geom, props)


def get_phenology_windows(config: Config, year: int) -> Dict[str, List[str]]:
    """Get phenological window dates for a given year."""
    windows = {}
    for stage, (start, end) in config.phenology.items():
        start_date = f"{year}-{start}"
        end_date = f"{year}-{end}"
        # Handle year wrap for sowing (Nov) -> next year stages
        if stage == "sowing_to_emergence":
            start_date = f"{year}-{start}"
            end_date = f"{year}-{end}"
        windows[stage] = [start_date, end_date]
    return windows


def extract_era5_for_district_year(
    district_fc: ee.FeatureCollection,
    year: int,
    config: Config
) -> ee.ImageCollection:
    """Extract ERA5 daily data for a district and year."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    collection = (
        ee.ImageCollection(config.era5_dataset)
        .filterDate(start_date, end_date)
        .filterBounds(district_fc.geometry())
        .select(config.era5_variables)
    )
    return collection


def aggregate_era5_by_windows(
    collection: ee.ImageCollection,
    windows: Dict[str, List[str]],
    district_fc: ee.FeatureCollection,
    variables: List[str]
) -> ee.FeatureCollection:
    """Aggregate ERA5 variables by phenological windows."""
    results = []

    for stage, (start, end) in windows.items():
        stage_collection = collection.filterDate(start, end)

        # Mean temperature (use temperature_2m for mean)
        if "temperature_2m" in variables:
            mean_temp = stage_collection.select("temperature_2m").mean()
            results.append(("mean_temp_" + stage, mean_temp))

        # Max temperature
        if "temperature_2m_max" in variables:
            max_temp = stage_collection.select("temperature_2m_max").max()
            results.append(("max_temp_" + stage, max_temp))

        # Min temperature
        if "temperature_2m_min" in variables:
            min_temp = stage_collection.select("temperature_2m_min").min()
            results.append(("min_temp_" + stage, min_temp))

        # Precipitation (cumulative)
        if "total_precipitation_sum" in variables:
            precip = stage_collection.select("total_precipitation_sum").sum()
            results.append(("cum_precip_" + stage, precip))

        # Solar radiation (mean)
        if "surface_solar_radiation_downwards_sum" in variables:
            solar = stage_collection.select("surface_solar_radiation_downwards_sum").mean()
            results.append(("mean_solar_" + stage, solar))

        # Dewpoint -> VPD proxy
        if "dewpoint_temperature_2m" in variables and "temperature_2m" in variables:
            dewpoint = stage_collection.select("dewpoint_temperature_2m").mean()
            temp = stage_collection.select("temperature_2m").mean()
            # VPD approximation: saturation VP - actual VP
            # Using simplified formula
            vpd = temp.subtract(dewpoint).rename("vpd_" + stage)
            results.append(("vpd_" + stage, vpd))

    # Combine all bands into one image
    if results:
        combined = ee.Image.cat([img for _, img in results])
        band_names = [name for name, _ in results]
        combined = combined.rename(band_names)

        # Reduce to district level (mean over polygon)
        reduced = combined.reduceRegions(
            collection=district_fc,
            reducer=ee.Reducer.mean(),
            scale=1000  # ERA5-Land is ~11km, use 1km for safety
        )
        return reduced

    return ee.FeatureCollection([])


def extract_soilgrids_for_districts(
    district_fc: ee.FeatureCollection,
    config: Config
) -> ee.FeatureCollection:
    """Extract soil properties from OpenLandMap for districts."""
    # OpenLandMap soil organic carbon (g/kg at 0-30cm ~ b30 band)
    soil_oc = ee.Image(config.soilgrids_dataset).select("b30").rename("organic_carbon")
    # OpenLandMap soil pH (pH x 10 at 0-30cm ~ b30 band)
    soil_ph = ee.Image(config.soilgrids_ph_dataset).select("b30").rename("phh2o")

    soil = ee.Image.cat([soil_oc, soil_ph])

    reduced = soil.reduceRegions(
        collection=district_fc,
        reducer=ee.Reducer.mean(),
        scale=250  # OpenLandMap resolution ~250m
    )
    return reduced


def extract_srtm_for_districts(
    district_fc: ee.FeatureCollection,
    config: Config
) -> ee.FeatureCollection:
    """Extract SRTM elevation and slope for districts."""
    srtm = ee.Image(config.srtm_dataset)
    elevation = srtm.select("elevation")
    slope = ee.Terrain.slope(elevation)

    topo = ee.Image.cat([elevation, slope]).rename(["elevation", "slope"])

    reduced = topo.reduceRegions(
        collection=district_fc,
        reducer=ee.Reducer.mean(),
        scale=90  # SRTM resolution
    )
    return reduced


def ee_fc_to_dataframe(fc: ee.FeatureCollection, properties: List[str]) -> pd.DataFrame:
    """Convert EE FeatureCollection to pandas DataFrame."""
    # Get features as list of dicts
    features = fc.getInfo()["features"]
    rows = []
    for feat in features:
        props = feat["properties"]
        row = {p: props.get(p) for p in properties}
        rows.append(row)
    return pd.DataFrame(rows)


def run_full_extraction(config: Config, districts_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Run complete covariate extraction for all districts and years."""
    initialize_gee(config.gee_project_id)

    # Convert districts to EE FeatureCollection
    district_features = [
        district_to_ee_feature(row) for _, row in districts_gdf.iterrows()
    ]
    district_fc = ee.FeatureCollection(district_features)

    all_rows = []

    for year in config.extraction_years:
        print(f"\n--- Processing year {year} ---")
        windows = get_phenology_windows(config, year)

        # Extract ERA5
        print("  Extracting ERA5...")
        era5_collection = extract_era5_for_district_year(district_fc, year, config)
        era5_agg = aggregate_era5_by_windows(
            era5_collection, windows, district_fc, config.era5_variables
        )

        # Get ERA5 properties
        era5_props = era5_agg.first().propertyNames().getInfo()
        era5_df = ee_fc_to_dataframe(era5_agg, era5_props)
        era5_df["year"] = year

        print(f"  Extracted {len(era5_df)} district-year records")

        all_rows.append(era5_df)

    # Combine all years
    weather_df = pd.concat(all_rows, ignore_index=True)

    # Extract soil (static, no year dimension)
    print("\n--- Extracting SoilGrids ---")
    soil_fc = extract_soilgrids_for_districts(district_fc, config)
    soil_props = soil_fc.first().propertyNames().getInfo()
    soil_df = ee_fc_to_dataframe(soil_fc, soil_props)

    # Extract topography (static)
    print("--- Extracting SRTM ---")
    topo_fc = extract_srtm_for_districts(district_fc, config)
    topo_props = topo_fc.first().propertyNames().getInfo()
    topo_df = ee_fc_to_dataframe(topo_fc, topo_props)

    # Merge: weather (district-year) with soil/topo (district)
    # District name column varies - find it
    district_col = [c for c in weather_df.columns if "NAME" in c.upper() or "district" in c.lower()][0]

    merged = weather_df.merge(soil_df, on=district_col, how="left")
    merged = merged.merge(topo_df, on=district_col, how="left")

    print(f"\nFinal covariate table: {merged.shape[0]} rows x {merged.shape[1]} columns")
    return merged