"""Configuration loader for the mega-environment classification project."""
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class Config:
    gee_project_id: str
    districts_boundary_file: str
    district_names: list
    phenology: Dict[str, list]
    extraction_years: list
    era5_dataset: str
    soilgrids_dataset: str
    soilgrids_ph_dataset: str
    srtm_dataset: str
    era5_variables: list
    soil_properties: list
    topography: list
    pca_variance_threshold: float
    pca_n_components_max: int
    clustering_algorithm: str
    clustering_k_range: list
    clustering_random_state: int
    clustering_n_init: int
    validation_n_splits: int
    validation_test_size: float
    validation_random_state: int
    output_paths: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return cls(
            gee_project_id=cfg["gee"]["project_id"],
            districts_boundary_file=cfg["districts"]["boundary_file"],
            district_names=cfg["districts"]["names"],
            phenology=cfg["phenology"],
            extraction_years=cfg["extraction"]["years"],
            era5_dataset=cfg["extraction"]["era5_dataset"],
            soilgrids_dataset=cfg["extraction"]["soilgrids_dataset"],
            soilgrids_ph_dataset=cfg["extraction"]["soilgrids_ph_dataset"],
            srtm_dataset=cfg["extraction"]["srtm_dataset"],
            era5_variables=cfg["era5_variables"],
            soil_properties=cfg["soil_properties"],
            topography=cfg["topography"],
            pca_variance_threshold=cfg["pca"]["variance_threshold"],
            pca_n_components_max=cfg["pca"]["n_components_max"],
            clustering_algorithm=cfg["clustering"]["algorithm"],
            clustering_k_range=cfg["clustering"]["k_range"],
            clustering_random_state=cfg["clustering"]["random_state"],
            clustering_n_init=cfg["clustering"]["n_init"],
            validation_n_splits=cfg["validation"]["n_splits"],
            validation_test_size=cfg["validation"]["test_size"],
            validation_random_state=cfg["validation"]["random_state"],
            output_paths=cfg["output"],
        )


def load_config(path: str | Path = "config/settings.yaml") -> Config:
    """Load configuration from YAML file, with optional local override."""
    # Load base config
    cfg = Config.from_yaml(path)
    
    # Check for local override (gitignored)
    local_path = Path(path).parent / "local_settings.yaml"
    if local_path.exists():
        with open(local_path, "r") as f:
            local_cfg = yaml.safe_load(f)
        # Override with local settings
        if "gee" in local_cfg and "project_id" in local_cfg["gee"]:
            cfg.gee_project_id = local_cfg["gee"]["project_id"]
    
    return cfg