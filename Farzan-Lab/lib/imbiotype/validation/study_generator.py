"""
Validation Study Generator

Main orchestrator for generating synthetic validation studies with known ground truth.
Creates complete IM-Biotype compatible study directories with data and ground truth files.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from .mixture_model import BiotypeMixtureModel, create_mixture_model
from .multimodal_generator import MultimodalGenerator, ViewConfig, create_two_view_generator
from .noise_controller import (
    NoiseController, SiteEffectConfig, MissingDataConfig, ConfoundConfig,
    create_noise_controller
)


@dataclass
class ValidationStudyConfig:
    """Configuration for a validation study."""
    # Study identification
    name: str = "validation_study"
    description: str = ""

    # Sample parameters
    n_samples: int = 200

    # Cluster/biotype parameters
    n_clusters: int = 3
    cluster_separation: float = 2.0  # Cohen's d equivalent
    cluster_balance: str = "balanced"  # "balanced" or "imbalanced"
    imbalance_ratio: float = 0.8

    # Latent space parameters
    n_latent_dims: int = 5
    n_shared_components: int = 3
    target_correlations: Optional[List[float]] = None

    # View X parameters
    n_features_x: int = 50
    modality_x: str = "eeg"
    categories_x: Optional[List[str]] = None
    snr_x: float = 2.0

    # View Y parameters
    n_features_y: int = 30
    modality_y: str = "clinical"
    categories_y: Optional[List[str]] = None
    snr_y: float = 2.0

    # Noise parameters
    n_sites: int = 1
    site_effect_magnitude: float = 0.5
    missing_rate: float = 0.0
    missing_mechanism: str = "MCAR"
    missing_feature_correlation: float = 1.0  # Strength for MAR/MNAR
    include_age: bool = False
    include_sex: bool = False
    confound_effect_magnitude: float = 0.3

    # Enhancement 1: Non-Gaussian observation model (per view)
    observation_model_x: str = "gaussian"
    observation_model_y: str = "gaussian"
    ordinal_n_levels_x: int = 5
    ordinal_n_levels_y: int = 5

    # Enhancement 2: Covariance structure
    covariance_type: str = "identity"  # "identity" or "ar1"
    ar1_rho: float = 0.5
    variance_scaling: str = "none"  # "none" or "beta"
    variance_scaling_magnitude: float = 1.5

    # Enhancement 3: Per-feature SNR heterogeneity
    snr_distribution_x: str = "uniform"  # "uniform" or "beta"
    snr_distribution_y: str = "uniform"
    snr_beta_a: float = 1.0
    snr_beta_b: float = 1.0
    snr_scale: float = 1.5

    # Enhancement 5: N-view support
    n_views: int = 2  # Default 2 for backward compat
    additional_view_configs: Optional[List[Dict]] = None  # Extra views beyond x/y

    # Reproducibility
    random_state: Optional[int] = 42

    # Visualization
    visualize: bool = False
    visualize_block: bool = True  # Whether plt.show() blocks execution

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)


class ValidationStudyGenerator:
    """
    Generates complete validation studies with known ground truth.

    The generator creates:
    1. Synthetic multimodal data with known biotype structure
    2. Ground truth files (cluster labels, canonical correlations, loadings)
    3. IM-Biotype compatible study directory structure
    4. Confound and site information for testing residualization

    Example:
        >>> config = ValidationStudyConfig(
        ...     name="test_study",
        ...     n_samples=200,
        ...     n_clusters=3,
        ...     modality_x="eeg",
        ...     modality_y="clinical"
        ... )
        >>> generator = ValidationStudyGenerator(config)
        >>> study_path = generator.generate("/path/to/output")
    """

    def __init__(self, config: ValidationStudyConfig):
        """
        Initialize the generator with configuration.

        Args:
            config: ValidationStudyConfig with all parameters
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

        # Initialize components
        self._init_mixture_model()
        self._init_multimodal_generator()
        self._init_noise_controller()

        # Storage for generated data
        self.data: Optional[Dict] = None
        self.ground_truth: Optional[Dict] = None

    def _init_mixture_model(self):
        """Initialize the biotype mixture model."""
        self.mixture_model = create_mixture_model(
            n_clusters=self.config.n_clusters,
            n_latent_dims=self.config.n_latent_dims,
            separation=self.config.cluster_separation,
            balance=self.config.cluster_balance,
            imbalance_ratio=self.config.imbalance_ratio,
            random_state=self.config.random_state
        )

    def _init_multimodal_generator(self):
        """Initialize the multimodal data generator."""
        target_corrs = None
        if self.config.target_correlations is not None:
            target_corrs = np.array(self.config.target_correlations)

        if self.config.n_views > 2 or self.config.additional_view_configs:
            self._init_multiview_generator(target_corrs)
        else:
            self.multimodal_generator = create_two_view_generator(
                n_clusters=self.config.n_clusters,
                n_latent_dims=self.config.n_latent_dims,
                separation=self.config.cluster_separation,
                n_features_x=self.config.n_features_x,
                n_features_y=self.config.n_features_y,
                modality_x=self.config.modality_x,
                modality_y=self.config.modality_y,
                categories_x=self.config.categories_x,
                categories_y=self.config.categories_y,
                snr_x=self.config.snr_x,
                snr_y=self.config.snr_y,
                target_correlations=target_corrs,
                random_state=self.config.random_state,
                observation_model_x=self.config.observation_model_x,
                observation_model_y=self.config.observation_model_y,
                ordinal_n_levels_x=self.config.ordinal_n_levels_x,
                ordinal_n_levels_y=self.config.ordinal_n_levels_y,
                covariance_type=self.config.covariance_type,
                ar1_rho=self.config.ar1_rho,
                variance_scaling=self.config.variance_scaling,
                variance_scaling_magnitude=self.config.variance_scaling_magnitude,
                snr_distribution_x=self.config.snr_distribution_x,
                snr_distribution_y=self.config.snr_distribution_y,
                snr_beta_a=self.config.snr_beta_a,
                snr_beta_b=self.config.snr_beta_b,
                snr_scale=self.config.snr_scale,
            )
            self._is_multiview = False

    def _init_multiview_generator(self, target_corrs):
        """Initialize N-view generator when n_views > 2."""
        from .multiview_generator import MultiviewGenerator, create_multiview_generator

        # Build view configs list: always start with view_x and view_y
        view_configs = [
            ViewConfig(
                name="view_x", n_features=self.config.n_features_x,
                modality=self.config.modality_x, categories=self.config.categories_x,
                snr=self.config.snr_x,
                observation_model=self.config.observation_model_x,
                ordinal_n_levels=self.config.ordinal_n_levels_x,
                snr_distribution=self.config.snr_distribution_x,
                snr_beta_a=self.config.snr_beta_a, snr_beta_b=self.config.snr_beta_b,
                snr_scale=self.config.snr_scale,
            ),
            ViewConfig(
                name="view_y", n_features=self.config.n_features_y,
                modality=self.config.modality_y, categories=self.config.categories_y,
                snr=self.config.snr_y,
                observation_model=self.config.observation_model_y,
                ordinal_n_levels=self.config.ordinal_n_levels_y,
                snr_distribution=self.config.snr_distribution_y,
                snr_beta_a=self.config.snr_beta_a, snr_beta_b=self.config.snr_beta_b,
                snr_scale=self.config.snr_scale,
            ),
        ]

        # Add additional views from config
        if self.config.additional_view_configs:
            for vc_dict in self.config.additional_view_configs:
                view_configs.append(ViewConfig(**vc_dict))

        self.multimodal_generator = create_multiview_generator(
            n_clusters=self.config.n_clusters,
            n_latent_dims=self.config.n_latent_dims,
            separation=self.config.cluster_separation,
            view_configs=view_configs,
            target_correlations=target_corrs,
            random_state=self.config.random_state,
            covariance_type=self.config.covariance_type,
            ar1_rho=self.config.ar1_rho,
            variance_scaling=self.config.variance_scaling,
            variance_scaling_magnitude=self.config.variance_scaling_magnitude,
        )
        self._is_multiview = True

    def _init_noise_controller(self):
        """Initialize the noise controller."""
        self.noise_controller = create_noise_controller(
            n_sites=self.config.n_sites,
            site_effect_magnitude=self.config.site_effect_magnitude,
            missing_rate=self.config.missing_rate,
            missing_mechanism=self.config.missing_mechanism,
            missing_feature_correlation=self.config.missing_feature_correlation,
            include_age=self.config.include_age,
            include_sex=self.config.include_sex,
            confound_effect_magnitude=self.config.confound_effect_magnitude,
            random_state=self.config.random_state
        )

    def generate(self, output_dir: Union[str, Path],
                 create_imbiotype_study: bool = True) -> Path:
        """
        Generate a complete validation study.

        Args:
            output_dir: Directory to create the study in
            create_imbiotype_study: If True, create full IM-Biotype study structure

        Returns:
            Path to the created study directory
        """
        output_path = Path(output_dir) / self.config.name
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer if enabled
        visualizer = None
        if self.config.visualize:
            from .visualizer import ValidationVisualizer
            visualizer = ValidationVisualizer(
                self.config,
                block=self.config.visualize_block
            )

        # Generate multimodal data
        self.data = self.multimodal_generator.generate(
            n_samples=self.config.n_samples,
            balance=self.config.cluster_balance,
            imbalance_ratio=self.config.imbalance_ratio
        )

        # Stage 1 visualization: Mixture model / latent space
        if visualizer:
            visualizer.show_stage1_mixture_model(self.data)

        # Stage 2 visualization: Multimodal data (before noise)
        if visualizer:
            visualizer.show_stage2_multimodal(self.data)

        # Apply noise
        self.data = self.noise_controller.apply(self.data)

        # Stage 3 visualization: Noise effects (if any noise configured)
        if visualizer and visualizer.has_noise():
            visualizer.show_stage3_noise(self.data)

        # Compile ground truth
        self.ground_truth = self._compile_ground_truth()

        # Stage 4 visualization: Final summary
        if visualizer:
            visualizer.show_stage4_summary(self.data)

        if create_imbiotype_study:
            self._create_imbiotype_structure(output_path)
        else:
            self._create_simple_structure(output_path)

        return output_path

    def _compile_ground_truth(self) -> Dict:
        """Compile all ground truth into a single dictionary."""
        dgt = self.data["ground_truth"]

        gt = {
            "config": self.config.to_dict(),
            "generated_at": datetime.now().isoformat(),

            # Cluster ground truth
            "cluster_labels": self.data["cluster_labels"].tolist(),
            "n_clusters": self.config.n_clusters,

            # Latent factors
            "latent_factors": self.data["latent_factors"].tolist(),

            # CCA ground truth
            "true_canonical_correlations": dgt["true_canonical_correlations"].tolist(),
            "n_shared_components": dgt["n_shared_components"],

            # Mixture model ground truth
            "cluster_centroids": dgt["mixture_model"]["cluster_centroids"].tolist(),
            "cluster_covariances": dgt["mixture_model"]["cluster_covariances"].tolist(),
            "mixing_proportions": dgt["mixture_model"]["mixing_proportions"].tolist(),
            "cluster_separation": dgt["mixture_model"]["separation"],
        }

        # Per-view loadings and feature names
        if self._is_multiview and "view_names" in dgt:
            view_names = dgt["view_names"]
            gt["view_names"] = view_names
            for vn in view_names:
                key = f"true_loadings_{vn}"
                if key in dgt:
                    gt[key] = dgt[key].tolist()
            # Feature names from views dict
            if "feature_names" in self.data:
                gt["feature_names"] = self.data["feature_names"]
            # Also add legacy 2-view keys if present
            if "true_loadings_x" in dgt:
                gt["true_loadings_x"] = dgt["true_loadings_x"].tolist()
                gt["true_loadings_y"] = dgt["true_loadings_y"].tolist()
            if "feature_names_x" in self.data:
                gt["feature_names_x"] = self.data["feature_names_x"]
                gt["feature_names_y"] = self.data["feature_names_y"]
        else:
            gt["true_loadings_x"] = dgt["true_loadings_x"].tolist()
            gt["true_loadings_y"] = dgt["true_loadings_y"].tolist()
            gt["feature_names_x"] = self.data["feature_names_x"]
            gt["feature_names_y"] = self.data["feature_names_y"]

        # Add noise ground truth if available
        if "noise_ground_truth" in self.data:
            noise_gt = self.data["noise_ground_truth"]
            gt["n_sites"] = noise_gt.get("n_sites", 1)
            if "site_labels" in noise_gt:
                gt["site_labels"] = noise_gt["site_labels"].tolist()
            if "confounds" in noise_gt:
                gt["confounds"] = {k: v.tolist() for k, v in noise_gt["confounds"].items()}
            if "missing_rate" in noise_gt:
                gt["missing_rate"] = noise_gt["missing_rate"]
                gt["actual_missing_rate_x"] = noise_gt.get("actual_missing_rate_x", 0)
                gt["actual_missing_rate_y"] = noise_gt.get("actual_missing_rate_y", 0)

        return gt

    def _create_imbiotype_structure(self, output_path: Path):
        """Create full IM-Biotype compatible study structure."""
        # Create directories
        raw_data_dir = output_path / "raw_data"
        processed_data_dir = output_path / "processed_data"
        features_dir = output_path / "features"
        ground_truth_dir = output_path / "ground_truth"

        for d in [raw_data_dir, processed_data_dir, features_dir, ground_truth_dir]:
            d.mkdir(exist_ok=True)

        # Create data files
        self._save_view_data(raw_data_dir, processed_data_dir)

        # Create feature definitions
        self._save_feature_definitions(output_path, features_dir)

        # Create confound file if applicable
        self._save_confounds(raw_data_dir, processed_data_dir)

        # Create metadata
        self._save_metadata(output_path)

        # Save ground truth
        self._save_ground_truth(ground_truth_dir)

    def _create_simple_structure(self, output_path: Path):
        """Create simple structure without full IM-Biotype compatibility."""
        subject_ids = [f"S{i:04d}" for i in range(self.config.n_samples)]

        # Save view data as CSVs
        if self._is_multiview and "views" in self.data:
            for vname, vdata in self.data["views"].items():
                fnames = self.data["feature_names"][vname]
                df = pd.DataFrame(vdata, columns=fnames)
                df.insert(0, "subject_id", subject_ids)
                df.to_csv(output_path / f"{vname}.csv", index=False)
        else:
            for suffix in ("x", "y"):
                df = pd.DataFrame(
                    self.data[f"view_{suffix}"],
                    columns=self.data[f"feature_names_{suffix}"]
                )
                df.insert(0, "subject_id", subject_ids)
                df.to_csv(output_path / f"view_{suffix}.csv", index=False)

        # Save ground truth
        with open(output_path / "ground_truth.json", "w") as f:
            json.dump(self.ground_truth, f, indent=2)

        # Save ground truth arrays as npz
        npz_data = dict(
            cluster_labels=self.data["cluster_labels"],
            latent_factors=self.data["latent_factors"],
            true_canonical_correlations=self.data["ground_truth"]["true_canonical_correlations"],
        )
        dgt = self.data["ground_truth"]
        if "true_loadings_x" in dgt:
            npz_data["true_loadings_x"] = dgt["true_loadings_x"]
            npz_data["true_loadings_y"] = dgt["true_loadings_y"]
        if self._is_multiview and "view_names" in dgt:
            for vn in dgt["view_names"]:
                key = f"true_loadings_{vn}"
                if key in dgt:
                    npz_data[key] = dgt[key]
        np.savez(output_path / "ground_truth.npz", **npz_data)

    def _save_view_data(self, raw_data_dir: Path, processed_data_dir: Path):
        """Save view data as CSV files."""
        n_samples = self.config.n_samples
        subject_ids = [f"S{i:04d}" for i in range(n_samples)]

        if self._is_multiview and "views" in self.data:
            for vname, vdata in self.data["views"].items():
                fnames = self.data["feature_names"][vname]
                df = pd.DataFrame(vdata, columns=fnames)
                df.insert(0, "subject_id", subject_ids)
                fname = f"{vname}_data.csv"
                df.to_csv(raw_data_dir / fname, index=False)
                df.to_csv(processed_data_dir / fname, index=False)
        else:
            # Legacy 2-view path
            for suffix, modality in [("x", self.config.modality_x), ("y", self.config.modality_y)]:
                df = pd.DataFrame(
                    self.data[f"view_{suffix}"],
                    columns=self.data[f"feature_names_{suffix}"]
                )
                df.insert(0, "subject_id", subject_ids)
                fname = f"{modality}_data.csv"
                df.to_csv(raw_data_dir / fname, index=False)
                df.to_csv(processed_data_dir / fname, index=False)

    def _save_feature_definitions(self, output_path: Path, features_dir: Path):
        """Save feature definitions as JSON and feature CSV files."""
        subject_ids = [f"S{i:04d}" for i in range(self.config.n_samples)]
        features = {}

        if self._is_multiview and "views" in self.data:
            for vname, vdata in self.data["views"].items():
                fnames = self.data["feature_names"][vname]
                feat_key = f"{vname}_features"
                features[feat_key] = {
                    "name": feat_key,
                    "dataset": f"{vname}_data.csv",
                    "columns": fnames,
                    "created_at": datetime.now().isoformat()
                }
                df = pd.DataFrame(vdata, columns=fnames)
                df.insert(0, "subject_id", subject_ids)
                df.to_csv(features_dir / f"{feat_key}.csv", index=False)
        else:
            for suffix, modality in [("x", self.config.modality_x), ("y", self.config.modality_y)]:
                fnames = self.data[f"feature_names_{suffix}"]
                feat_key = f"{modality}_features"
                features[feat_key] = {
                    "name": feat_key,
                    "dataset": f"{modality}_data.csv",
                    "columns": fnames,
                    "created_at": datetime.now().isoformat()
                }
                df = pd.DataFrame(self.data[f"view_{suffix}"], columns=fnames)
                df.insert(0, "subject_id", subject_ids)
                df.to_csv(features_dir / f"{feat_key}.csv", index=False)

        with open(output_path / "features.json", "w") as f:
            json.dump(features, f, indent=2)

    def _save_confounds(self, raw_data_dir: Path, processed_data_dir: Path):
        """Save confound data if applicable."""
        if "confounds" not in self.data or not self.data["confounds"]:
            return

        n_samples = self.config.n_samples
        subject_ids = [f"S{i:04d}" for i in range(n_samples)]

        confound_df = pd.DataFrame({"subject_id": subject_ids})

        for name, values in self.data["confounds"].items():
            confound_df[name] = values

        # Add site labels if available
        if "site_labels" in self.data:
            confound_df["site"] = [f"site_{s}" for s in self.data["site_labels"]]

        confound_df.to_csv(raw_data_dir / "confounds.csv", index=False)
        confound_df.to_csv(processed_data_dir / "confounds.csv", index=False)

    def _save_metadata(self, output_path: Path):
        """Save study metadata."""
        metadata = {
            "study_name": self.config.name,
            "study_id": f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "description": self.config.description or "Validation study with known ground truth",
            "is_validation_study": True,
            "generator_config": self.config.to_dict()
        }

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_ground_truth(self, ground_truth_dir: Path):
        """Save ground truth files."""
        # JSON for human readability
        with open(ground_truth_dir / "ground_truth.json", "w") as f:
            json.dump(self.ground_truth, f, indent=2)

        # NPZ for efficient loading
        np.savez(
            ground_truth_dir / "ground_truth.npz",
            cluster_labels=self.data["cluster_labels"],
            latent_factors=self.data["latent_factors"],
            true_canonical_correlations=self.data["ground_truth"]["true_canonical_correlations"],
            true_loadings_x=self.data["ground_truth"]["true_loadings_x"],
            true_loadings_y=self.data["ground_truth"]["true_loadings_y"],
            cluster_centroids=self.data["ground_truth"]["mixture_model"]["cluster_centroids"],
            mixing_proportions=self.data["ground_truth"]["mixture_model"]["mixing_proportions"]
        )

        # Save cluster labels as CSV for easy inspection
        labels_df = pd.DataFrame({
            "subject_id": [f"S{i:04d}" for i in range(self.config.n_samples)],
            "true_cluster": self.data["cluster_labels"]
        })
        labels_df.to_csv(ground_truth_dir / "cluster_labels.csv", index=False)

    def get_ground_truth(self) -> Dict:
        """
        Get the compiled ground truth dictionary.

        Returns:
            Dictionary containing all ground truth parameters

        Raises:
            ValueError: If generate() has not been called
        """
        if self.ground_truth is None:
            raise ValueError("Must call generate() before getting ground truth")
        return self.ground_truth

    def get_data(self) -> Dict:
        """
        Get the generated data dictionary.

        Returns:
            Dictionary containing all generated data

        Raises:
            ValueError: If generate() has not been called
        """
        if self.data is None:
            raise ValueError("Must call generate() before getting data")
        return self.data
