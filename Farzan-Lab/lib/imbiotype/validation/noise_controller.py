"""
Noise Controller for Validation Study Generator

Adds realistic noise patterns to generated data:
- Multi-site batch effects
- Missing data (MCAR, MAR)
- Categorical and continuous confounds
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class SiteEffectConfig:
    """Configuration for multi-site batch effects."""
    n_sites: int = 3
    site_effect_magnitude: float = 0.5  # Relative to signal std
    site_proportions: Optional[np.ndarray] = None  # Samples per site
    site_biotype_independent: bool = True  # Sites orthogonal to biotypes
    site_effect_rank: int = 3  # Low-rank structure for realistic batch effects

    def __post_init__(self):
        if self.site_proportions is None:
            self.site_proportions = np.ones(self.n_sites) / self.n_sites


@dataclass
class MissingDataConfig:
    """Configuration for missing data patterns."""
    missing_rate: float = 0.1  # Overall proportion of missing values
    mechanism: str = "MCAR"  # MCAR, MAR, or MNAR
    feature_correlation: float = 0.0  # For MAR: correlation with other features


@dataclass
class ConfoundConfig:
    """Configuration for confounding variables."""
    include_age: bool = True
    include_sex: bool = True
    include_continuous: List[str] = field(default_factory=list)
    age_range: Tuple[float, float] = (18.0, 65.0)
    sex_ratio: float = 0.5  # Proportion female
    confound_effect_magnitude: float = 0.3  # Effect on features


class NoiseController:
    """
    Applies realistic noise patterns to generated multimodal data.

    Site effects, missing data, and confounds are applied after the
    base data generation to maintain ground truth interpretability.
    """

    def __init__(self,
                 site_config: Optional[SiteEffectConfig] = None,
                 missing_config: Optional[MissingDataConfig] = None,
                 confound_config: Optional[ConfoundConfig] = None,
                 random_state: Optional[int] = None):
        """
        Initialize noise controller.

        Args:
            site_config: Multi-site effect configuration
            missing_config: Missing data configuration
            confound_config: Confound variable configuration
            random_state: Random seed
        """
        self.site_config = site_config or SiteEffectConfig(n_sites=1)
        self.missing_config = missing_config
        self.confound_config = confound_config
        self.rng = np.random.default_rng(random_state)

        # Store applied effects for ground truth
        self.site_labels: Optional[np.ndarray] = None
        self.site_effects: Dict[str, np.ndarray] = {}
        self.site_effects_x: Optional[np.ndarray] = None  # Legacy
        self.site_effects_y: Optional[np.ndarray] = None  # Legacy
        self.confounds: Dict[str, np.ndarray] = {}
        self.missing_masks: Dict[str, np.ndarray] = {}
        self.missing_mask_x: Optional[np.ndarray] = None  # Legacy
        self.missing_mask_y: Optional[np.ndarray] = None  # Legacy

    def _get_view_keys(self, data: Dict) -> List[str]:
        """
        Get view keys from data dict. Supports both 2-view (view_x/view_y)
        and N-view (views dict) formats.

        Returns:
            List of view keys present in data
        """
        if "views" in data:
            return list(data["views"].keys())
        else:
            keys = []
            if "view_x" in data:
                keys.append("view_x")
            if "view_y" in data:
                keys.append("view_y")
            return keys

    def _get_view(self, data: Dict, key: str) -> np.ndarray:
        """Get view data by key, supporting both formats."""
        if "views" in data and key in data["views"]:
            return data["views"][key]
        return data[key]

    def _set_view(self, data: Dict, key: str, value: np.ndarray):
        """Set view data by key, supporting both formats."""
        if "views" in data and key in data["views"]:
            data["views"][key] = value
        if key in data:
            data[key] = value

    def apply(self, data: Dict) -> Dict:
        """
        Apply all noise patterns to generated data.

        Args:
            data: Dictionary from MultimodalGenerator.generate() or
                  MultiviewGenerator.generate()

        Returns:
            Modified data dictionary with noise applied
        """
        view_keys = self._get_view_keys(data)
        if not view_keys:
            raise ValueError("No view data found in data dict")

        # Apply site effects
        if self.site_config.n_sites > 1:
            data = self._apply_site_effects(data)

        # Generate confounds
        if self.confound_config is not None:
            data = self._apply_confounds(data)

        # Apply missing data (last, so ground truth is preserved)
        if self.missing_config is not None and self.missing_config.missing_rate > 0:
            data = self._apply_missing_data(data)

        # Add noise ground truth to data
        data["noise_ground_truth"] = self.get_ground_truth()

        return data

    def _apply_site_effects(self, data: Dict) -> Dict:
        """
        Apply multi-site batch effects.

        Site effects are additive shifts applied to all features.
        Uses low-rank structure for realistic batch effects (correlated across features).
        """
        view_keys = self._get_view_keys(data)
        first_view = self._get_view(data, view_keys[0])
        n_samples = first_view.shape[0]

        # Assign samples to sites
        self.site_labels = self.rng.choice(
            self.site_config.n_sites,
            size=n_samples,
            p=self.site_config.site_proportions
        )

        # Generate and apply site effects per view
        self.site_effects = {}
        for vk in view_keys:
            view_data = self._get_view(data, vk)
            n_features = view_data.shape[1]
            signal_std = np.std(view_data)

            rank = min(self.site_config.site_effect_rank, n_features)
            U = self.rng.standard_normal((self.site_config.n_sites, rank))
            V = self.rng.standard_normal((n_features, rank))
            site_effect = (U @ V.T) * self.site_config.site_effect_magnitude * signal_std / np.sqrt(rank)
            self.site_effects[vk] = site_effect

            view_noisy = view_data.copy()
            for site in range(self.site_config.n_sites):
                mask = self.site_labels == site
                view_noisy[mask] += site_effect[site]
            self._set_view(data, vk, view_noisy)

        # Legacy attributes for backward compat
        if "view_x" in self.site_effects:
            self.site_effects_x = self.site_effects.get("view_x")
            self.site_effects_y = self.site_effects.get("view_y")

        data["site_labels"] = self.site_labels
        return data

    def _apply_confounds(self, data: Dict) -> Dict:
        """
        Generate and apply confounding variables.

        Confounds are generated independently of biotypes but may
        have effects on the observed features. The total effect magnitude
        is normalized by the number of confounds so that adding more confounds
        does not increase total variance beyond the specified magnitude.
        """
        view_keys = self._get_view_keys(data)
        first_view = self._get_view(data, view_keys[0])
        n_samples = first_view.shape[0]

        self.confounds = {}

        # Generate age (using truncated normal for realism)
        if self.confound_config.include_age:
            age_min, age_max = self.confound_config.age_range
            age_mean = (age_min + age_max) / 2
            age_std = (age_max - age_min) / 4
            age = self.rng.normal(age_mean, age_std, n_samples)
            self.confounds["age"] = np.clip(age, age_min, age_max)

        # Generate sex (0 = male, 1 = female)
        if self.confound_config.include_sex:
            self.confounds["sex"] = self.rng.binomial(
                1, self.confound_config.sex_ratio, n_samples
            )

        # Generate additional continuous confounds
        for name in self.confound_config.include_continuous:
            self.confounds[name] = self.rng.standard_normal(n_samples)

        # Apply confound effects to all views
        if self.confound_config.confound_effect_magnitude > 0 and len(self.confounds) > 0:
            n_confounds = len(self.confounds)
            per_confound_magnitude = self.confound_config.confound_effect_magnitude / np.sqrt(n_confounds)

            for vk in view_keys:
                view_noisy = self._get_view(data, vk).copy()
                n_features = view_noisy.shape[1]

                for name, values in self.confounds.items():
                    values_std = (values - values.mean()) / (values.std() + 1e-8)
                    effect = self.rng.standard_normal(n_features) * per_confound_magnitude
                    view_noisy += np.outer(values_std, effect)

                self._set_view(data, vk, view_noisy)

        data["confounds"] = self.confounds
        return data

    def _apply_missing_data(self, data: Dict) -> Dict:
        """
        Apply missing data patterns. Supports both 2-view and N-view data.

        MCAR: Missing Completely At Random - uniform probability
        MAR: Missing At Random - probability depends on OTHER observed variables
        MNAR: Missing Not At Random - missingness depends on the values themselves
        """
        view_keys = self._get_view_keys(data)
        self.missing_masks = {}

        if self.missing_config.mechanism == "MCAR":
            for vk in view_keys:
                view = self._get_view(data, vk)
                self.missing_masks[vk] = self.rng.random(view.shape) < \
                                         self.missing_config.missing_rate

        elif self.missing_config.mechanism == "MAR":
            # For each view, use mean of ALL OTHER views as predictor
            for vk in view_keys:
                view = self._get_view(data, vk)
                n_samples, n_features = view.shape

                # Predictor: mean of all other views' row means
                other_preds = []
                for ok in view_keys:
                    if ok != vk:
                        other_preds.append(np.mean(self._get_view(data, ok), axis=1))

                if other_preds:
                    predictor = np.mean(other_preds, axis=0)
                else:
                    predictor = np.zeros(n_samples)

                pred_std = (predictor - predictor.mean()) / (predictor.std() + 1e-8)
                logit = self.missing_config.feature_correlation * pred_std
                prob_raw = self._sigmoid(logit)
                scale = self.missing_config.missing_rate / (prob_raw.mean() + 1e-8)
                prob = np.clip(prob_raw * scale, 0.0, 1.0)
                self.missing_masks[vk] = self.rng.random((n_samples, n_features)) < \
                                         prob[:, np.newaxis]

        elif self.missing_config.mechanism == "MNAR":
            alpha = self.missing_config.feature_correlation
            for vk in view_keys:
                view = self._get_view(data, vk)
                self.missing_masks[vk] = self._compute_mnar_mask(view, alpha)

        else:
            raise ValueError(
                f"Unknown missing mechanism: '{self.missing_config.mechanism}'. "
                f"Supported mechanisms: 'MCAR', 'MAR', 'MNAR'."
            )

        # Apply missing values
        for vk in view_keys:
            view_missing = self._get_view(data, vk).copy()
            view_missing[self.missing_masks[vk]] = np.nan
            self._set_view(data, vk, view_missing)
            data[f"missing_mask_{vk}"] = self.missing_masks[vk]

        # Legacy attributes for backward compat
        self.missing_mask_x = self.missing_masks.get("view_x")
        self.missing_mask_y = self.missing_masks.get("view_y")

        return data

    def _compute_mnar_mask(self, view: np.ndarray, alpha: float) -> np.ndarray:
        """
        Compute MNAR missingness mask: P(missing | X_ij) depends on X_ij itself.

        Mechanism: higher values → higher probability of being missing.
        P(missing) = sigmoid(alpha * z-scored(X_ij)), rescaled to target missing_rate.

        Args:
            view: Data matrix (n_samples, n_features)
            alpha: Strength of value-dependent missingness (higher = stronger MNAR)

        Returns:
            Boolean mask (n_samples, n_features) where True = missing
        """
        # Per-feature standardization
        col_mean = np.nanmean(view, axis=0)
        col_std = np.nanstd(view, axis=0)
        col_std = np.maximum(col_std, 1e-8)
        view_std = (view - col_mean) / col_std

        # Compute per-element missingness probability
        logits = alpha * view_std
        probs_raw = self._sigmoid(logits)

        # Rescale to achieve target missing rate on average
        scale = self.missing_config.missing_rate / (probs_raw.mean() + 1e-8)
        probs = np.clip(probs_raw * scale, 0.0, 1.0)

        return self.rng.random(view.shape) < probs

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function for MAR/MNAR probability mapping."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def get_ground_truth(self) -> Dict:
        """
        Export noise ground truth parameters.

        Returns:
            Dictionary containing all applied noise parameters
        """
        gt = {
            "n_sites": self.site_config.n_sites,
            "site_effect_magnitude": self.site_config.site_effect_magnitude
        }

        if self.site_labels is not None:
            gt["site_labels"] = self.site_labels.copy()
            if self.site_effects_x is not None:
                gt["site_effects_x"] = self.site_effects_x.copy()
            if self.site_effects_y is not None:
                gt["site_effects_y"] = self.site_effects_y.copy()
            if hasattr(self, 'site_effects') and self.site_effects:
                gt["site_effects"] = {k: v.copy() for k, v in self.site_effects.items()}

        if self.confounds:
            gt["confounds"] = {k: v.copy() for k, v in self.confounds.items()}

        if self.missing_config is not None:
            gt["missing_rate"] = self.missing_config.missing_rate
            gt["missing_mechanism"] = self.missing_config.mechanism
            if self.missing_masks:
                for vk, mask in self.missing_masks.items():
                    gt[f"actual_missing_rate_{vk}"] = mask.mean()
            # Legacy keys
            if self.missing_mask_x is not None:
                gt["actual_missing_rate_x"] = self.missing_mask_x.mean()
            if self.missing_mask_y is not None:
                gt["actual_missing_rate_y"] = self.missing_mask_y.mean()

        return gt


def create_noise_controller(
    n_sites: int = 1,
    site_effect_magnitude: float = 0.5,
    missing_rate: float = 0.0,
    missing_mechanism: str = "MCAR",
    missing_feature_correlation: float = 1.0,
    include_age: bool = False,
    include_sex: bool = False,
    confound_effect_magnitude: float = 0.3,
    random_state: Optional[int] = None
) -> NoiseController:
    """
    Factory function to create a NoiseController with common configurations.

    Args:
        n_sites: Number of sites (1 = no site effects)
        site_effect_magnitude: Relative magnitude of site effects
        missing_rate: Proportion of missing values (0 = no missing)
        missing_mechanism: MCAR, MAR, or MNAR
        missing_feature_correlation: For MAR, strength of correlation between
            missingness and the OTHER view (0 = like MCAR, higher = stronger MAR)
        include_age: Generate age confound
        include_sex: Generate sex confound
        confound_effect_magnitude: Effect of confounds on features
        random_state: Random seed

    Returns:
        Configured NoiseController
    """
    site_config = SiteEffectConfig(
        n_sites=n_sites,
        site_effect_magnitude=site_effect_magnitude
    )

    missing_config = None
    if missing_rate > 0:
        missing_config = MissingDataConfig(
            missing_rate=missing_rate,
            mechanism=missing_mechanism,
            feature_correlation=missing_feature_correlation
        )

    confound_config = None
    if include_age or include_sex:
        confound_config = ConfoundConfig(
            include_age=include_age,
            include_sex=include_sex,
            confound_effect_magnitude=confound_effect_magnitude
        )

    return NoiseController(
        site_config=site_config,
        missing_config=missing_config,
        confound_config=confound_config,
        random_state=random_state
    )
