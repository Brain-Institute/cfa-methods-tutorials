"""
Modality Templates for Validation Study Generator

Provides realistic feature naming templates for different research domains.
Users select modality type to generate domain-appropriate variable names.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import itertools


@dataclass
class ModalityTemplate:
    """Template for a data modality with feature naming patterns."""
    name: str
    categories: Dict[str, List[str]]
    description: str = ""
    typical_range: Tuple[float, float] = (0.0, 100.0)
    distribution: str = "normal"  # normal, gamma, beta, binary

    def get_features(self, categories: Optional[List[str]] = None,
                     n_features: Optional[int] = None) -> List[str]:
        """
        Get feature names from specified categories.

        Args:
            categories: List of category names to include. If None, uses all.
            n_features: Max features to return. If None, returns all.

        Returns:
            List of feature names
        """
        if categories is None:
            categories = list(self.categories.keys())

        features = []
        for cat in categories:
            if cat in self.categories:
                features.extend(self.categories[cat])

        if n_features is not None and len(features) > n_features:
            features = features[:n_features]

        return features

    def get_cross_product_features(self, cat1: str, cat2: str,
                                    sep: str = "_") -> List[str]:
        """Generate cross-product of two categories (e.g., power × channel)."""
        if cat1 not in self.categories or cat2 not in self.categories:
            return []

        return [f"{a}{sep}{b}" for a, b in
                itertools.product(self.categories[cat1], self.categories[cat2])]


# =============================================================================
# Clinical/Psychiatric Scales
# =============================================================================

CLINICAL_TEMPLATE = ModalityTemplate(
    name="clinical",
    description="Psychiatric and clinical assessment scales",
    typical_range=(0.0, 100.0),
    distribution="normal",
    categories={
        "depression": [
            "PHQ9_total", "QIDS_total", "HAMD_total", "BDI_total",
            "MADRS_total", "CES_D_total"
        ],
        "anxiety": [
            "GAD7_total", "STAI_state", "STAI_trait", "BAI_total",
            "HAMA_total", "PSWQ_total"
        ],
        "cognition": [
            "MoCA_total", "MMSE_total", "digit_span_forward", "digit_span_backward",
            "trail_making_A", "trail_making_B", "stroop_interference",
            "DSST_correct", "verbal_fluency"
        ],
        "quality_of_life": [
            "SF36_physical", "SF36_mental", "WHO_QoL_physical",
            "WHO_QoL_psychological", "WHO_QoL_social", "WHO_QoL_environment",
            "EQ5D_index", "WHODAS_total"
        ],
        "sleep": [
            "PSQI_total", "ISI_total", "ESS_total", "sleep_efficiency",
            "sleep_latency", "wake_after_onset"
        ],
        "trauma": [
            "PCL5_total", "CTQ_total", "LEC_count", "CAPS_total",
            "ACE_total", "DES_total"
        ],
        "functioning": [
            "GAF_score", "SOFAS_score", "PSP_total", "SDS_total",
            "work_productivity", "social_functioning"
        ],
        "anhedonia": [
            "SHAPS_total", "TEPS_anticipatory", "TEPS_consummatory",
            "DARS_total"
        ],
        "rumination": [
            "RRS_total", "RRS_brooding", "RRS_reflection", "PTQ_total"
        ]
    }
)

# =============================================================================
# EEG Features
# =============================================================================

EEG_TEMPLATE = ModalityTemplate(
    name="eeg",
    description="Electroencephalography features",
    typical_range=(0.0, 50.0),
    distribution="gamma",
    categories={
        "power_bands": [
            "delta", "theta", "alpha", "beta", "gamma",
            "low_alpha", "high_alpha", "low_beta", "high_beta"
        ],
        "channels_10_20": [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "Oz", "O2"
        ],
        "channels_frontal": ["Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8"],
        "channels_central": ["C3", "Cz", "C4"],
        "channels_parietal": ["P3", "Pz", "P4"],
        "channels_occipital": ["O1", "Oz", "O2"],
        "connectivity": [
            "coherence", "phase_lag_index", "power_envelope_correlation",
            "imaginary_coherence", "weighted_phase_lag_index"
        ],
        "erp_components": [
            "P100_amplitude", "P100_latency",
            "N100_amplitude", "N100_latency",
            "P200_amplitude", "P200_latency",
            "N200_amplitude", "N200_latency",
            "P300_amplitude", "P300_latency",
            "N400_amplitude", "N400_latency",
            "P600_amplitude", "P600_latency",
            "MMN_amplitude", "MMN_latency",
            "ERN_amplitude", "LPP_amplitude"
        ],
        "asymmetry": [
            "frontal_alpha_asymmetry", "parietal_alpha_asymmetry",
            "F4_F3_alpha", "P4_P3_alpha"
        ],
        "complexity": [
            "sample_entropy", "permutation_entropy", "multiscale_entropy",
            "lempel_ziv_complexity", "hurst_exponent", "fractal_dimension"
        ],
        "aperiodic": [
            "aperiodic_offset", "aperiodic_exponent", "aperiodic_knee"
        ]
    }
)

# =============================================================================
# MRI Features
# =============================================================================

MRI_TEMPLATE = ModalityTemplate(
    name="mri",
    description="Structural and functional MRI features",
    typical_range=(0.0, 10000.0),  # Volumes in mm³
    distribution="normal",
    categories={
        "subcortical_volumes": [
            "hippocampus_L", "hippocampus_R",
            "amygdala_L", "amygdala_R",
            "thalamus_L", "thalamus_R",
            "caudate_L", "caudate_R",
            "putamen_L", "putamen_R",
            "pallidum_L", "pallidum_R",
            "accumbens_L", "accumbens_R",
            "brainstem", "ICV"
        ],
        "cortical_thickness": [
            "precentral_L", "precentral_R",
            "postcentral_L", "postcentral_R",
            "superiorFrontal_L", "superiorFrontal_R",
            "middleFrontal_L", "middleFrontal_R",
            "inferiorFrontal_L", "inferiorFrontal_R",
            "orbitofrontal_L", "orbitofrontal_R",
            "ACC_L", "ACC_R",
            "PCC_L", "PCC_R",
            "insula_L", "insula_R",
            "superiorTemporal_L", "superiorTemporal_R",
            "middleTemporal_L", "middleTemporal_R",
            "inferiorTemporal_L", "inferiorTemporal_R",
            "fusiform_L", "fusiform_R",
            "parahippocampal_L", "parahippocampal_R",
            "precuneus_L", "precuneus_R",
            "superiorParietal_L", "superiorParietal_R",
            "inferiorParietal_L", "inferiorParietal_R"
        ],
        "white_matter_FA": [
            "FA_corpus_callosum", "FA_genu", "FA_splenium", "FA_body_CC",
            "FA_cingulum_L", "FA_cingulum_R",
            "FA_fornix",
            "FA_uncinate_L", "FA_uncinate_R",
            "FA_SLF_L", "FA_SLF_R",
            "FA_ILF_L", "FA_ILF_R",
            "FA_IFOF_L", "FA_IFOF_R",
            "FA_corticospinal_L", "FA_corticospinal_R"
        ],
        "white_matter_MD": [
            "MD_corpus_callosum", "MD_cingulum_L", "MD_cingulum_R",
            "MD_fornix", "MD_uncinate_L", "MD_uncinate_R"
        ],
        "functional_connectivity": [
            "DMN_connectivity", "SN_connectivity", "CEN_connectivity",
            "DMN_SN_connectivity", "DMN_CEN_connectivity", "SN_CEN_connectivity",
            "PCC_mPFC_connectivity", "amygdala_mPFC_connectivity",
            "striatum_PFC_connectivity"
        ],
        "regional_activity": [
            "ALFF_mPFC", "ALFF_PCC", "ALFF_insula", "ALFF_amygdala",
            "fALFF_mPFC", "fALFF_PCC", "fALFF_insula",
            "ReHo_mPFC", "ReHo_PCC", "ReHo_insula"
        ],
        "network_metrics": [
            "global_efficiency", "local_efficiency", "modularity",
            "small_worldness", "clustering_coefficient", "path_length"
        ]
    }
)

# =============================================================================
# Genetics Features
# =============================================================================

GENETICS_TEMPLATE = ModalityTemplate(
    name="genetics",
    description="Genetic and polygenic risk scores",
    typical_range=(-3.0, 3.0),  # Z-scored PRS
    distribution="normal",
    categories={
        "polygenic_scores": [
            "PRS_MDD", "PRS_SCZ", "PRS_BIP", "PRS_ADHD", "PRS_ASD",
            "PRS_anxiety", "PRS_PTSD", "PRS_OCD",
            "PRS_BMI", "PRS_height", "PRS_education",
            "PRS_neuroticism", "PRS_extraversion", "PRS_wellbeing",
            "PRS_cognitive_ability", "PRS_insomnia"
        ],
        "candidate_genes": [
            "BDNF_val66met", "COMT_val158met", "5HTTLPR_short",
            "APOE_e4_count", "MAOA_activity", "DRD2_Taq1A",
            "OXTR_rs53576", "FKBP5_rs1360780", "NR3C1_BclI"
        ],
        "pathway_scores": [
            "serotonin_pathway", "dopamine_pathway", "glutamate_pathway",
            "GABA_pathway", "HPA_axis_genes", "inflammation_genes",
            "synaptic_plasticity", "circadian_genes"
        ],
        "ancestry": [
            "PC1_ancestry", "PC2_ancestry", "PC3_ancestry", "PC4_ancestry"
        ]
    }
)

# =============================================================================
# Biometrics/Physiology Features
# =============================================================================

BIOMETRICS_TEMPLATE = ModalityTemplate(
    name="biometrics",
    description="Physiological and biometric measurements",
    typical_range=(0.0, 200.0),
    distribution="normal",
    categories={
        "cardiac": [
            "HR_mean", "HR_std", "HR_resting",
            "HRV_RMSSD", "HRV_SDNN", "HRV_pNN50",
            "HRV_LF", "HRV_HF", "HRV_LF_HF_ratio",
            "blood_pressure_systolic", "blood_pressure_diastolic",
            "pulse_wave_velocity"
        ],
        "electrodermal": [
            "SCL_mean", "SCL_std", "SCR_frequency", "SCR_amplitude",
            "skin_conductance_slope", "tonic_EDA", "phasic_EDA"
        ],
        "respiratory": [
            "resp_rate", "resp_variability", "tidal_volume",
            "resp_sinus_arrhythmia", "end_tidal_CO2"
        ],
        "actigraphy": [
            "sleep_efficiency", "sleep_latency", "wake_after_onset",
            "total_sleep_time", "sleep_midpoint", "social_jet_lag",
            "interdaily_stability", "intradaily_variability",
            "step_count", "activity_level", "sedentary_time"
        ],
        "metabolic": [
            "BMI", "waist_circumference", "body_fat_percentage",
            "glucose_fasting", "HbA1c", "insulin_fasting",
            "cholesterol_total", "HDL", "LDL", "triglycerides"
        ],
        "inflammatory": [
            "CRP", "IL6", "TNF_alpha", "IL1_beta", "IL10",
            "fibrinogen", "ESR"
        ],
        "hormonal": [
            "cortisol_morning", "cortisol_evening", "cortisol_CAR",
            "cortisol_slope", "DHEA", "testosterone", "estradiol"
        ],
        "pupillometry": [
            "pupil_size_baseline", "pupil_dilation_max",
            "pupil_constriction_latency", "PIPR"
        ]
    }
)

# =============================================================================
# Behavioral/Task Performance Features
# =============================================================================

BEHAVIORAL_TEMPLATE = ModalityTemplate(
    name="behavioral",
    description="Cognitive task and behavioral measures",
    typical_range=(0.0, 1000.0),
    distribution="normal",
    categories={
        "reaction_time": [
            "RT_mean", "RT_median", "RT_std", "RT_cv",
            "RT_correct", "RT_error", "RT_congruent", "RT_incongruent"
        ],
        "accuracy": [
            "accuracy_overall", "accuracy_go", "accuracy_nogo",
            "hit_rate", "false_alarm_rate", "d_prime", "criterion"
        ],
        "learning": [
            "learning_rate", "learning_asymptote", "trials_to_criterion",
            "reversal_errors", "perseverative_errors", "win_stay", "lose_shift"
        ],
        "decision_making": [
            "risk_preference", "delay_discounting_k", "loss_aversion",
            "probability_weighting", "ambiguity_aversion"
        ],
        "attention": [
            "sustained_attention", "selective_attention", "divided_attention",
            "attention_lapses", "vigilance_decrement"
        ],
        "memory": [
            "immediate_recall", "delayed_recall", "recognition_accuracy",
            "working_memory_span", "spatial_span", "n_back_accuracy"
        ],
        "executive_function": [
            "inhibition_score", "switching_cost", "updating_accuracy",
            "planning_moves", "set_shifting_errors"
        ],
        "motor": [
            "motor_speed", "motor_accuracy", "grip_strength",
            "fine_motor_dexterity", "motor_learning_rate"
        ]
    }
)

# =============================================================================
# Registry of all templates
# =============================================================================

MODALITY_TEMPLATES: Dict[str, ModalityTemplate] = {
    "clinical": CLINICAL_TEMPLATE,
    "eeg": EEG_TEMPLATE,
    "mri": MRI_TEMPLATE,
    "genetics": GENETICS_TEMPLATE,
    "biometrics": BIOMETRICS_TEMPLATE,
    "behavioral": BEHAVIORAL_TEMPLATE
}


def get_template(modality: str) -> ModalityTemplate:
    """Get a modality template by name."""
    if modality not in MODALITY_TEMPLATES:
        raise ValueError(f"Unknown modality: {modality}. "
                        f"Available: {list(MODALITY_TEMPLATES.keys())}")
    return MODALITY_TEMPLATES[modality]


def list_modalities() -> List[str]:
    """List available modality types."""
    return list(MODALITY_TEMPLATES.keys())


def list_categories(modality: str) -> List[str]:
    """List categories within a modality."""
    template = get_template(modality)
    return list(template.categories.keys())


def generate_feature_names(modality: str,
                           categories: Optional[List[str]] = None,
                           n_features: Optional[int] = None,
                           cross_product: Optional[Tuple[str, str]] = None) -> List[str]:
    """
    Generate feature names for a modality.

    Args:
        modality: Modality type (clinical, eeg, mri, genetics, biometrics, behavioral)
        categories: Specific categories to include. If None, uses all.
        n_features: Maximum number of features. If None, returns all.
        cross_product: Tuple of (cat1, cat2) to generate cross-product names.
                      E.g., ("power_bands", "channels") -> "alpha_Fz", "theta_C3"

    Returns:
        List of feature names

    Example:
        >>> generate_feature_names("eeg", ["power_bands"], n_features=5)
        ['delta', 'theta', 'alpha', 'beta', 'gamma']

        >>> generate_feature_names("eeg", cross_product=("power_bands", "channels_frontal"))
        ['delta_Fp1', 'delta_Fp2', ..., 'gamma_F8']
    """
    template = get_template(modality)

    if cross_product is not None:
        features = template.get_cross_product_features(cross_product[0], cross_product[1])
        if n_features is not None:
            features = features[:n_features]
        return features

    return template.get_features(categories, n_features)
