"""
Unified CCA entrypoint for IMBiotype.

This thin wrapper standardizes the run_cca signature and delegates to the
enhanced CCA implementation while allowing a simple, stable API:

    run_cca(feature_data: Dict[str, pd.DataFrame], settings: CCASettings, progress_callback=None)

It supports both two-view and multiview CCA according to settings.
"""
from typing import Dict, Optional
import pandas as pd

# Delegate to the enhanced implementation
from .enhanced_cca import run_cca as _enhanced_run_cca


def _find_stratification_variable(feature_data: Dict[str, pd.DataFrame],
                                  group_var: Optional[str]) -> Optional[pd.DataFrame]:
    if not group_var:
        return None
    # Search in all provided datasets for the group var
    for dataset_name, data in feature_data.items():
        # Handle both DataFrames and numpy arrays
        if isinstance(data, pd.DataFrame):
            if group_var in data.columns:
                return data[[group_var]]
        elif hasattr(data, 'columns'):
            # Handle other DataFrame-like objects
            if group_var in data.columns:
                return data[[group_var]]
        # Skip numpy arrays and other non-DataFrame types
    return None


def run_cca(feature_data: Dict[str, pd.DataFrame], settings, progress_callback=None,
            preprocessing_output_dir: Optional[str] = None, feature_processing_configs=None):
    """
    Unified CCA wrapper with the standard signature.

    - For two-view CCA, settings.x_main_feature_vector_name and
      settings.y_main_feature_vector_name must be provided and present in feature_data.
    - For multiview CCA (settings.is_multiview), settings.views must specify the
      dataset keys for each view present in feature_data.

    Args:
        preprocessing_output_dir: Optional directory to save intermediate preprocessing CSVs
        feature_processing_configs: Optional list of FeatureProcessingConfig objects
    """
    if settings.is_multiview:
        # Views are dataset keys
        if not settings.views or not isinstance(settings.views, list):
            raise ValueError("Valid 'views' must be provided for multiview CCA in settings.views")
        views_data = []
        if isinstance(settings.views[0], str):
            for view_key in settings.views:
                if view_key not in feature_data:
                    raise ValueError(f"View key '{view_key}' not found in feature_data")
                view_data = feature_data[view_key]
                # Ensure view data is a DataFrame with string column names
                if not isinstance(view_data, pd.DataFrame):
                    if hasattr(view_data, 'shape'):
                        view_data = pd.DataFrame(view_data)
                        view_data.columns = view_data.columns.astype(str)
                    else:
                        view_data = pd.DataFrame([view_data])
                        view_data.columns = view_data.columns.astype(str)
                views_data.append(view_data)
        else:
            # Column-list based views: expect a single DataFrame in feature_data
            if len(feature_data) != 1:
                raise ValueError("For column-based views, feature_data must contain exactly one DataFrame")
            main_df = next(iter(feature_data.values()))
            # Ensure main_df is a DataFrame with string column names
            if not isinstance(main_df, pd.DataFrame):
                if hasattr(main_df, 'shape'):
                    main_df = pd.DataFrame(main_df)
                    main_df.columns = main_df.columns.astype(str)
                else:
                    main_df = pd.DataFrame([main_df])
                    main_df.columns = main_df.columns.astype(str)
            for col_list in settings.views:
                views_data.append(main_df[col_list])
        # Delegate to enhanced implementation (multiview handled inside)
        return _enhanced_run_cca(
            x_data=views_data[0],
            y_data=views_data[1],
            settings=settings,
            progress_callback=progress_callback,
            preprocessing_output_dir=preprocessing_output_dir,
            feature_processing_configs=feature_processing_configs,
        )

    # Two-view CCA path
    x_name = settings.x_main_feature_vector_name
    y_name = settings.y_main_feature_vector_name
    if not x_name or not y_name:
        raise ValueError("x_main_feature_vector_name and y_main_feature_vector_name must be specified for CCA.")

    if x_name not in feature_data:
        raise ValueError(f"X main feature vector '{x_name}' not found in feature_data.")
    if y_name not in feature_data:
        raise ValueError(f"Y main feature vector '{y_name}' not found in feature_data.")

    x_df = feature_data[x_name]
    y_df = feature_data[y_name]

    # Ensure x_df and y_df are DataFrames with string column names
    if not isinstance(x_df, pd.DataFrame):
        if hasattr(x_df, 'shape'):
            x_df = pd.DataFrame(x_df)
            # Convert integer column names to strings to avoid sklearn warnings
            x_df.columns = x_df.columns.astype(str)
        else:
            x_df = pd.DataFrame([x_df])
            x_df.columns = x_df.columns.astype(str)

    if not isinstance(y_df, pd.DataFrame):
        if hasattr(y_df, 'shape'):
            y_df = pd.DataFrame(y_df)
            # Convert integer column names to strings to avoid sklearn warnings
            y_df.columns = y_df.columns.astype(str)
        else:
            y_df = pd.DataFrame([y_df])
            y_df.columns = y_df.columns.astype(str)

    # Optional covariates
    x_covariates = None
    y_covariates = None
    if settings.model_settings.x_covariate_feature_vector_name:
        x_cov_name = settings.model_settings.x_covariate_feature_vector_name
        x_cov_data = feature_data.get(x_cov_name)
        # Ensure covariates are DataFrames with string column names
        if x_cov_data is not None:
            if isinstance(x_cov_data, pd.DataFrame):
                x_covariates = x_cov_data
            elif hasattr(x_cov_data, 'shape'):
                x_covariates = pd.DataFrame(x_cov_data)
                x_covariates.columns = x_covariates.columns.astype(str)
            else:
                x_covariates = pd.DataFrame([x_cov_data])
                x_covariates.columns = x_covariates.columns.astype(str)

    if settings.model_settings.y_covariate_feature_vector_name:
        y_cov_name = settings.model_settings.y_covariate_feature_vector_name
        y_cov_data = feature_data.get(y_cov_name)
        # Ensure covariates are DataFrames with string column names
        if y_cov_data is not None:
            if isinstance(y_cov_data, pd.DataFrame):
                y_covariates = y_cov_data
            elif hasattr(y_cov_data, 'shape'):
                y_covariates = pd.DataFrame(y_cov_data)
                y_covariates.columns = y_covariates.columns.astype(str)
            else:
                y_covariates = pd.DataFrame([y_cov_data])
                y_covariates.columns = y_covariates.columns.astype(str)

    # Optional site-aware permutation stratification variable
    stratification_variable = None
    if (settings.statistical_testing.enabled and settings.statistical_testing.respect_groups
            and settings.statistical_testing.group_variable):
        stratification_variable = _find_stratification_variable(
            feature_data, settings.statistical_testing.group_variable
        )

    return _enhanced_run_cca(
        x_data=x_df,
        y_data=y_df,
        settings=settings,
        x_covariates=x_covariates,
        y_covariates=y_covariates,
        stratification_variable=stratification_variable,
        progress_callback=progress_callback,
        preprocessing_output_dir=preprocessing_output_dir,
        feature_processing_configs=feature_processing_configs,
    )

