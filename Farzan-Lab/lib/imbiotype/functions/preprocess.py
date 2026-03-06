"""
Core preprocessing functions for imbiotype datasets.
"""

from typing import Dict, Optional, List, Union
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import re

def apply_scaling(
    df: pd.DataFrame,
    column_operations: Dict[str, str]
) -> pd.DataFrame:
    """
    Applies standardization or normalization to specified columns of a DataFrame.

    Args:
        df: The input pandas DataFrame.
        column_operations: Dictionary mapping column names to operations
                           ('standardize', 'normalize', or 'none'). Columns not
                           present in the dictionary are left unchanged.

    Returns:
        A new DataFrame with the specified transformations applied.

    Raises:
        ValueError: If an invalid operation is specified or a target column doesn't exist.
        TypeError: If trying to scale a non-numeric column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(column_operations, dict):
        raise TypeError("Input 'column_operations' must be a dictionary.")

    df_copy = df.copy()
    valid_ops = ['standardize', 'normalize', 'none']

    for col_name, operation in column_operations.items():
        operation = operation.lower() # Ensure case-insensitivity

        if col_name not in df_copy.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame.")

        if operation not in valid_ops:
            raise ValueError(f"Invalid operation '{operation}' specified for column '{col_name}'. Valid options are: {valid_ops}")

        if operation == 'none':
            continue # Skip to the next column

        # Check if column is numeric before applying scaling
        if not pd.api.types.is_numeric_dtype(df_copy[col_name]):
            raise TypeError(f"Cannot apply scaling operation '{operation}' to non-numeric column '{col_name}'.")

        # Extract column as DataFrame for scaler compatibility
        target_col_df = df_copy[[col_name]]

        # Handle columns that might be constant (causing scaler errors)
        if target_col_df[col_name].nunique() <= 1:
             print(f"Warning: Column '{col_name}' has 0 or 1 unique values. Skipping scaling operation '{operation}'.")
             continue

        try:
            if operation == 'standardize':
                scaler = StandardScaler()
                # Fit and transform, result is a numpy array
                scaled_values = scaler.fit_transform(target_col_df)
                # Assign back to the DataFrame column
                df_copy[col_name] = scaled_values.flatten() # Flatten needed as fit_transform returns 2D array

            elif operation == 'normalize':
                scaler = MinMaxScaler()
                # Fit and transform, result is a numpy array
                scaled_values = scaler.fit_transform(target_col_df)
                # Assign back to the DataFrame column
                df_copy[col_name] = scaled_values.flatten() # Flatten needed

        except Exception as e:
             # Catch potential errors during scaling
             raise RuntimeError(f"Error applying '{operation}' to column '{col_name}': {e}") from e

    return df_copy 

def apply_imputation(
    df: pd.DataFrame,
    column_configs: Dict[str, Dict[str, Union[str, float, int, bool]]]
) -> pd.DataFrame:
    """
    Imputes missing values (NaN) in specified columns of a DataFrame based on
    per-column configurations. Can also delete rows based on missing values.

    Args:
        df: The input pandas DataFrame.
        column_configs: A dictionary where keys are column names and values are
                        dictionaries specifying the imputation method for that column.
                        Each inner dictionary must have a 'strategy' key ('mean',
                        'median', 'mode', 'constant', or 'delete_row'). If strategy is 'constant',
                        it must also include a 'value' key.
                        Example: {
                            'numeric_col': {'strategy': 'mean'},
                            'categorical_col': {'strategy': 'mode'},
                            'another_col': {'strategy': 'constant', 'value': 0},
                            'critical_col': {'strategy': 'delete_row'}
                        }

    Returns:
        A new DataFrame with missing values handled according to the configs.

    Raises:
        ValueError: If configuration is invalid (missing keys, bad strategy,
                    mismatched type for strategy, constant value missing).
        TypeError: If df is not a DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(column_configs, dict):
        raise TypeError("Input 'column_configs' must be a dictionary.")

    df_copy = df.copy()
    valid_strategies = ['mean', 'median', 'mode', 'constant', 'delete_row']
    numeric_strategies = ['mean', 'median']

    # --- Expand pattern-based configurations ---    
    effective_column_configs = {}
    # Process patterns first
    for config_key, config_value in column_configs.items():
        if not isinstance(config_value, dict) or 'strategy' not in config_value:
            # If it's not a valid config dict, it might be an old format or an error.
            # For now, we'll assume keys not matching patterns are direct column names to be processed later.
            if not (config_key.startswith("prefix:") or config_key.startswith("suffix:") or config_key.startswith("contains:")):
                effective_column_configs[config_key] = config_value # Keep it for exact matching pass
            else:
                 raise ValueError(f"Invalid configuration value for key '{config_key}'. Must be a dict with a 'strategy'.")
            continue

        # If we reach here, config_value is a dict with 'strategy'
        # Standardize strategy to lowercase and underscore
        strategy_to_apply = str(config_value['strategy']).lower().replace(" ", "_") 
        if strategy_to_apply not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy_to_apply}' in pattern config for '{config_key}'. Valid: {valid_strategies}")

        matched_cols = []
        if config_key.startswith("prefix:"):
            pattern = config_key.split("prefix:", 1)[1]
            matched_cols = [col for col in df_copy.columns if col.startswith(pattern)]
        elif config_key.startswith("suffix:"):
            pattern = config_key.split("suffix:", 1)[1]
            matched_cols = [col for col in df_copy.columns if col.endswith(pattern)]
        elif config_key.startswith("contains:"):
            pattern = config_key.split("contains:", 1)[1]
            matched_cols = [col for col in df_copy.columns if pattern in col]
        
        if matched_cols:
            print(f"Pattern '{config_key}' matched columns: {matched_cols}")
            for col_name in matched_cols:
                if col_name not in effective_column_configs: # Avoid overriding if an exact config is also coming
                    effective_column_configs[col_name] = config_value

    # Process exact column configurations (they take precedence)
    for config_key, config_value in column_configs.items():
        if not (config_key.startswith("prefix:") or config_key.startswith("suffix:") or config_key.startswith("contains:")):
            if config_key in df_copy.columns: # Ensure it's an actual column name
                 if not isinstance(config_value, dict) or 'strategy' not in config_value:
                     raise ValueError(f"Invalid configuration value for column '{config_key}'. Must be a dict with a 'strategy'.")
                 effective_column_configs[config_key] = config_value
            elif config_key not in effective_column_configs: # If it wasn't a column and not a pattern, raise error or warn
                 print(f"Warning: Configuration key '{config_key}' is not a valid column or recognized pattern. It will be ignored.")

    # --- Identify columns and rows for deletion first --- 
    cols_for_row_deletion = [
        col_name for col_name, config in effective_column_configs.items()
        if isinstance(config, dict) and str(config.get('strategy')).lower() == 'delete_row'
    ]
    
    rows_to_delete_idx = pd.Index([]) # Initialize empty index
    if cols_for_row_deletion:
        # Check if specified columns exist
        missing_del_cols = [col for col in cols_for_row_deletion if col not in df_copy.columns]
        if missing_del_cols:
             print(f"Warning: Columns specified for 'delete_row' not found: {missing_del_cols}. Skipping deletion for these.")
             # Filter out non-existent columns
             cols_for_row_deletion = [col for col in cols_for_row_deletion if col in df_copy.columns]

        if cols_for_row_deletion: # Proceed if valid columns remain
             # Find index of rows with NaN in *any* of these columns
             rows_to_delete_idx = df_copy[df_copy[cols_for_row_deletion].isnull().any(axis=1)].index
             print(f"Identified {len(rows_to_delete_idx)} rows for deletion based on NaNs in columns: {cols_for_row_deletion}")
             if not rows_to_delete_idx.empty:
                 df_copy.drop(index=rows_to_delete_idx, inplace=True)
                 print(f"Successfully deleted {len(rows_to_delete_idx)} rows.")

    # --- Apply other imputation strategies --- 
    imputed_cols_count = 0
    for col_name, config in effective_column_configs.items():
        # Skip columns marked for row deletion, as they are handled separately
        if col_name in cols_for_row_deletion:
            continue

        if col_name not in df_copy.columns:
            print(f"Warning: Column '{col_name}' in config not found. Skipping.")
            continue

        if not isinstance(config, dict) or 'strategy' not in config:
            raise ValueError(f"Invalid configuration for column '{col_name}'. Missing 'strategy'.")

        # Standardize strategy to lowercase and underscore here as well for the second loop
        strategy = str(config['strategy']).lower().replace(" ", "_")
        
        # Skip if strategy is 'delete_row' (already handled) or invalid
        if strategy == 'delete_row':
             continue
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}' for column '{col_name}'. Valid: {valid_strategies}")

        is_numeric = pd.api.types.is_numeric_dtype(df_copy[col_name])

        if strategy in numeric_strategies and not is_numeric:
            raise ValueError(f"Strategy '{strategy}' cannot be applied to non-numeric column '{col_name}'.")

        # Check if imputation is needed for this column
        if not df_copy[col_name].isnull().any():
            continue # Skip if no NaNs

        try:
            fill_value = None
            if strategy == 'mean':
                fill_value = df_copy[col_name].mean()
            elif strategy == 'median':
                fill_value = df_copy[col_name].median()
            elif strategy == 'mode':
                # Mode can return multiple values if they have the same frequency
                # We'll take the first one as the fill value
                mode_result = df_copy[col_name].mode()
                if not mode_result.empty:
                    fill_value = mode_result[0]
                else:
                    # Handle case where mode is empty (e.g., all unique values)
                    # Fallback strategy could be added here, or raise error/warning
                    print(f"Warning: Could not determine mode for column '{col_name}'. Skipping imputation.")
                    continue
            elif strategy == 'constant':
                if 'value' not in config:
                    raise ValueError(f"Missing 'value' in config for constant imputation of column '{col_name}'.")
                fill_value = config['value']
                # Ensure type compatibility for constant value if column is numeric
                if is_numeric:
                     try:
                         # Attempt conversion, will raise ValueError if incompatible
                         pd.to_numeric(fill_value)
                     except ValueError:
                         raise ValueError(f"Constant value '{fill_value}' is not compatible with numeric column '{col_name}'.")

            # Apply imputation
            if fill_value is not None:
                df_copy[col_name].fillna(fill_value, inplace=True)
                imputed_cols_count += 1
            else:
                 # Should not happen with current logic, but good practice
                 print(f"Warning: Could not determine fill value for column '{col_name}' with strategy '{strategy}'.")

        except Exception as e:
            raise RuntimeError(f"Error applying imputation ('{strategy}') to column '{col_name}': {e}") from e

    if imputed_cols_count > 0:
        print(f"Imputed missing values in {imputed_cols_count} column(s) according to configuration.")
    else:
        print("No missing values imputed based on the provided configuration and data.")

    return df_copy

def select_columns(
    df: pd.DataFrame,
    keep_columns: Optional[List[str]] = None,
    remove_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Selects or removes specific columns from a DataFrame.

    Args:
        df: The input pandas DataFrame.
        keep_columns: An optional list of column names to keep. All other columns
                      will be dropped. Cannot be used with 'remove_columns'.
        remove_columns: An optional list of column names to remove. Cannot be
                        used with 'keep_columns'.

    Returns:
        A new DataFrame containing only the selected columns or with specified
        columns removed.

    Raises:
        ValueError: If both keep_columns and remove_columns are provided,
                    if specified columns are not found, or if neither is provided.
        TypeError: If df is not a DataFrame or arguments are not lists/None.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if keep_columns is not None and not isinstance(keep_columns, list):
        raise TypeError("Input 'keep_columns' must be a list or None.")
    if remove_columns is not None and not isinstance(remove_columns, list):
        raise TypeError("Input 'remove_columns' must be a list or None.")

    if keep_columns is not None and remove_columns is not None:
        raise ValueError("Cannot provide both 'keep_columns' and 'remove_columns'.")

    df_copy = df.copy()
    original_columns = df_copy.columns.tolist()

    if keep_columns is not None:
        # Validate columns to keep exist
        missing_keep = [col for col in keep_columns if col not in original_columns]
        if missing_keep:
            raise ValueError(f"Columns specified in 'keep_columns' not found in DataFrame: {missing_keep}")
        # Select only the columns to keep
        print(f"Keeping columns: {keep_columns}")
        return df_copy[keep_columns]

    elif remove_columns is not None:
        # Validate columns to remove exist
        missing_remove = [col for col in remove_columns if col not in original_columns]
        if missing_remove:
            raise ValueError(f"Columns specified in 'remove_columns' not found in DataFrame: {missing_remove}")
        # Drop the specified columns
        print(f"Removing columns: {remove_columns}")
        return df_copy.drop(columns=remove_columns)

    else:
        # If neither is specified, maybe return the copy or raise error?
        # Let's raise an error for clarity, as the function call is ambiguous.
        raise ValueError("Must provide either 'keep_columns' or 'remove_columns'.")

def create_formula_column(
    df: pd.DataFrame,
    new_column_name: str,
    formula: str,
    pre_cast_to_str: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Creates a new column in the DataFrame based on a formula evaluated using pandas.eval.

    Args:
        df: The input pandas DataFrame.
        new_column_name: The name for the new column.
        formula: The formula string to evaluate. Column names can be used directly.
                 Example: 'column_a + column_b / 2' or 'log(column_c)'.
        pre_cast_to_str: An optional list of column names that should be cast
                         to string type before evaluating the formula. This is useful
                         for ensuring correct string concatenation if columns are numeric.

    Returns:
        A new DataFrame with the added formula column.

    Raises:
        ValueError: If the new column name already exists, the formula is invalid
                    (e.g., refers to non-existent columns, syntax errors), or
                    if a column in pre_cast_to_str does not exist.
        TypeError: If df is not a DataFrame.
        Exception: Catches potential errors during pandas.eval execution.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(new_column_name, str) or not new_column_name:
        raise ValueError("'new_column_name' must be a non-empty string.")
    if not isinstance(formula, str) or not formula:
        raise ValueError("'formula' must be a non-empty string.")
    if pre_cast_to_str is not None and not isinstance(pre_cast_to_str, list):
        raise TypeError("'pre_cast_to_str' must be a list of strings or None.")

    if new_column_name in df.columns:
        raise ValueError(f"Column '{new_column_name}' already exists in the DataFrame.")

    df_copy = df.copy()

    # Pre-cast specified columns to object dtype (Python strings)
    if pre_cast_to_str:
        for col_to_cast in pre_cast_to_str:
            if col_to_cast not in df_copy.columns:
                raise ValueError(f"Column '{col_to_cast}' specified in 'pre_cast_to_str' not found in DataFrame.")
            print(f"Pre-casting column '{col_to_cast}' to object dtype (Python strings) for formula evaluation.")
            df_copy[col_to_cast] = df_copy[col_to_cast].astype(str)

    operation_done_directly = False
    # Try to handle simple string concatenations directly first
    # Pattern 1: 'literal' + Col  (or "literal" + Col)
    match1 = re.fullmatch(r"\s*(['\"])(.*?)\1\s*\+\s*([a-zA-Z_][\w]*)\s*", formula)
    # Pattern 2: Col + 'literal' (or Col + "literal")
    match2 = re.fullmatch(r"\s*([a-zA-Z_][\w]*)\s*\+\s*(['\"])(.*?)\2\s*", formula)
    # Pattern 3: Col1 + Col2
    match3 = re.fullmatch(r"\s*([a-zA-Z_][\w]*)\s*\+\s*([a-zA-Z_][\w]*)\s*", formula)

    if match1:
        literal = match1.group(2)
        col_name = match1.group(3)
        if col_name in df_copy.columns:
            try:
                # Ensure the column is string type (object dtype for broad compatibility here)
                if not isinstance(df_copy[col_name].dtype, object) and df_copy[col_name].dtype.name != 'string':
                    df_copy[col_name] = df_copy[col_name].astype(str)
                df_copy[new_column_name] = literal + df_copy[col_name]
                operation_done_directly = True
                print(f"Successfully created column '{new_column_name}' via direct literal + column string concatenation.")
            except Exception as e:
                print(f"Direct literal + column concatenation failed: {e}. Falling back to df.eval().")
        else:
            print(f"Column '{col_name}' in formula not found for direct concatenation. Falling back to df.eval().")
    elif match2:
        col_name = match2.group(1)
        literal = match2.group(3)
        if col_name in df_copy.columns:
            try:
                if not isinstance(df_copy[col_name].dtype, object) and df_copy[col_name].dtype.name != 'string':
                    df_copy[col_name] = df_copy[col_name].astype(str)
                df_copy[new_column_name] = df_copy[col_name] + literal
                operation_done_directly = True
                print(f"Successfully created column '{new_column_name}' via direct column + literal string concatenation.")
            except Exception as e:
                print(f"Direct column + literal concatenation failed: {e}. Falling back to df.eval().")
        else:
            print(f"Column '{col_name}' in formula not found for direct concatenation. Falling back to df.eval().")
    elif match3:
        col1_name = match3.group(1)
        col2_name = match3.group(2)
        if col1_name in df_copy.columns and col2_name in df_copy.columns:
            try:
                if not isinstance(df_copy[col1_name].dtype, object) and df_copy[col1_name].dtype.name != 'string':
                    df_copy[col1_name] = df_copy[col1_name].astype(str)
                if not isinstance(df_copy[col2_name].dtype, object) and df_copy[col2_name].dtype.name != 'string':
                    df_copy[col2_name] = df_copy[col2_name].astype(str)
                df_copy[new_column_name] = df_copy[col1_name] + df_copy[col2_name]
                operation_done_directly = True
                print(f"Successfully created column '{new_column_name}' via direct column + column string concatenation.")
            except Exception as e:
                print(f"Direct column + column concatenation failed: {e}. Falling back to df.eval().")
        else:
            print(f"One or both columns ('{col1_name}', '{col2_name}') in formula not found for direct concatenation. Falling back to df.eval().")

    if not operation_done_directly:
        print(f"Formula '{formula}' not handled by direct concatenation, attempting with df.eval().")
        try:
            # Use DataFrame.eval to compute the new column
            # The result is assigned directly using the column name assignment syntax
            # within eval, which is generally efficient.
            df_copy[new_column_name] = df_copy.eval(formula, engine='python') # Use python engine for wider compatibility initially
            # Note: engine='numexpr' might be faster for purely numeric ops but has limitations.
            print(f"Successfully created column '{new_column_name}'.")
        except KeyError as e:
            # More specific error for missing columns
            raise ValueError(f"Formula evaluation error: Column not found - {e}. Formula: '{formula}'") from e
        except pd.errors.UndefinedVariableError as e: # Corrected UndefinedVariableError path
             # Catch errors if a variable (column) in the formula doesn't exist
             raise ValueError(f"Formula evaluation error: Column not found - {e}. Formula: '{formula}'") from e
        except Exception as e:
            # Catch other potential errors during eval (syntax errors, unsupported ops, etc.)
            raise ValueError(f"Error evaluating formula '{formula}': {e}") from e

    return df_copy 

def filter_rows_by_condition(
    df: pd.DataFrame,
    query_string: str
) -> pd.DataFrame:
    """
    Filters rows of a DataFrame based on a query string using pandas.query.

    Args:
        df: The input pandas DataFrame.
        query_string: The query string to evaluate for filtering.
                      Example: 'Column1 > 0.5 and (Column2 == "BABY" or Column3 < 5)'

    Returns:
        A new DataFrame containing only the rows that satisfy the condition.

    Raises:
        ValueError: If the query string is invalid (syntax error, refers to
                    non-existent columns, etc.).
        TypeError: If df is not a DataFrame.
        Exception: Catches potential errors during pandas.query execution.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not isinstance(query_string, str) or not query_string:
        raise ValueError("'query_string' must be a non-empty string.")

    try:
        # Use DataFrame.query to filter rows
        filtered_df = df.query(query_string, engine='python') # Use python engine for wider compatibility
        print(f"Filtered DataFrame using query: '{query_string}'. Original rows: {len(df)}, Filtered rows: {len(filtered_df)}.")
        return filtered_df
    except pd.core.computation.ops.UndefinedVariableError as e:
        # Catch errors if a variable (column) in the query doesn't exist
        raise ValueError(f"Query error: Column not found - {e}. Query: '{query_string}'") from e
    except Exception as e:
        # Catch other potential errors during query (syntax errors, etc.)
        raise ValueError(f"Error evaluating query '{query_string}': {e}") from e

def join_datasets(
    anchor_df: pd.DataFrame,
    adjoining_df: pd.DataFrame,
    anchor_on: Union[str, List[str], None] = None,
    adjoining_on: Union[str, List[str], None] = None,
    adjoining_columns_to_keep: Optional[List[str]] = None,
    how: str = 'left'
) -> pd.DataFrame:
    """
    Joins two DataFrames based on specified columns, or concatenates them horizontally.

    Args:
        anchor_df: The left DataFrame (primary table).
        adjoining_df: The right DataFrame (table to join).
        anchor_on: Column name or list of column names in anchor_df to join on.
                   Not required if how='concat'.
        adjoining_on: Column name or list of column names in adjoining_df to join on.
                      Not required if how='concat'.
        adjoining_columns_to_keep: List of column names from adjoining_df to include
                                     in the final merged DataFrame. The join column(s)
                                     specified in 'adjoining_on' must be included here.
                                     If None and how='concat', all columns are kept.
        how: Type of merge to be performed. Defaults to 'left'.
             Options: 'left', 'right', 'outer', 'inner', 'concat'.
             - 'concat': Horizontal concatenation by row index (no join keys needed).

    Returns:
        A new DataFrame resulting from the join operation.

    Raises:
        ValueError: If join columns are not found, if 'adjoining_on' columns are missing
                    from 'adjoining_columns_to_keep', or if an invalid 'how' is specified.
        TypeError: If inputs are not DataFrames or column specs are invalid types.
        Exception: Catches potential errors during pandas.merge execution.
    """
    if not isinstance(anchor_df, pd.DataFrame) or not isinstance(adjoining_df, pd.DataFrame):
        raise TypeError("Both 'anchor_df' and 'adjoining_df' must be pandas DataFrames.")

    valid_how = ['left', 'right', 'outer', 'inner', 'concat']
    if how not in valid_how:
        raise ValueError(f"Invalid join type '{how}'. Valid options are: {valid_how}")

    # Handle concatenation mode (no join keys needed)
    if how == 'concat':
        # Select columns from adjoining dataframe if specified
        if adjoining_columns_to_keep is not None:
            if not isinstance(adjoining_columns_to_keep, list):
                raise TypeError("'adjoining_columns_to_keep' must be a list of strings or None.")
            missing_keep = [col for col in adjoining_columns_to_keep if col not in adjoining_df.columns]
            if missing_keep:
                raise ValueError(f"Columns specified in 'adjoining_columns_to_keep' not found in adjoining DataFrame: {missing_keep}")
            adjoining_df_selected = adjoining_df[adjoining_columns_to_keep]
        else:
            adjoining_df_selected = adjoining_df

        # Check that both dataframes have the same number of rows
        if len(anchor_df) != len(adjoining_df_selected):
            raise ValueError(f"Cannot concatenate DataFrames with different row counts: "
                           f"anchor has {len(anchor_df)} rows, adjoining has {len(adjoining_df_selected)} rows")

        # Reset indices to ensure alignment
        anchor_reset = anchor_df.reset_index(drop=True)
        adjoining_reset = adjoining_df_selected.reset_index(drop=True)

        # Concatenate horizontally
        try:
            concatenated_df = pd.concat([anchor_reset, adjoining_reset], axis=1)
            print(f"Successfully concatenated datasets horizontally. Resulting shape: {concatenated_df.shape}")
            return concatenated_df
        except Exception as e:
            raise RuntimeError(f"Error during dataset concatenation: {e}") from e

    # Standard join mode - validate required parameters
    if anchor_on is None or adjoining_on is None:
        raise ValueError("'anchor_on' and 'adjoining_on' are required for join operations (how != 'concat')")
    if adjoining_columns_to_keep is None:
        raise ValueError("'adjoining_columns_to_keep' is required for join operations (how != 'concat')")

    if not isinstance(anchor_on, (str, list)) or not isinstance(adjoining_on, (str, list)):
        raise TypeError("'anchor_on' and 'adjoining_on' must be a string or a list of strings.")
    if not isinstance(adjoining_columns_to_keep, list):
        raise TypeError("'adjoining_columns_to_keep' must be a list of strings.")

    # --- Validate columns ---
    anchor_cols = [anchor_on] if isinstance(anchor_on, str) else anchor_on
    adjoining_cols = [adjoining_on] if isinstance(adjoining_on, str) else adjoining_on
    keep_cols_set = set(adjoining_columns_to_keep)  # Use set for validation
    join_cols_in_adjoining = set(adjoining_cols)

    missing_anchor = [col for col in anchor_cols if col not in anchor_df.columns]
    if missing_anchor:
        raise ValueError(f"Join column(s) not found in anchor DataFrame: {missing_anchor}")

    missing_adjoining_base = [col for col in adjoining_cols if col not in adjoining_df.columns]
    if missing_adjoining_base:
        raise ValueError(f"Join column(s) not found in adjoining DataFrame: {missing_adjoining_base}")

    missing_keep = [col for col in keep_cols_set if col not in adjoining_df.columns]
    if missing_keep:
        raise ValueError(f"Columns specified in 'adjoining_columns_to_keep' not found in adjoining DataFrame: {missing_keep}")

    # Ensure the join keys for the adjoining table are included in the columns to keep
    if not join_cols_in_adjoining.issubset(keep_cols_set):
         missing_join_keys = list(join_cols_in_adjoining - keep_cols_set)
         raise ValueError(f"The adjoining join column(s) {missing_join_keys} must also be included in 'adjoining_columns_to_keep'.")

    # Select only the necessary columns from the adjoining dataframe
    # Preserve the order specified in adjoining_columns_to_keep
    adjoining_df_selected = adjoining_df[adjoining_columns_to_keep]

    try:
        # Perform the merge
        merged_df = pd.merge(
            anchor_df,
            adjoining_df_selected,
            left_on=anchor_on,
            right_on=adjoining_on,
            how=how,
            suffixes= ('', '_adj') # Add suffix to overlapping columns from adjoining df, except join keys
        )
        print(f"Successfully joined datasets using '{how}' join. Resulting rows: {len(merged_df)}")
        return merged_df

    except Exception as e:
        raise RuntimeError(f"Error during dataset join operation: {e}") from e

def _find_cols_by_pattern(df_columns: List[str], pattern: str, pattern_type: str) -> List[str]:
    """Helper to find columns by pattern."""
    pattern_type = pattern_type.lower()
    if pattern_type == "prefix":
        return [col for col in df_columns if col.startswith(pattern)]
    elif pattern_type == "suffix":
        return [col for col in df_columns if col.endswith(pattern)]
    elif pattern_type == "contains":
        return [col for col in df_columns if pattern in col]
    else:
        raise ValueError(f"Invalid pattern type: {pattern_type}. Must be 'prefix', 'suffix', or 'contains'.")

def calculate_grouped_difference(
    df: pd.DataFrame,
    group_by_col: str,
    value_col_pattern: str,
    value_col_pattern_type: str,
    condition_col: str,
    condition_val_a: any,
    condition_val_b: any,
    new_column_name: str,
    columns_to_keep_from_condition_b: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates differences for multiple value columns based on conditions within groups,
    keeps specified additional columns from condition B, and filters for complete groups.
    Result = (Value where condition=A) - (Value where condition=B) for each value column.

    Args:
        df: The input pandas DataFrame.
        group_by_col: Column name to group by (e.g., 'SUBJLABEL').
        value_col_pattern: Pattern to identify value columns (e.g., "QIDS_SR_").
        value_col_pattern_type: Type of pattern ("prefix", "suffix", "contains").
        condition_col: Column name containing conditions (e.g., 'EVENTNAME').
        condition_val_a: Value in condition_col for the minuend (e.g., 'Week 8').
        condition_val_b: Value in condition_col for the subtrahend (e.g., 'Baseline').
        new_column_name: The prefix for the new difference columns.
        columns_to_keep_from_condition_b: Optional list of column names from condition_val_b
                                          data to include in the output.

    Returns:
        A pandas DataFrame indexed by group_by_col, including kept columns and
        difference columns. Returns an empty DataFrame if no complete groups or
        no value columns are found.

    Raises:
        ValueError: If specified columns/patterns are invalid or not found.
        TypeError: If df is not a DataFrame or value columns are not numeric.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    required_cols = [group_by_col, condition_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found in DataFrame: {missing_cols}")

    if columns_to_keep_from_condition_b is None:
        columns_to_keep_from_condition_b = []
    
    missing_keep_cols = [col for col in columns_to_keep_from_condition_b if col not in df.columns]
    if missing_keep_cols:
        raise ValueError(f"Columns in 'columns_to_keep_from_condition_b' not found in DataFrame: {missing_keep_cols}")

    value_cols_to_process = _find_cols_by_pattern(df.columns.tolist(), value_col_pattern, value_col_pattern_type)
    if not value_cols_to_process:
        print(f"Warning: No value columns found matching pattern '{value_col_pattern}' with type '{value_col_pattern_type}'. Returning empty DataFrame.")
        return pd.DataFrame()

    for v_col in value_cols_to_process:
        if not pd.api.types.is_numeric_dtype(df[v_col]):
            raise TypeError(f"Value column '{v_col}' identified by pattern must be numeric to calculate difference.")

    df_a = df[df[condition_col] == condition_val_a]
    df_b = df[df[condition_col] == condition_val_b]

    groups_a = set(df_a[group_by_col].unique())
    groups_b = set(df_b[group_by_col].unique())
    complete_groups = sorted(list(groups_a.intersection(groups_b))) # Sorted for consistent output

    if not complete_groups:
        print(f"No groups found with both condition '{condition_val_a}' and '{condition_val_b}'. Returning empty DataFrame.")
        # Define expected columns for empty DataFrame
        diff_col_names = [f"{col}_diff" for col in value_cols_to_process]
        
        # Ensure group_by_col is structured to be first if not elsewhere, then other kept columns, then diffs
        # This ensures the column order is somewhat predictable for an empty DataFrame.
        final_empty_cols = []
        temp_keep_cols = list(columns_to_keep_from_condition_b) # Make a copy

        if group_by_col in temp_keep_cols:
            # If group_by_col is explicitly asked to be kept, put it first from this list for consistency
            final_empty_cols.append(group_by_col)
            temp_keep_cols.remove(group_by_col)
            final_empty_cols.extend(temp_keep_cols) # Add remaining kept columns
        else:
            # If group_by_col was not in keep_cols, add it as the primary identifier
            final_empty_cols.append(group_by_col)
            final_empty_cols.extend(temp_keep_cols) # Add all specified kept columns
        
        final_empty_cols.extend(diff_col_names)
        
        # Create empty DataFrame with all columns. group_by_col will be a regular column.
        return pd.DataFrame(columns=final_empty_cols)

    df_a_complete = df_a[df_a[group_by_col].isin(complete_groups)]
    df_b_complete = df_b[df_b[group_by_col].isin(complete_groups)]

    # Use groupby().first() to ensure one row per group per condition before processing
    # This handles potential duplicates if (group_by_col, condition_col) is not unique
    data_a_grouped = df_a_complete.groupby(group_by_col, observed=True).first()
    data_b_grouped = df_b_complete.groupby(group_by_col, observed=True).first()
    
    # Align data_a_grouped and data_b_grouped to complete_groups to ensure proper subtraction
    # Reindex will introduce NaNs for groups not in data_a_grouped/data_b_grouped, but these should be complete_groups
    # This step might be redundant if groupby.first() already aligns with complete_groups, but explicit is safer.
    # However, since we filtered by complete_groups already, groupby().first() should yield indices that are subsets of complete_groups.
    # If a group from complete_groups was entirely filtered out by .first() due to all NaNs in grouping keys (unlikely for group_by_col), reindex handles this.

    data_a_values = data_a_grouped.reindex(complete_groups)[value_cols_to_process]
    data_b_values = data_b_grouped.reindex(complete_groups)[value_cols_to_process]

    diff_df = data_a_values.subtract(data_b_values)

    # New naming logic (Option A style)
    new_diff_column_names = []
    for col in value_cols_to_process:
        base_col_name = col
        if value_col_pattern_type == "prefix" and col.startswith(value_col_pattern):
            base_col_name = col[len(value_col_pattern):]
        elif value_col_pattern_type == "suffix" and col.endswith(value_col_pattern):
            base_col_name = col[:-len(value_col_pattern)]
        elif value_col_pattern_type == "contains":
            # For 'contains', a simple replace might be okay if the pattern is unique enough.
            # Replacing only the first occurrence to be safer if pattern could appear multiple times.
            base_col_name = col.replace(value_col_pattern, "", 1) 
        
        # Sanitize base_col_name further? (e.g., if it becomes empty after stripping)
        if not base_col_name: # If stripping pattern left nothing (e.g. col was exactly the pattern)
            base_col_name = "value" # Fallback name part
            
        new_diff_column_names.append(f"{new_column_name}{base_col_name}")
    diff_df.columns = new_diff_column_names

    result_df = diff_df
    
    if columns_to_keep_from_condition_b:
        # Create a copy of the list to avoid modifying the original argument
        cols_to_select_for_kept_data = list(columns_to_keep_from_condition_b)
        
        # If group_by_col is in the list of columns to keep, 
        # remove it from selection here because it's the index and will be added back by reset_index().
        # Trying to select it as a column when it's the index causes the error.
        if group_by_col in cols_to_select_for_kept_data:
            cols_to_select_for_kept_data.remove(group_by_col)

        # Only proceed if there are still other columns to select after potentially removing group_by_col
        if cols_to_select_for_kept_data: 
            kept_data = data_b_grouped.reindex(complete_groups)[cols_to_select_for_kept_data]
            result_df = kept_data.join(diff_df) # Join on the group_by_col index
        # If cols_to_select_for_kept_data is empty (e.g., only group_by_col was asked to be kept),
        # result_df remains diff_df, and group_by_col will be added by reset_index() anyway.

    print(f"Calculated grouped difference for {len(value_cols_to_process)} value columns, for {len(complete_groups)} groups.")
    return result_df.reset_index() # Reset index to make group_by_col a column