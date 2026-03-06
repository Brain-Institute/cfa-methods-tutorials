"""
Data management functionality for IMBiotype
"""

import os
import shutil
import json
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy.io import loadmat
from datetime import datetime
import re
import csv

class DataManager:
    """
    Manages data import and feature creation/manipulation.
    Operates primarily on data within the 'processed_data' directory,
    while preserving original files in 'raw_data'.
    """
    
    def __init__(self, study_path: str):
        """
        Initialize the data manager
        
        Args:
            study_path: Path to the main study directory
        """
        if not os.path.isdir(study_path):
             raise FileNotFoundError(f"Study directory not found: {study_path}")

        self.study_path = study_path
        self.raw_data_path = os.path.join(study_path, "raw_data")
        self.processed_data_path = os.path.join(study_path, "processed_data") # Path to processed data
        self.features_path = os.path.join(study_path, "features.json")
        
        # Ensure necessary directories exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Create features directory for storing individual feature CSV files
        # Stays at the study level
        self.features_dir = os.path.join(study_path, "features")
        os.makedirs(self.features_dir, exist_ok=True)

        # --- Migration Check --- >
        self._check_and_migrate_old_study()
        # < --- End Migration Check ---

        self._load_features()
        self._cleaning_history: Dict[str, List[str]] = {}
        
    def _check_and_migrate_old_study(self):
        """
        Checks if the study is in the old format (processed_data missing or empty
        while raw_data exists and is not empty) and copies raw_data contents
        to processed_data if necessary.
        """
        processed_exists = os.path.exists(self.processed_data_path)
        processed_is_empty = not processed_exists or not os.listdir(self.processed_data_path)
        raw_exists = os.path.exists(self.raw_data_path)
        raw_is_not_empty = raw_exists and os.listdir(self.raw_data_path)

        if raw_is_not_empty and processed_is_empty:
            print(f"Detected old study format. Migrating data from '{self.raw_data_path}' to '{self.processed_data_path}'...")
            migrated_count = 0
            skipped_count = 0
            try:
                for filename in os.listdir(self.raw_data_path):
                    source_path = os.path.join(self.raw_data_path, filename)
                    dest_path = os.path.join(self.processed_data_path, filename)

                    # Check if it's a file and has a supported extension
                    if os.path.isfile(source_path) and filename.lower().endswith(('.csv', '.mat')):
                        print(f"  Copying {filename}...")
                        shutil.copy2(source_path, dest_path)
                        migrated_count += 1
                    elif os.path.isfile(source_path):
                         print(f"  Skipping unsupported file: {filename}")
                         skipped_count += 1
                    # Optionally handle directories if needed

                print(f"Migration complete. Copied {migrated_count} files, skipped {skipped_count}.")

            except Exception as e:
                print(f"Error during study migration: {e}")
                # Handle error case - maybe raise an exception or proceed with caution?
                # For now, just print the error.

    def _load_features(self):
        """Load or initialize features"""
        if os.path.exists(self.features_path):
            try:
                with open(self.features_path, 'r') as f:
                    self._features = json.load(f)

                # Export all features to CSV to ensure the features directory is synchronized
                # This uses get_feature_data which now reads from processed_data
                self.export_all_features_to_csv()
            except json.JSONDecodeError:
                 print(f"Warning: Could not decode {self.features_path}. Initializing empty features.")
                 self._features = {}
                 self._save_features()
            except Exception as e:
                 print(f"Error loading features: {e}. Initializing empty features.")
                 self._features = {}
                 self._save_features()

        else:
            self._features = {}
            self._save_features()
            
    def _save_features(self):
        """Save features to disk"""
        try:
            with open(self.features_path, 'w') as f:
                json.dump(self._features, f, indent=2)
        except Exception as e:
             print(f"Error saving features to {self.features_path}: {e}")
            
    def _get_full_path(self, filename: str, use_raw: bool = False) -> str:
        """
        Constructs the full path to a dataset file.

        Args:
            filename: The name of the dataset file.
            use_raw: If True, returns path from raw_data. Otherwise, processed_data.

        Returns:
            The full path to the file.
        """
        base_path = self.raw_data_path if use_raw else self.processed_data_path
        return os.path.join(base_path, filename)

    def import_dataset(self, file_path: str, 
                      dataset_name: Optional[str] = None) -> str:
        """
        Import a dataset file into the study. Copies the file to both
        raw_data (for backup) and processed_data (for modification).
        
        Args:
            file_path: Path to the data file
            dataset_name: Optional name for the dataset
            
        Returns:
            Name of the imported dataset
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if dataset_name is None:
            dataset_name = os.path.basename(file_path)
            
        # Ensure the dataset name has the correct extension
        source_ext = os.path.splitext(file_path)[1].lower()
        if not source_ext:
            try:
                pd.read_csv(file_path, nrows=5)
                source_ext = '.csv'
            except Exception:
                try:
                    loadmat(file_path)
                    source_ext = '.mat'
                except Exception:
                    raise ValueError("Could not determine file type. Must be CSV or MAT.")
        
        if not dataset_name.lower().endswith(('.csv', '.mat')):
            dataset_name = f"{dataset_name}{source_ext}"
            
        # Define destination paths
        dest_raw_path = self._get_full_path(dataset_name, use_raw=True)
        dest_processed_path = self._get_full_path(dataset_name, use_raw=False)

        # Track if this is an overwrite
        overwritten = os.path.exists(dest_raw_path) or os.path.exists(dest_processed_path)

        # Copy file to raw_data directory (backup)
        try:
             shutil.copy2(file_path, dest_raw_path)
        except Exception as e:
             raise IOError(f"Failed to copy file to raw_data: {e}")
        
        # Copy file to processed_data directory (working copy)
        try:
             shutil.copy2(file_path, dest_processed_path)
        except Exception as e:
             # Clean up raw copy if processed copy fails
             if os.path.exists(dest_raw_path):
                 os.remove(dest_raw_path)
             raise IOError(f"Failed to copy file to processed_data: {e}")
        
        # Validate the processed file can be loaded
        try:
            self._validate_dataset(dest_processed_path)
        except Exception as e:
             # Clean up both copies if validation fails
             if os.path.exists(dest_raw_path):
                 os.remove(dest_raw_path)
             if os.path.exists(dest_processed_path):
                 os.remove(dest_processed_path)
             raise # Re-raise the validation error
        
        return dataset_name, overwritten

    def _validate_dataset(self, file_path: str) -> None:
        """
        Validate that a dataset file can be loaded (checks file at given path).

        Args:
            file_path: Path to the dataset file to validate.
        """
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.csv':
                self._validate_csv_structure(file_path)
                pd.read_csv(file_path, nrows=5) # Read only a few rows for validation
            elif ext == '.mat':
                loadmat(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except ValueError:
            raise
        except Exception as e:
            # Do not clean up here, let the caller handle cleanup based on context
            raise ValueError(f"Invalid dataset file ({os.path.basename(file_path)}): {str(e)}")

    def _validate_csv_structure(self, file_path: str, max_rows: int = 1000) -> None:
        """
        Validate CSV structural integrity using stdlib csv module.
        Detects mixed delimiters, inconsistent column counts, empty files, etc.

        Args:
            file_path: Path to the CSV file.
            max_rows: Max data rows to check (default 1000).

        Raises:
            ValueError: If structural problems are found.
        """
        # 1. Empty file check
        with open(file_path, 'r', newline='', errors='replace') as f:
            sample = f.read(8192)
        if not sample.strip():
            raise ValueError("CSV file is empty")

        # 2. Detect delimiter
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ','

        # 3. Header check
        with open(file_path, 'r', newline='', errors='replace') as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty")

            if not header or all(col.strip() == '' for col in header):
                raise ValueError("CSV header row is empty")

            expected_cols = len(header)

            # 4. No data rows check
            try:
                first_row = next(reader)
            except StopIteration:
                raise ValueError("CSV has header but no data rows")

            # 5. Row-by-row structural check
            alt_delimiters = [d for d in [',', ';', '\t', '|'] if d != delimiter]
            delimiter_names = {',': 'comma', ';': 'semicolon', '\t': 'tab', '|': 'pipe'}

            rows_to_check = [first_row]
            for i, row in enumerate(reader):
                rows_to_check.append(row)
                if i >= max_rows - 2:  # -2 because first_row already consumed
                    break

            for i, row in enumerate(rows_to_check):
                row_num = i + 2  # 1-indexed, header is row 1
                # Skip blank lines (trailing newlines, blank rows)
                if len(row) == 0 or (len(row) == 1 and row[0].strip() == ''):
                    continue
                if len(row) != expected_cols:
                    # Re-read the raw line to try alternative delimiters
                    raw_line = delimiter.join(row)
                    found_alt = False
                    for alt_d in alt_delimiters:
                        alt_cols = len(next(csv.reader([raw_line], delimiter=alt_d)))
                        if alt_cols == expected_cols:
                            alt_name = delimiter_names.get(alt_d, repr(alt_d))
                            expected_name = delimiter_names.get(delimiter, repr(delimiter))
                            raise ValueError(
                                f"Mixed delimiters: row {row_num} uses "
                                f"{alt_name} instead of {expected_name}"
                            )
                    raise ValueError(
                        f"Inconsistent column count at row {row_num}: "
                        f"expected {expected_cols}, found {len(row)}"
                    )
            
    def _save_feature_to_csv(self, feature_name: str) -> None:
        """
        Save a feature's data to a CSV file in the features directory.
        Data is sourced based on the feature definition (from processed_data).
        
        Args:
            feature_name: Name of the feature to save
        """
        if feature_name not in self._features:
            print(f"Warning: Cannot save feature '{feature_name}' to CSV, definition not found.")
            return
            
        try:
            # Get the feature data (which now loads from processed_data based source)
            df = self.get_feature_data(feature_name)
            
            # Create a sanitized filename
            temp_name_chars = []
            for char_in_loop in feature_name: # char_in_loop to avoid conflict if char was a var name
                if char_in_loop.isalnum() or char_in_loop == '_' or char_in_loop == '-':
                    temp_name_chars.append(char_in_loop)
                else:
                    temp_name_chars.append('_')
            temp_name = "".join(temp_name_chars)

            # Step 2: Replace multiple underscores with a single underscore
            safe_name = re.sub(r'_+', '_', temp_name)

            # Step 3: Remove leading/trailing underscores and ensure not empty
            safe_name = safe_name.strip('_')
            if not safe_name: # Handle case where feature_name was all invalid characters
                safe_name = 'untitled_feature'
            
            csv_path = os.path.join(self.features_dir, f"{safe_name}.csv")
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Warning: Failed to save feature '{feature_name}' to CSV: {str(e)}")

    def create_feature(self, name: str, dataset: str, columns: List[str]) -> Dict:
        """
        Create a new feature from selected columns in a dataset.
        Validates columns against the *processed* version of the dataset.
        If a feature with the same name exists, it will be overwritten.
        
        Args:
            name: Name for the feature
            dataset: Name of the source dataset (in processed_data)
            columns: List of column names to include
            
        Returns:
            Feature metadata
        """
        # Check if feature exists and remove old definition if overwriting
        if name in self._features:
            pass  # Silently overwrite existing feature
            # Consider deleting the old CSV file as well?
            # safe_name = '_'.join(c for c in name if c.isalnum() or c in ('_', '-')).rstrip()
            # old_csv_path = os.path.join(self.features_dir, f"{safe_name}.csv")
            # if os.path.exists(old_csv_path):
            #     try: os.remove(old_csv_path)
            #     except: pass # Ignore deletion errors
            del self._features[name]
            
        # Validate columns exist in the processed dataset
        processed_file_path = self._get_full_path(dataset, use_raw=False)
        if not os.path.exists(processed_file_path):
             raise FileNotFoundError(f"Processed dataset not found: {dataset}")
            
        # Load processed dataset preview to validate columns
        try:
            df = self.preview_dataset(dataset) # preview uses processed path
            if df is None:
                 raise ValueError(f"Could not load processed dataset for validation: {dataset}")
        except Exception as e:
             raise ValueError(f"Error loading processed dataset {dataset} for validation: {e}")
            
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in processed dataset {dataset}: {missing_cols}")
            
        # Create feature metadata
        feature = {
            "name": name,
            "dataset": dataset, # Store the source dataset name
            "columns": columns,
            "created_at": datetime.now().isoformat()
        }
        
        # Store feature definition
        self._features[name] = feature
        
        # Save features JSON catalog
        self._save_features()
        
        # Save feature data to CSV in features directory
        # This implicitly uses the processed data because get_feature_data does
        self._save_feature_to_csv(name)
        
        return feature
        
    def get_dataset_columns(self, dataset: str) -> List[str]:
        """
        Get all columns from a dataset (from processed_data).
        
        Args:
            dataset: Name of the dataset
            
        Returns:
            List of column names
        """
        file_path = self._get_full_path(dataset, use_raw=False) # Use processed path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed dataset file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.csv':
                # Read only headers to get columns quickly
                df_cols = pd.read_csv(file_path, nrows=0).columns.tolist()
                return df_cols
            elif ext == '.mat':
                mat_data = loadmat(file_path)
                # Filter out system keys like '__header__', '__version__', '__globals__'
                return [k for k in mat_data.keys() if not k.startswith('__')]
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
             raise IOError(f"Error reading columns from {file_path}: {e}")
            
    def find_columns_by_pattern(self, dataset: str, 
                              pattern: str, 
                              pattern_type: str = "prefix") -> List[str]:
        """
        Find columns matching a pattern in the processed dataset.
        
        Args:
            dataset: Name of the dataset
            pattern: Pattern to match
            pattern_type: Type of pattern ("prefix", "suffix", or "contains")
            
        Returns:
            List of matching column names
        """
        columns = self.get_dataset_columns(dataset) # Gets columns from processed data
        
        if pattern_type == "prefix":
            return [col for col in columns if str(col).startswith(pattern)]
        elif pattern_type == "suffix":
            return [col for col in columns if str(col).endswith(pattern)]
        elif pattern_type == "contains":
            return [col for col in columns if pattern in str(col)]
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")
            
    def get_feature_data(self, feature_name: str) -> pd.DataFrame:
        """
        Get data for a feature. Loads the source dataset from *processed_data*
        and extracts the specified columns.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            DataFrame containing the feature data
        """
        if feature_name not in self._features:
            raise ValueError(f"Feature not found: {feature_name}")
            
        feature_info = self._features[feature_name]
        dataset_name = feature_info["dataset"]
        columns = feature_info["columns"]
        
        # Load the source dataset from processed_data
        file_path = self._get_full_path(dataset_name, use_raw=False)
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Source dataset for feature '{feature_name}' not found in processed_data: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.csv':
                df = pd.read_csv(file_path, usecols=columns)
            elif ext == '.mat':
                mat_data = loadmat(file_path)
                # Construct DataFrame from selected columns (assuming they are numpy arrays)
                data_dict = {col: mat_data[col].flatten() for col in columns if col in mat_data and isinstance(mat_data[col], np.ndarray)}
                # Check if all columns were found and were arrays
                if len(data_dict) != len(columns):
                     missing = set(columns) - set(data_dict.keys())
                     print(f"Warning: Could not load columns {missing} from MAT file {dataset_name} as numpy arrays.")
                df = pd.DataFrame(data_dict)
            else:
                raise ValueError(f"Unsupported file type for feature source: {ext}")

            # Ensure columns are in the order specified in the feature definition
            df = df[columns]
            return df

        except KeyError as e:
             raise ValueError(f"Column not found in dataset {dataset_name} when loading feature '{feature_name}': {e}")
        except Exception as e:
             raise IOError(f"Error loading data for feature '{feature_name}' from {file_path}: {e}")
            
    def get_features(self) -> Dict[str, Dict]:
        """Get all feature definitions"""
        return self._features.copy()
        
    def get_raw_datasets(self) -> List[str]:
        """
        Get list of all datasets in the raw_data directory.
        """
        if not os.path.exists(self.raw_data_path):
            return []
        return [f for f in os.listdir(self.raw_data_path)
                if os.path.isfile(os.path.join(self.raw_data_path, f))
                and f.lower().endswith(('.csv', '.mat'))]

    def get_datasets(self) -> List[str]:
        """
        Get list of all datasets currently available for processing (in processed_data).
        """
        if not os.path.exists(self.processed_data_path):
            return []
        return [f for f in os.listdir(self.processed_data_path)
                if os.path.isfile(os.path.join(self.processed_data_path, f))
                and f.lower().endswith(('.csv', '.mat'))] # Filter for supported types
        
    def preview_dataset(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Preview a dataset's contents (from processed_data).
        
        Args:
            filename: Name of the dataset to preview.
            
        Returns:
            DataFrame with preview data (first 50 rows) or None if error.
        """
        if not filename:
             return None
            
        file_path = self._get_full_path(filename, use_raw=False) # Use processed path
        if not os.path.exists(file_path):
             print(f"Warning: Processed dataset file not found for preview: {file_path}")
             return None
            
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.csv':
                # Read only the first N rows for preview
                return pd.read_csv(file_path, nrows=50)
            elif ext == '.mat':
                # For MAT files, load all data but maybe select top N variables?
                # Or just load all, as previewing structure might be more important.
                # Let's load all variables but wrap in DataFrame for consistency.
                mat_data = loadmat(file_path)
                # Create a dictionary focusing on array data, limit number of elements shown?
                preview_dict = {}
                for k, v in mat_data.items():
                    if k.startswith('__'): continue # Skip metadata
                    if isinstance(v, np.ndarray):
                         # Limit rows/cols? For now, just flatten and take first 50 elements
                         preview_dict[k] = v.flatten()[:50]
                    else:
                         preview_dict[k] = [v] # Wrap non-array data

                # Create DataFrame, might have unequal lengths - pad or handle?
                # Easiest is pandas default handling which might create object columns
                return pd.DataFrame.from_dict(preview_dict, orient='index').transpose()

            else:
                print(f"Warning: Unsupported file type for preview: {ext}")
                return None
        except Exception as e:
            print(f"Error previewing dataset {filename}: {str(e)}")
            return None
            
    def check_dataset_quality(self, filename: str) -> bool:
        """
        Checks the quality of a dataset in the processed_data directory.
        Currently checks only for the presence of missing values.

        Args:
            filename: The name of the dataset file in processed_data.

        Returns:
            bool: True if the dataset has no missing values, False otherwise.

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            Exception: If there's an error loading the dataset.
        """
        df = self.load_dataset(filename)
        if df is None:
            # load_dataset should raise FileNotFoundError if applicable
            # This case handles other potential loading issues, though less likely now
            raise Exception(f"Could not load dataset '{filename}' for quality check.")

        # Check for missing values
        no_missing_values = not df.isnull().values.any()

        # Standardization check removed
        # is_standardized = False
        # try:
        #     # Select numeric columns
        #     numeric_cols = df.select_dtypes(include=np.number)
        #     if not numeric_cols.empty:
        #         # Check if mean is close to 0 and std dev is close to 1 for all numeric cols
        #         means_close_to_zero = np.allclose(numeric_cols.mean(), 0, atol=1e-5)
        #         stds_close_to_one = np.allclose(numeric_cols.std(), 1, atol=1e-5)
        #         is_standardized = means_close_to_zero and stds_close_to_one
        #     else:
        #         # If no numeric columns, consider it vacuously standardized?
        #         # Or perhaps return False? Let's say False for now.
        #         is_standardized = False # No numeric data to standardize
        # except Exception as e:
        #     print(f"Warning: Could not perform standardization check on {filename}: {e}")
        #     is_standardized = False # Error during check

        return no_missing_values # Return only the missing value status

    def load_dataset(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Loads the *full* specified dataset from the processed_data directory.
        
        Args:
            filename: The name of the dataset file.
            
        Returns:
            A pandas DataFrame containing the loaded data, or None if an error occurs.
        """
        file_path = self._get_full_path(filename, use_raw=False) # Use processed path
        if not os.path.exists(file_path):
             print(f"Warning: Processed dataset file not found: {file_path}")
             return None

        ext = os.path.splitext(file_path)[1].lower()
        try:
             if ext == '.csv':
                  return pd.read_csv(file_path)
             elif ext == '.mat':
                  mat_data = loadmat(file_path)
                  # Attempt to create a more useful DataFrame from MAT
                  data_dict = {}
                  for k, v in mat_data.items():
                       if k.startswith('__'): continue
                       if isinstance(v, np.ndarray):
                            # Handle multi-dimensional arrays if necessary, flatten for simple case
                            data_dict[k] = v.flatten()
                  # Pad shorter arrays if necessary for DataFrame creation?
                  # This simple version assumes arrays are compatible or will be object type
                  return pd.DataFrame.from_dict(data_dict, orient='index').transpose()
             else:
                  print(f"Warning: Unsupported file type for loading: {ext}")
                  return None
        except Exception as e:
             print(f"Error loading dataset {filename}: {str(e)}")
             return None

    def delete_feature(self, feature_name: str) -> bool:
        """
        Deletes a feature definition and its corresponding CSV file.
        
        Args:
            feature_name: Name of the feature to delete.
            
        Returns:
            True if deletion was successful, False otherwise.
        """
        if feature_name not in self._features:
            return False
            
        try:
            # Delete the feature definition
            del self._features[feature_name]
            self._save_features()

            # Delete the corresponding CSV file
            safe_name = '_'.join(c for c in feature_name if c.isalnum() or c in ('_', '-')).rstrip()
            csv_path = os.path.join(self.features_dir, f"{safe_name}.csv")
            if os.path.exists(csv_path):
                os.remove(csv_path)

            return True
        except Exception as e:
            print(f"Error deleting feature '{feature_name}': {str(e)}")
            return False

    def export_all_features_to_csv(self) -> int:
         """
         Exports all currently defined features to individual CSV files
         in the 'features' directory. Overwrites existing files.
         
         Returns:
             The number of features successfully exported.
         """
         count = 0
         for feature_name in list(self._features.keys()): # Iterate over keys copy
              try:
                   self._save_feature_to_csv(feature_name)
                   count += 1
              except Exception as e:
                   # Error is printed within _save_feature_to_csv
                   pass
         return count

    # --- Cleaning History Methods (Placeholder/Example) ---
    # These would likely operate on files in processed_data

    def add_cleaning_history_entry(self, filename: str, action: str) -> None:
        """
        Add an entry to the cleaning history for a file (conceptual).
        This would track operations performed on files in processed_data.
        """
        processed_filename = filename # Assume filename refers to processed data
        if processed_filename not in self._cleaning_history:
            self._cleaning_history[processed_filename] = []
        self._cleaning_history[processed_filename].append(f"{datetime.now().isoformat()}: {action}")
        # Persist history? Maybe in a separate JSON or within metadata?

    def get_cleaning_history(self, filename: str) -> List[str]:
        """Get cleaning history for a processed file (conceptual)."""
        return self._cleaning_history.get(filename, [])

    def reset_cleaning_history(self, filename: str) -> None:
         """Resets cleaning history and potentially reverts the processed file (conceptual)."""
         if filename in self._cleaning_history:
              del self._cleaning_history[filename]
         # Re-copy from raw_data to processed_data to reset
         raw_path = self._get_full_path(filename, use_raw=True)
         processed_path = self._get_full_path(filename, use_raw=False)
         if os.path.exists(raw_path):
              try:
                   shutil.copy2(raw_path, processed_path)
                   print(f"Reset processed file: {filename}")
              except Exception as e:
                   print(f"Error resetting processed file {filename}: {e}")
         else:
              print(f"Warning: Raw file {filename} not found for reset.")

    def save_processed_dataset(self, df: pd.DataFrame, filename: str) -> None:
        """
        Saves a DataFrame to a specified file within the processed_data directory.
        Overwrites the file if it already exists.

        Args:
            df: The pandas DataFrame to save.
            filename: The name of the file (including extension) to save the DataFrame as.

        Raises:
            IOError: If saving fails.
            TypeError: If df is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        file_path = self._get_full_path(filename, use_raw=False) # Save to processed path
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == '.csv':
                df.to_csv(file_path, index=False)
            # Add elif for .mat if needed (requires scipy.io.savemat)
            # elif ext == '.mat':
            #     from scipy.io import savemat
            #     # Convert DataFrame to dict suitable for savemat
            #     mat_dict = {col_name: df[col_name].values for col_name in df.columns}
            #     savemat(file_path, mat_dict)
            else:
                raise ValueError(f"Unsupported file type for saving: {ext}. Only .csv is currently supported.")

            print(f"Successfully saved processed dataset to: {file_path}")

        except Exception as e:
            raise IOError(f"Error saving processed dataset to {file_path}: {e}") from e 

    def delete_processed_dataset(self, filename: str) -> bool:
        """
        Deletes a dataset file from the processed_data directory.

        Args:
            filename: The name of the dataset file to delete.

        Returns:
            True if deletion was successful, False otherwise.

        Raises:
            FileNotFoundError: If the file doesn't exist in processed_data.
            IOError: If there's a permission error or other OS issue during deletion.
        """
        file_path = self._get_full_path(filename, use_raw=False) # Get path in processed_data

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found in processed_data: {filename}")

        try:
            os.remove(file_path)
            print(f"Successfully deleted processed dataset: {file_path}")
            # Remove from cleaning history if present
            if filename in self._cleaning_history:
                 del self._cleaning_history[filename]
            return True
        except PermissionError as e:
            raise IOError(f"Permission denied while trying to delete {file_path}: {e}") from e
        except Exception as e:
            raise IOError(f"Error deleting processed dataset {file_path}: {e}") from e 

    def restore_raw_data_to_processed(self) -> Tuple[int, int, List[str]]:
        """
        Restores the processed_data directory to match the raw_data directory.
        First, deletes all files in processed_data.
        Then, copies all files from raw_data to processed_data.

        Returns:
            A tuple containing:
                - int: Number of files successfully deleted from processed_data.
                - int: Number of files successfully copied from raw_data.
                - List[str]: A list of error messages encountered during the process.
        """
        deleted_count = 0
        copied_count = 0
        errors = []

        # Step 1: Delete all files in processed_data
        if os.path.exists(self.processed_data_path):
            for filename in os.listdir(self.processed_data_path):
                file_path = os.path.join(self.processed_data_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path): # Check if it's a file or a symlink
                        os.unlink(file_path) # Use unlink to remove files and symlinks
                        deleted_count += 1
                    elif os.path.isdir(file_path):
                        # Optionally handle subdirectories, for now, we'll skip them
                        # or raise an error if unexpected subdirectories are found.
                        # errors.append(f"Skipped unexpected directory in processed_data: {filename}")
                        print(f"Warning: Skipped unexpected directory in processed_data: {filename}")
                except Exception as e:
                    error_msg = f"Error deleting file {filename} from processed_data: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
        
        # Step 2: Copy all files from raw_data to processed_data
        if os.path.exists(self.raw_data_path):
            for filename in os.listdir(self.raw_data_path):
                source_path = os.path.join(self.raw_data_path, filename)
                dest_path = os.path.join(self.processed_data_path, filename)
                try:
                    if os.path.isfile(source_path):
                        # Ensure processed_data_path exists (it should, but defensive check)
                        os.makedirs(self.processed_data_path, exist_ok=True)
                        shutil.copy2(source_path, dest_path)
                        copied_count += 1
                except Exception as e:
                    error_msg = f"Error copying file {filename} from raw_data to processed_data: {str(e)}"
                    print(error_msg)
                    errors.append(error_msg)
        
        # Clear any existing cleaning history as processed data is reset
        self._cleaning_history.clear()
        
        # Optionally, re-validate features if any exist, as their source datasets might change
        # or remove features whose source datasets no longer exist in processed_data.
        # For now, let's just clear and re-save existing features to ensure consistency.
        # This will attempt to save their CSVs based on the new processed_data.
        # If a feature's source dataset is no longer in processed_data, _save_feature_to_csv will warn.
        if self._features:
            print("Updating feature CSVs after restoring raw data...")
            self.export_all_features_to_csv() # This will try to resave all feature CSVs
                                            # based on current definitions and new processed_data


        return deleted_count, copied_count, errors 