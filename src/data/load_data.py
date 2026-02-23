"""
Data loading utilities for the Sexism Classification project.

This module provides functions to load raw and processed datasets
with proper error handling and validation.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths (adjust as needed)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"


class DataLoader:
    """Utility class for loading datasets."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory. Defaults to project data/ dir.
        """
        self.data_dir = data_dir or DATA_DIR
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Create directories if they don't exist
        for directory in [self.raw_dir, self.processed_dir, self.external_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_raw_dataset(
        self,
        filename: str,
        **read_csv_kwargs
    ) -> pd.DataFrame:
        """
        Load raw dataset from CSV.
        
        Args:
            filename: Name of CSV file in raw/ directory
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with raw data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or invalid
        """
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found: {filepath}\n"
                f"Please place your dataset in {self.raw_dir}/"
            )
        
        logger.info(f"Loading raw dataset from {filepath}")
        
        try:
            df = pd.read_csv(filepath, **read_csv_kwargs)
            
            if df.empty:
                raise ValueError(f"Dataset is empty: {filepath}")
            
            logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_processed_dataset(
        self,
        filename: str,
        **read_csv_kwargs
    ) -> pd.DataFrame:
        """
        Load processed dataset from CSV.
        
        Args:
            filename: Name of CSV file in processed/ directory
            **read_csv_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with processed data
        """
        filepath = self.processed_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {filepath}\n"
                f"Have you run preprocessing yet?"
            )
        
        logger.info(f"Loading processed dataset from {filepath}")
        df = pd.read_csv(filepath, **read_csv_kwargs)
        logger.info(f"Loaded {len(df):,} rows")
        return df
    
    def save_processed_dataset(
        self,
        df: pd.DataFrame,
        filename: str,
        **to_csv_kwargs
    ) -> Path:
        """
        Save processed dataset to CSV.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            **to_csv_kwargs: Additional arguments for df.to_csv
            
        Returns:
            Path to saved file
        """
        filepath = self.processed_dir / filename
        
        logger.info(f"Saving processed dataset to {filepath}")
        df.to_csv(filepath, index=False, **to_csv_kwargs)
        logger.info(f"Saved {len(df):,} rows to {filepath}")
        
        return filepath
    
    def load_train_test_split(
        self,
        train_filename: str = "train.csv",
        test_filename: str = "test.csv",
        val_filename: Optional[str] = "val.csv"
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Load train/test/validation splits.
        
        Args:
            train_filename: Training set filename
            test_filename: Test set filename
            val_filename: Validation set filename (optional)
            
        Returns:
            Tuple of (train_df, test_df) or (train_df, val_df, test_df)
        """
        train_df = self.load_processed_dataset(train_filename)
        test_df = self.load_processed_dataset(test_filename)
        
        if val_filename:
            try:
                val_df = self.load_processed_dataset(val_filename)
                logger.info(
                    f"Loaded splits - Train: {len(train_df):,}, "
                    f"Val: {len(val_df):,}, Test: {len(test_df):,}"
                )
                return train_df, val_df, test_df
            except FileNotFoundError:
                logger.warning(f"Validation file {val_filename} not found, skipping")
        
        logger.info(f"Loaded splits - Train: {len(train_df):,}, Test: {len(test_df):,}")
        return train_df, test_df
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        stratify_column: Optional[str] = None,
        random_state: int = 42,
        save: bool = True
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Create and optionally save train/test/validation splits.
        
        Args:
            df: Full dataset
            test_size: Proportion for test set (0.0-1.0)
            val_size: Proportion for validation set (optional)
            stratify_column: Column name for stratified split
            random_state: Random seed for reproducibility
            save: Whether to save splits to processed/ directory
            
        Returns:
            Tuple of DataFrames: (train, test) or (train, val, test)
        """
        logger.info(f"Creating train/test split (test_size={test_size})")
        
        stratify = df[stratify_column] if stratify_column else None
        
        if val_size:
            # Create train/val/test split
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            # Further split train into train/val
            val_proportion = val_size / (1 - test_size)
            stratify_train = train_df[stratify_column] if stratify_column else None
            
            train_df, val_df = train_test_split(
                train_df,
                test_size=val_proportion,
                random_state=random_state,
                stratify=stratify_train
            )
            
            logger.info(
                f"Split sizes - Train: {len(train_df):,}, "
                f"Val: {len(val_df):,}, Test: {len(test_df):,}"
            )
            
            if save:
                self.save_processed_dataset(train_df, "train.csv")
                self.save_processed_dataset(val_df, "val.csv")
                self.save_processed_dataset(test_df, "test.csv")
            
            return train_df, val_df, test_df
        else:
            # Create train/test split only
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )
            
            logger.info(f"Split sizes - Train: {len(train_df):,}, Test: {len(test_df):,}")
            
            if save:
                self.save_processed_dataset(train_df, "train.csv")
                self.save_processed_dataset(test_df, "test.csv")
            
            return train_df, test_df
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """
        Get summary information about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add label distribution if there's a common label column
        for label_col in ["label", "target", "class", "y"]:
            if label_col in df.columns:
                info["label_distribution"] = df[label_col].value_counts().to_dict()
                break
        
        return info
    

# Convenience functions
def load_raw_data(filename: str, **kwargs) -> pd.DataFrame:
    """Load raw dataset (convenience function)."""
    loader = DataLoader()
    return loader.load_raw_dataset(filename, **kwargs)


def load_processed_data(filename: str, **kwargs) -> pd.DataFrame:
    """Load processed dataset (convenience function)."""
    loader = DataLoader()
    return loader.load_processed_dataset(filename, **kwargs)


def load_splits(**kwargs):
    """Load train/test splits (convenience function)."""
    loader = DataLoader()
    return loader.load_train_test_split(**kwargs)


# Example usage
if __name__ == "__main__":
    # Example: Load and split a dataset
    loader = DataLoader()
    
    try:
        # Load raw data
        train_df = loader.load_raw_dataset("train.csv")
        dev_df = loader.load_raw_dataset("dev.csv")
        test_df = loader.load_raw_dataset("test.csv")

        
        # Print info
        train_info = loader.get_dataset_info(train_df)
        dev_info   = loader.get_dataset_info(dev_df)
        test_info  = loader.get_dataset_info(test_df)

        print("\nTrain Dataset Info:")
        for key, value in train_info.items():
            print(f"  {key}: {value}")

        print("\nDev Dataset Info:")
        for key, value in dev_info.items():
            print(f"  {key}: {value}")

        print("\nTest Dataset Info:")
        for key, value in test_info.items():
            print(f"  {key}: {value}")
                
        print("\nDatasets loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nTo use this script:")
        print("1. Place your dataset in data/raw/")
        print("2. Update the filename and column names")
        print("3. Run this script again")