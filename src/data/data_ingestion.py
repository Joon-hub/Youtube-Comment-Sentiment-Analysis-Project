import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from typing import Tuple

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f'Parameters retrieved from {params_path}')
        return params
    except FileNotFoundError:
        logger.error(f'File not found: {params_path}')
        raise
    except yaml.YAMLError as e:
        logger.error(f'YAML error: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug(f'Data loaded from {data_url}')
        logger.debug(f'Data shape: {df.shape}')
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the CSV file: {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error occurred while loading the data: {e}')
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if 'clean_comment' not in df.columns or 'category' not in df.columns:
        raise KeyError("Required columns missing from DataFrame")

    try:
        initial_shape = df.shape
        logger.debug(f'Initial data shape: {initial_shape}')

        # Drop missing values
        df.dropna(inplace=True)
        logger.debug(f'After dropping NA - shape: {df.shape}')

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        logger.debug(f'After dropping duplicates - shape: {df.shape}')

        # Remove empty comments
        df = df[~(df['clean_comment'].str.strip() == '')]
        logger.debug(f'After removing empty comments - shape: {df.shape}')

        # Encode target variable
        df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
        logger.debug(f'After encoding target variable - shape: {df.shape}')

        # Remove rows with NaN categories
        df = df.dropna(subset=['category'])
        logger.debug(f'Final data shape: {df.shape}')

        return df
    except Exception as e:
        logger.error(f'Error during preprocessing: {e}')
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        
        train_path = os.path.join(raw_data_path, "train.csv")
        test_path = os.path.join(raw_data_path, "test.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.debug(f'Data saved successfully:\nTrain: {train_path}\nTest: {test_path}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise

def split_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets."""
    try:
        train_data, test_data = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42, 
            stratify=df['category']
        )
        logger.debug(f'Data split - Train: {train_data.shape}, Test: {test_data.shape}')
        return train_data, test_data
    except ValueError as e:
        logger.error(f'Error in train-test split: {e}')
        raise

def main():
    try:
        # Load parameters
        params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
        params = load_params(params_path)
        test_size = params['data_ingestion']['test_size']
        
        # Load and process data
        data_url = 'https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        df = load_data(data_url)
        final_df = preprocess_data(df)
        
        # Split and save data
        train_data, test_data = split_data(final_df, test_size)
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
        save_data(train_data, test_data, data_path)
        
        logger.info('Data ingestion completed successfully')
        
    except Exception as e:
        logger.error(f'Data ingestion failed: {e}')
        raise

if __name__ == '__main__':
    main()