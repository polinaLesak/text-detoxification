import logging
import argparse
import pandas as pd
import io
from zipfile import ZipFile
import os
import zipfile
from consts import FILE_PATH, PREPROCESSED_DATA

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def extract_and_load_data():
    """
    Extract and load data from a ZIP file in TSV format directly into a DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    zip_file_path = FILE_PATH  # Импортируем путь из constants.py
    with zipfile.ZipFile(FILE_PATH, "r") as zip_ref:
        with zip_ref.open("filtered.tsv") as file:
            return  pd.read_csv(file, sep='\t')

def preprocess_data(dataset_df):
    """
        Preprocess the dataset by filtering rows based on specific conditions.

        Args:
            dataset_df (pd.DataFrame): The input DataFrame containing the dataset.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with filtered rows.
        """
    dataset_df = dataset_df[(dataset_df['ref_tox'] - dataset_df['trn_tox'] > 0.2) & (dataset_df['similarity'] > 0.8)]
    return dataset_df


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Extract and load data directly into a DataFrame
        logger.info("Extracting and loading data from ZIP...")
        dataset_df = extract_and_load_data()
        logger.info("Data loaded into DataFrame successfully.")

        # Preprocess the data
        logger.info("Preprocessing the data...")
        dataset_df = preprocess_data(dataset_df)
        logger.info("Data preprocessing complete.")

        # Save preprocessed data
        logger.info("Saving the data...")
        dataset_df.to_csv(PREPROCESSED_DATA, index=False)  # Здесь данные сохраняются в формате CSV
        logger.info("Data saved in data_interm.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
