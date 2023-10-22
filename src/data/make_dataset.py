import logging
import pandas as pd
import zipfile
from consts import FILE_PATH, PREPROCESSED_DATA, TEST_DATA_PATH

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
        pd.DataFrame: The DataFrame containing the testing data.
    """

    # Filter rows based on ref_tox - trn_tox >= 0.5 and similarity > 0.7
    filtered_df = dataset_df[(dataset_df['ref_tox'] - dataset_df['trn_tox'] >= 0.5) & (dataset_df['similarity'] > 0.7)]

    # Create another DataFrame with the remaining rows (up to 5000 rows)
    test_df = dataset_df[~dataset_df.index.isin(filtered_df.index)& (dataset_df['ref_tox'] > 0.7)].head(5000)
    test_df = test_df[["reference", "ref_tox"]]

    return filtered_df, test_df


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
        dataset_df, test_df = preprocess_data(dataset_df)
        logger.info("Data preprocessing complete.")

        # Save preprocessed data
        logger.info("Saving the data...")
        dataset_df.to_csv(PREPROCESSED_DATA, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)
        logger.info("Data saved in data/interm.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
