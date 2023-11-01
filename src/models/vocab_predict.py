import json
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from src.data.consts import VOCAB_PATH, TEST_DATA_PATH, TOX_LEVEL_MODEL_PATH, VOCAB_PREDICT_PATH

def initialize_logger():
    """
    Initialize the logging configuration.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_vocab(filename):
    """
    Load vocabulary from a JSON file.

    Args:
        filename (str): Path to the JSON file.

    Returns:
        dict: Loaded vocabulary as a dictionary.
    """
    with open(filename, "r") as json_file:
        vocab = json.load(json_file)
    return vocab

def load_test_data(test_data_path):
    """
     Load test data from a CSV file into a DataFrame.

     Args:
        test_data_path (str): Path to the CSV file.

     Returns:
        pd.DataFrame: DataFrame containing the test data.
    """
    return pd.read_csv(test_data_path)

def process_data(test_df, vocab_dict):
    """
    Process test data by replacing words using a vocabulary dictionary.

    Args:
        test_df (pd.DataFrame): DataFrame containing the test data.
        vocab_dict (dict): Vocabulary dictionary for word replacements.

    Returns:
        pd.DataFrame: Processed DataFrame with replaced words.
    """
    for original, item in vocab_dict.items():
        test_df['translate'] = test_df['translate'].str.replace(original, item)
    return test_df

def predict_toxicity(test_df, model_path):
    """
    Predict toxicity using a machine learning model.

    Args:
        test_df (pd.DataFrame): DataFrame with test data.
        model_path (str): Path to the machine learning model.

    Returns:
        pd.DataFrame: DataFrame with toxicity predictions.
    """
    model = load_model(model_path)
    X_test = test_df['translate']
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_test_tfidf = tfidf_vectorizer.fit_transform(X_test)
    y_pred = model.predict(X_test_tfidf.toarray())
    test_df['tr_tox'] = y_pred
    return test_df

def calculate_differences(test_df):
    """
    Calculate differences in toxicity levels between reference and translated texts.

    Args:
        test_df (pd.DataFrame): DataFrame with toxicity data.

    Returns:
        pd.DataFrame: DataFrame with toxicity differences.
    """
    test_df['result'] = test_df['ref_tox'] - test_df['tr_tox']
    return test_df

def save_results(test_df, output_path):
    """
    Save the results to a CSV file.

    Args:
        test_df (pd.DataFrame): DataFrame with results.
        output_path (str): Path to the output CSV file.
    """
    test_df.to_csv(output_path)
    logging.info(f"Results saved to {output_path}")

def main():

    initialize_logger()
    logging.info("Data loading....")
    vocab_dict = load_vocab(VOCAB_PATH)
    test_df = load_test_data(TEST_DATA_PATH)
    logging.info("Data translating....")
    test_df = process_data(test_df, vocab_dict)
    logging.info("Toxicity predicting...")
    test_df = predict_toxicity(test_df, TOX_LEVEL_MODEL_PATH)
    logging.info("Differences calculating...")
    test_df = calculate_differences(test_df)
    logging.info("Results saving...")
    save_results(test_df, VOCAB_PATH)
    logging.info("Results saved to predictVOCAB.csv")

if __name__ == "__main__":
    main()