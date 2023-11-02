import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging
from src.data.consts import VOCAB_PREDICT_PATH, CONDBERT_PREDICT_PATH, RESULT_PATH

def create_directory(directory):
    """
    Create a directory if it doesn't exist.

    Args:
        directory (str): Path to the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

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

def visualize_toxicity_difference():
    """
    Visualize the difference in toxicity levels between translation models and save the plot as an image.

    This function reads prediction data from two CSV files, combines and sorts them, and then plots the difference in
    toxicity levels between translations. It saves the resulting plot as an image.

    """

    combined_df = pd.read_csv(RESULT_PATH, index_col=None)

    create_directory("reports/figures")

    result = combined_df['result'].tolist()
    n_bins = 120
    plt.figure(figsize=(6, 4))
    plt.ylabel('number of examples')
    plt.xlabel('difference between tox levels')
    plt.hist(result, bins=n_bins)
    plt.title('Tox Level Difference')
    plt.savefig("reports/figures/final_toxicity.png")
    plt.close()

def calculate_average_toxicity():
    """
    Calculate the average toxicity levels of translation models and the reference.

    This function reads prediction data from two CSV files, combines and sorts them, and then calculates the average
    toxicity levels of the translation models and the reference.

    Returns:
        tuple: A tuple containing the average toxicity levels for Result model, Vocab model, BERT model, and the reference.
    """
    condbert_pred = pd.read_csv(CONDBERT_PREDICT_PATH, index_col=None)
    vocab_pred = pd.read_csv(VOCAB_PREDICT_PATH, index_col=None)
    combined_df = pd.read_csv(RESULT_PATH, index_col=None)

    average_value = combined_df['tr_tox'].mean()
    av_bert = condbert_pred['tr_tox'].mean()
    av_vocab = vocab_pred['tr_tox'].mean()
    av_ref = vocab_pred['ref_tox'].mean()

    return average_value, av_vocab, av_bert, av_ref

def visualize_toxicity_comparison(average_values):
    """
    Visualize and compare the average toxicity levels of translation models and the reference.

    Args:
        average_values (tuple): A tuple containing the average toxicity levels for Result model, Vocab model, BERT model, and the reference.

    This function takes the average toxicity levels and visualizes the comparison of translation toxicity by creating a
    bar chart and saving it as an image.

    """
    labels = ['tox_result', 'tox_vocab', 'tox_bert', "ref_tox"]
    values = average_values

    plt.bar(labels, values)
    plt.title('Comparison of translation toxicity')
    plt.ylabel('Toxicity coefficient')
    plt.savefig("reports/figures/compare_toxicity.png")
    plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Creating visualization started...")
    visualize_toxicity_difference()
    average_values = calculate_average_toxicity()
    visualize_toxicity_comparison(average_values)
    logging.info("Visualization figures saved to reports/figures")
