import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import nltk
import json
import logging
from src.data.consts import VOCAB_PATH, PREPROCESSED_DATA

def initialize_nltk():
    """
    Initialize NLTK by downloading necessary resources.
    """
    nltk.download('punkt')


def find_replacements(df):
    """
    Find word replacements based on sentiment analysis.

    Args:
        df (pd.DataFrame): DataFrame containing reference and translation sentences.

    Returns:
        dict: Dictionary of word replacements.
    """
    sia = SentimentIntensityAnalyzer()
    replacements = {}

    for index, row in df.iterrows():
        reference_sentence = row['reference']
        translation_sentence = row['translation']

        reference_tokens = word_tokenize(reference_sentence)
        translation_tokens = word_tokenize(translation_sentence)

        for ref_token, trn_token in zip(reference_tokens, translation_tokens):
            ref_sentiment = sia.polarity_scores(ref_token)['compound']
            if ref_sentiment < 0:
                replacements[ref_token] = trn_token

    return replacements


def save_replacements_to_json(replacements, output_file):
    """
    Save word replacements to a JSON file.

    Args:
        replacements (dict): Dictionary of word replacements.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, "w") as json_file:
        json.dump(replacements, json_file)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    initialize_nltk()
    logging.info(f"Reading data...")
    df = pd.read_csv(PREPROCESSED_DATA)
    logging.info(f"Finding replacements...")
    replacements = find_replacements(df)
    logging.info(f"Saving replacements to the vocab...")
    save_replacements_to_json(replacements, VOCAB_PATH)
    logging.info(f"Replacements vocab saved to vocab.json")


if __name__ == "__main__":
    main()
