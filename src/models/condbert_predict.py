import torch
import numpy as np
import pickle
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from models.condbert import CondBertRewriter
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import logging
from src.data.consts import VOCAB_ROOT, TEST_DATA_PATH, TOX_LEVEL_MODEL_PATH, CONDBERT_PREDICT_PATH

def initialize_condbert():
    """
    Initialize CondBERT model and its components for text translation.
    Returns:
        CondBertRewriter: Initialized CondBERT rewriter.
    """
    device = torch.device('cpu')
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)

    vocab_root = '../data/interm/vocab/'

    with open(VOCAB_ROOT + "negative-words.txt", "r") as f:
        s = f.readlines()
    negative_words = list(map(lambda x: x[:-1], s))

    with open(VOCAB_ROOT + "positive-words.txt", "r") as f:
        s = f.readlines()
    positive_words = list(map(lambda x: x[:-1], s))

    with open(VOCAB_ROOT + 'tox_coef.pkl', 'rb') as f:
        tox_coef = pickle.load(f)

    token_tox = []
    with open(vocab_root + 'token_tox.txt', 'r') as f:
        for line in f.readlines():
            token_tox.append(float(line))
    token_tox = np.array(token_tox)
    token_tox = np.maximum(0, np.log(1/(1/token_tox-1)))  # log odds ratio

    for tok in ['.', ',', '-']:
        token_tox[tokenizer.encode(tok)][1] = 3

    for tok in ['you']:
        token_tox[tokenizer.encode(tok)][1] = 0

    editor = CondBertRewriter(
        model=model,
        tokenizer=tokenizer,
        device=device,
        neg_words=negative_words,
        pos_words=positive_words,
        tox_coef=tox_coef,
        token_tox=token_tox,
    )

    return editor

def translate_text(editor, text):
    """
    Translate the given text using CondBERT rewriter.

    Args:
        editor (CondBertRewriter): CondBERT rewriter instance.
        text (str): Input text to be translated.

    Returns:
        str: Translated text.
    """
    return editor.translate(text, prnt=False)

def load_and_predict_data(editor):
    """
    Load test data, translate it, and make toxicity predictions.

    Args:
        editor (CondBertRewriter): CondBERT rewriter instance.

    Returns:
        pd.DataFrame: DataFrame with translated and predicted data.
    """
    logging.info("Loading and preprocessing data...")
    filtered_df = pd.read_csv(TEST_DATA_PATH, index_col=None)
    logging.info("Making predictions...")
    filtered_df['translate'] = filtered_df['reference'].apply(translate_text, args=(editor,))
    return filtered_df

def predict_toxicity(dataframe):
    """
    Predict the toxicity of the translated text using a machine learning model.

    Args:
        dataframe (pd.DataFrame): DataFrame with translated data.

    Returns:
        pd.DataFrame: DataFrame with toxicity predictions.
    """

    logging.info("Predicting toxicity of translation...")
    model = load_model(TOX_LEVEL_MODEL_PATH)
    X_test = dataframe['translate']
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_test_tfidf = tfidf_vectorizer.fit_transform(X_test)
    y_pred = model.predict(X_test_tfidf.toarray())
    dataframe['tr_tox'] = y_pred
    return dataframe

def calculate_differences(dataframe):
    """
    Calculate the differences in toxicity levels between reference and translated texts.

    Args:
        dataframe (pd.DataFrame): DataFrame with toxicity data.

    Returns:
        pd.DataFrame: DataFrame with toxicity differences.
    """
    logging.info("Calculating differences...")
    dataframe['result'] = dataframe['ref_tox'] - dataframe['tr_tox']
    return dataframe

def save_results(dataframe, filename):
    """
    Save the results to a CSV file.

    Args:
        dataframe (pd.DataFrame): DataFrame with results.
        filename (str): Path to the output CSV file.
    """
    logging.info(f"Saving results to {filename}...")
    dataframe.to_csv(filename, index=False)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("CondBERT prediction started...")
    editor = initialize_condbert()
    data = load_and_predict_data(editor)
    data_with_predictions = predict_toxicity(data)
    data_with_differences = calculate_differences(data_with_predictions)
    save_results(data_with_differences, CONDBERT_PREDICT_PATH)
    logging.info("CondBERT prediction finished.")

if __name__ == "__main__":
    main()
