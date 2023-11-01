import os

import nltk
import numpy as np
from nltk.tbl import feature
from torch.distributed.pipeline.sync import pipe
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from collections import Counter
from collections import defaultdict
import logging
from src.data.consts import VOCAB_ROOT, TOX_PATH, NORM_PATH


class NgramSalienceCalculator():
    """
        Calculate salience scores for n-grams based on their frequency in toxic and normal corpora.

        Args:
            tox_corpus (list): List of toxic sentences.
            norm_corpus (list): List of normal sentences.
            use_ngrams (bool, optional): If True, use n-grams; otherwise, use unigrams. Default is False.

        Attributes:
            vectorizer (CountVectorizer): A CountVectorizer for n-grams.
            tox_vocab (dict): Vocabulary for toxic n-grams.
            tox_counts (numpy.ndarray): Total counts of toxic n-grams.
            norm_vocab (dict): Vocabulary for normal n-grams.
            norm_counts (numpy.ndarray): Total counts of normal n-grams.

        Methods:
            salience(self, feature, attribute='tox', lmbda=0.5):
                Calculate the salience score of a feature (n-gram or unigram) based on its attribute (toxic or normal).
                Args:
                    feature (str): The feature (n-gram or unigram) for which to calculate salience.
                    attribute (str, optional): The attribute to consider ('tox' for toxic, 'norm' for normal). Default is 'tox'.
                    lmbda (float, optional): Smoothing parameter to avoid division by zero. Default is 0.5.
    """
    def __init__(self, tox_corpus, norm_corpus, use_ngrams=False):
        ngrams = (1, 3) if use_ngrams else (1, 1)
        self.vectorizer = CountVectorizer(ngram_range=ngrams)

        tox_count_matrix = self.vectorizer.fit_transform(tox_corpus)
        self.tox_vocab = self.vectorizer.vocabulary_
        self.tox_counts = np.sum(tox_count_matrix, axis=0)

        norm_count_matrix = self.vectorizer.fit_transform(norm_corpus)
        self.norm_vocab = self.vectorizer.vocabulary_
        self.norm_counts = np.sum(norm_count_matrix, axis=0)
        self.salience(self, feature,'tox', 0.5)

def initialize_nltk():
    """
    Initialize NLTK by downloading necessary resources.
    """
    nltk.download('punkt')

def prepare_drg_vocabulary(tox_corpus_path, norm_corpus_path, VOCAB_DIRNAME, threshold):
    """
    Prepare DRG-like vocabularies using toxic and normal corpora.

    Args:
        tox_corpus_path (str): Path to the toxic corpus.
        norm_corpus_path (str): Path to the normal corpus.
        VOCAB_DIRNAME (str): Path to the vocabulary directory.
        threshold (int): Threshold for selecting words.

    Returns:
        corpus_tox (list): List of toxic sentences.
        corpus_norm (list): List of normal sentences.
    """
    if not os.path.exists(VOCAB_DIRNAME):
        os.makedirs(VOCAB_DIRNAME)

    c = Counter()

    for fn in [tox_corpus_path, norm_corpus_path]:
        with open(fn, 'r') as corpus:
            for line in corpus.readlines():
                for tok in line.strip().split():
                    c[tok] += 1

    vocab = {w for w, _ in c.most_common() if _ > 0}

    with open(tox_corpus_path, 'r') as tox_corpus, open(norm_corpus_path, 'r') as norm_corpus:
        corpus_tox = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in tox_corpus.readlines()]
        corpus_norm = [' '.join([w if w in vocab else '<unk>' for w in line.strip().split()]) for line in norm_corpus.readlines()]

    neg_out_name = os.path.join(VOCAB_DIRNAME, 'negative-words.txt')
    pos_out_name = os.path.join(VOCAB_DIRNAME, 'positive-words.txt')

    sc = NgramSalienceCalculator(corpus_tox, corpus_norm, False)
    seen_grams = set()

    with open(neg_out_name, 'w') as neg_out, open(pos_out_name, 'w') as pos_out:
        for gram in set(sc.tox_vocab.keys()).union(set(sc.norm_vocab.keys())):
            if gram not in seen_grams:
                seen_grams.add(gram)
                toxic_salience = sc.salience(gram, attribute='tox')
                polite_salience = sc.salience(gram, attribute='norm')
                if toxic_salience > threshold:
                    neg_out.writelines(f'{gram}\n')
                elif polite_salience > threshold:
                    pos_out.writelines(f'{gram}\n')

    return corpus_tox, corpus_norm

def evaluate_word_toxicities(corpus_tox, corpus_norm, VOCAB_DIRNAME):
    """
    Evaluate word toxicities with a logistic regression and save the results to a file.

    Args:
        corpus_tox (list): List of toxic sentences.
        corpus_norm (list): List of normal sentences.
        VOCAB_DIRNAME (str): Path to the vocabulary directory.

    Returns:
        None
    """
    X_train = corpus_tox + corpus_norm
    y_train = [1] * len(corpus_tox) + [0] * len(corpus_norm)
    pipe.fit(X_train, y_train)

    coefs = pipe[1].coef_[0]
    word2coef = {w: coefs[idx] for w, idx in pipe[0].vocabulary_.items()}

    with open(os.path.join(VOCAB_DIRNAME, 'word2coef.pkl'), 'wb') as f:
        pickle.dump(word2coef, f)

def label_bert_tokens(corpus_tox, corpus_norm, tokenizer, VOCAB_DIRNAME):
    """
    Label BERT tokens by toxicity and save the results to a file.

    Args:
        corpus_tox (list): List of toxic sentences.
        corpus_norm (list): List of normal sentences.
        tokenizer (BertTokenizer): BERT tokenizer.
        VOCAB_DIRNAME (str): Path to the vocabulary directory.

    Returns:
        None
    """
    toxic_counter = defaultdict(lambda: 1)
    nontoxic_counter = defaultdict(lambda: 1)

    for text in tqdm(corpus_tox):
        for token in tokenizer.encode(text):
            toxic_counter[token] += 1
    for text in tqdm(corpus_norm):
        for token in tokenizer.encode(text):
            nontoxic_counter[token] += 1

    token_toxicities = [toxic_counter[i] / (nontoxic_counter[i] + toxic_counter[i]) for i in range(len(tokenizer.vocab))]

    with open(os.path.join(VOCAB_DIRNAME, 'token_toxicities.txt'), 'w') as f:
        for t in token_toxicities:
            f.write(str(t))
            f.write('\n')

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    threshold = 4

    initialize_nltk()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logging.info(f"Creating lists of words...")
    corpus_tox, corpus_norm = prepare_drg_vocabulary(TOX_PATH, NORM_PATH, VOCAB_ROOT, threshold)
    logging.info(f"Evaluating toxicity...")
    evaluate_word_toxicities(corpus_tox, corpus_norm, VOCAB_ROOT)
    logging.info(f"Saving tokens...")
    label_bert_tokens(corpus_tox, corpus_norm, tokenizer, VOCAB_ROOT)
    logging.info(f"All necessary vocab files is created.")

if __name__ == "__main__":
    main()
