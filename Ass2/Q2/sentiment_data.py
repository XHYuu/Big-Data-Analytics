import os
import pickle
import re
import warnings
from typing import List

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


np.random.seed(42)


###############################
# Please do no change this file
###############################


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[string]): list of words
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, phrase_id, label=None):
        self.words = words
        self.phrase_id = phrase_id
        self.label = label

    def __repr__(self):
        return f"Phase id: {self.phrase_id}, words: {repr(self.words)}; label: {repr(self.label)}"

    def __str__(self):
        return self.__repr__()


def load_sentiment_examples(file_path: str, split: str) -> List[SentimentExample]:
    ''' Read sentiment examples from raw file. Tokenizes and cleans the sentences.'''

    if os.path.exists(f"data/{split}_exs.pkl"):
        with open(f"data/{split}_exs.pkl", "rb") as f:
            exs = pickle.load(f)
    else:
        data = pd.read_csv(file_path)
        exs = []

        if split == "train" or split == "valid":
            # For train or valid data
            for row in tqdm(data.itertuples(), total=len(data), desc=f"Load {split} data"):
                label = getattr(row, "label")
                phrase_id = getattr(row, "Index")
                phase = getattr(row, "text")

                # preprocessing
                word_list = text_preprocessing(phase)
                if len(word_list) > 0:
                    exs.append(SentimentExample(
                        word_list, phrase_id, label))
        elif split == "test":
            # For test data
            for row in tqdm(data.itertuples(), total=len(data), desc=f"Load {split} data"):
                phrase_id = getattr(row, "Index")
                phase = getattr(row, "text")
                # preprocessing
                word_list = text_preprocessing(phase)
                if len(word_list) > 0:
                    exs.append(SentimentExample(word_list, phrase_id))

        with open(f"data/{split}_exs.pkl", "wb") as f:
            pickle.dump(exs, f)

    return exs


def text_preprocessing(sentence: str) -> List[str]:
    '''Preprocess text'''

    # Gets text without tags or markup, remove html
    cur_sentence = BeautifulSoup(sentence, "lxml").get_text()
    # Obtain only letters
    cur_sentence = re.sub("[^a-zA-Z]", " ", cur_sentence)
    # Lower case, tokenization
    words = word_tokenize(cur_sentence.lower())

    preprocessed_words = []
    lemmatizer = WordNetLemmatizer()
    for word in words:
        # Lemmatizing
        lemma_word = lemmatizer.lemmatize(word)
        # Remove stop words
        if lemma_word not in stopwords.words("english"):
            preprocessed_words.append(lemma_word)

    return preprocessed_words
