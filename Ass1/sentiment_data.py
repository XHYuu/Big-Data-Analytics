import os
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


np.random.seed(42)


###############################
# Please complete this file
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
        return f"Phase id: {self.phrase_id}, Phase: {repr(self.words)}; label: {repr(self.label)}"

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
            # For train data
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

    words = []
    
    # ------------------
    # Write your code here



    # ------------------

    return words


def test_text_preprocessing():
    # test code for text preprocessing
    train_exs = load_sentiment_examples("data/train.csv", "train")
    from collections import Counter
    counter = Counter()
    for ex in train_exs:
        for word in ex.words:
            counter[word] += 1

    print(counter.most_common(10))


if __name__ == "__main__":
    test_text_preprocessing()
