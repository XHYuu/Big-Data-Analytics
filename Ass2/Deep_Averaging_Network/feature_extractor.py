from abc import abstractmethod
from typing import List

import numpy as np

from sentiment_data import SentimentExample

np.random.seed(42)


###############################
# Please do no change this file
###############################


class FeatureExtractor(object):
    def __init__(self):
        raise NotImplementedError("Don't call me, call my subclasses")

    @abstractmethod
    def extract_features(self, exs: List[SentimentExample]) -> np.array:
        raise NotImplementedError("Don't call me, call my subclasses")


class RawTextFeatureExtractor(FeatureExtractor):
    def __init__(self, train_exs: List[SentimentExample]) -> None:
        # Construct a vocabulary
        all_words = []
        self.max_len = 0
        for ex in train_exs:
            self.max_len = max(self.max_len, len(ex.words))
            all_words += ex.words
        distinct_words = sorted(list(set(all_words)))

        num_words = len(distinct_words)
        self.vocab = dict()
        for i in range(num_words):
            self.vocab[distinct_words[i]] = i
        self.vocab["<unk>"] = num_words
        self.vocab["<pad>"] = num_words + 1

    def extract_features(self, exs: List[SentimentExample]) -> np.array:
        feats = []

        for ex in exs:
            feat = []
            for word in ex.words:
                if word in self.vocab:
                    feat.append(self.vocab[word])
                else:
                    feat.append(self.vocab["<unk>"])

            num_words = len(feat)
            feat += [self.vocab["<pad>"]] * (self.max_len - num_words)
            feats.append(feat)

        return np.array(feats, dtype=np.int16)
