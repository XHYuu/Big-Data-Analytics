from abc import abstractmethod
from typing import Counter, List
import numpy as np
from tqdm import tqdm
from sentiment_data import SentimentExample
from collections import Counter
from scipy.sparse import coo_matrix
from sentiment_data import load_sentiment_examples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


np.random.seed(42)


###############################
# Please complete this file
###############################


class FeatureExtractor(object):
    def __init__(self):
        raise NotImplementedError("Don't call me, call my subclasses")

    @abstractmethod
    def extract_feature_for_multiple_exs(self, exs: List[List[SentimentExample]]) -> np.array:
        raise NotImplementedError("Don't call me, call my subclasses")

    @abstractmethod
    def extract_feature_per_ex(self, ex: SentimentExample) -> np.array:
        raise NotImplementedError("Don't call me, call my subclasses")


class BoWFeatureExtractor(FeatureExtractor):
    '''
    Extracts bag-of-words features from a sentence. 
    '''

    def __init__(self, train_exs: List[SentimentExample]):
        '''Build your vocabulary'''

        # ------------------
        # Write your code here


        # ------------------

    def extract_feature_for_multiple_exs(self, exs: List[SentimentExample]):
        ''' Extract feature for multiple sentiment examples in training or validation phase.
        '''

        feat = np.array([])

        # ------------------
        # Write your code here


        # ------------------

        return feat

    def extract_feature_per_ex(self, ex: SentimentExample) -> np.array:
        ''' Extract feature for each sentiment example in test phase.
        '''

        feat = np.array([])

        # ------------------
        # Write your code here


        # ------------------

        return feat


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, train_exs: List[SentimentExample]):

        # ------------------
        # (Optional) Write your code here


        # ------------------
    
    def extract_feature_for_multiple_exs(self, exs: List[SentimentExample]):   

        feat = np.array([])

        # ------------------
        # (Optional) Write your code here


        # ------------------

        return feat

    def extract_feature_per_ex(self, ex: SentimentExample) -> np.array:
        
        feat = np.array([])

        # ------------------
        # (Optional) Write your code here


        # ------------------

        return feat
    

def test_bow():
    # test code for bow feature extractor
    train_exs = load_sentiment_examples("data/train.csv", "train")

    feat_extractor = BoWFeatureExtractor(train_exs)
    feat = feat_extractor.extract_feature_for_multiple_exs(train_exs)
    print("-"*40)
    print(
        f"Number of samples: {feat.shape[0]}\t Feature dimension: {feat.shape[1]}")
    word_num_per_cnt = np.sum(feat, axis=1)
    print(
        f"5 percentile of number of words: {np.percentile(word_num_per_cnt, 5)}")
    print(
        f"95 percentile of number of words: {np.percentile(word_num_per_cnt, 95)}")


if __name__ == "__main__":
    test_bow()
