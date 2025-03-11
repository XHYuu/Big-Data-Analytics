import string

import pandas as pd
import torch
from torch.utils.data import DataLoader


class Vocab(object):
    def __init__(self):
        self.int2obj = dict()
        self.obj2int = dict()

        # ------------------
        # Write your code here
        # Define a vocabulary here. (a-z+" " -> 0-26)
        self.vocabulary = {chr(idx): idx - ord('a')
                           for idx in range(ord('a'), ord('z') + 1)} | {' ': 26}
        # ------------------

    def index_of(self, x: str):
        ''' Get index of a given character x'''
        # ------------------
        # Write your code here
        if x in self.vocabulary:
            return self.vocabulary[x]
        raise ValueError(f"Character '{x}' not in vocabulary.")

        # ------------------

    def object_of(self, x: int):
        ''' Get character of a given index'''

        # ------------------
        # Write your code here
        if x == 26:
            return ' '
        elif ord('a') <= x <= ord('z'):
            return chr(ord('a') + x)
        raise ValueError(f"Index '{x}' not in vocabulary.")
        # ------------------

    def __len__(self):
        ''' Return the size of your vocabulary'''

        # ------------------
        # Write your code here
        return len(self.vocabulary)
        # ------------------


vocab = Vocab()


def generate_cls_batch(batch):
    # You also need vocab to convert each character into an index
    X = []
    Y = []
    for sent, y in batch:
        indices = [vocab.index_of(token) for token in sent]
        X.append(indices)
        Y.append(y)

    return torch.tensor(X), torch.tensor(Y).unsqueeze(1).float()


def create_cls_dataloader(batch_size: int, file_path: str, split: str = "train") -> DataLoader:
    data_df = pd.read_csv(file_path)
    X = data_df["Sentence"].values.tolist()
    Y = data_df["label"].values

    if split == "train":
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(list(zip(X, Y)), batch_size=batch_size,
                            shuffle=shuffle, collate_fn=generate_cls_batch)

    return dataloader
