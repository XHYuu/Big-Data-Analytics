import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Vocab


####################################
# Models for Q1
####################################


class RNNClsModel(nn.Module):
    def __init__(self,
                 vocab_size: int = 27,
                 embed_dim: int = 32,
                 hidden_dim: int = 32
                 ) -> None:
        super().__init__()

        # ------------------
        # Write your code here
        # hidden_dim: the dimension of hidden state in RNN and feed-forward network
        # Hint: you can use nn.Sequential to package the feed-forward network as a module
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.LSTM(input_size=embed_dim,
                           hidden_size=hidden_dim,
                           batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Assumes binary classification; adjust as needed
        )

        # ------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.Tensor()

        # ------------------
        # Write your code here
        # The shape of x is [batch_size, length_of_sequence]
        # The shape of out is [batch_size, 1]
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        last_timestep = rnn_out[:, -1, :]
        out = self.classifier(last_timestep)
        # ------------------

        return out
