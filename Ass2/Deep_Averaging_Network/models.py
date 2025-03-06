import os
from abc import abstractmethod
from typing import List

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sentiment_data import SentimentExample

np.random.seed(42)


###############################
# Please complete this file
###############################


class SentimentClassifier(object):
    @abstractmethod
    def fit(self, train_exs: List[SentimentExample]) -> None:
        raise NotImplementedError("Don't call me.")

    @abstractmethod
    def predict(self, sentence: List[str]) -> int:
        raise NotImplementedError("Don't call me.")


class TrivialSentimentClassifier(SentimentClassifier):
    '''Sentiment classifier that do random prediction.'''

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def fit(self, *args) -> None:
        pass

    def predict(self, feats: np.array) -> int:
        return np.random.randint(self.num_classes, size=len(feats))


class DeepAveragingNetwork(nn.Module):
    '''Network architecture of Deep Averaginng Network'''

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int) -> None:
        '''
        params:
            input_dim: The input dimension of embedding layer
            embedding_dim: The output_dimension of embedding layer
            hidden_dim: The hidden size of FFNN
        '''
        super().__init__()

        # ------------------
        # Write your code here


        # ------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ''' Feedforward of Deep Average Network'''

        outputs = torch.Tensor()

        # ------------------
        # Write your code here


        # ------------------

        return outputs


class DANSentimentClassifier(SentimentClassifier):
    '''DAN sentiment classifier'''

    def __init__(self,
                 num_classes: int,
                 vocab: dict,
                 feature_dim: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int = 10,
                 optimizer: str = "sgd",
                 ckpt_dir: str = "ckpts",
                 tb_dir: str = "tb_logger") -> None:

        self.num_classes = num_classes
        # Vocabulary of training words
        self.vocab = vocab
        # The index of pad token
        self.pad_idx = self.vocab["<pad>"]
        self.model = DeepAveragingNetwork(
            input_dim=feature_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.batch_size = batch_size
        self.epochs = epochs
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(tb_dir)

        # ------------------
        # Write your code here



        # Implement loss function
        self.loss = nn.BCELoss()

        # ------------------

    def fit(self, train_feats: np.array, train_targets: np.array,
            val_feats: np.array, val_targets: np.array) -> None:

        num_samples = len(train_feats)

        # convert numpy into torch
        train_input_ids = torch.from_numpy(train_feats).long()
        train_targets = torch.from_numpy(train_targets[:, None]).float()
        val_input_ids = torch.from_numpy(val_feats).long()
        val_targets = torch.from_numpy(val_targets[:, None]).float()

        loss_values = []
        total_step = 0
        
        start_t = time.time()
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_nsample = 0.0

            start = time.time()
            # shuffle current data
            cur_idxs = torch.randperm(num_samples)
            train_input_ids = train_input_ids[cur_idxs]
            train_targets = train_targets[cur_idxs]
            
            for step in tqdm(range(num_samples // self.batch_size + 1), desc=f"Epoch {epoch}", leave=False):
                left = step * self.batch_size
                right = min((step+1) * self.batch_size, num_samples)

                batch_ids = train_input_ids[left:right]
                batch_y = train_targets[left:right]

                # ------------------
                # Write your code here
                # Forward and backward of NN
                

                # ------------------

                running_loss += loss.item() * (right - left)
                running_nsample += (right - left)

                self.writer.add_scalar("Loss/Train", loss.item(), total_step)
                total_step += 1

            epoch_loss = running_loss/running_nsample
            loss_values.append(epoch_loss)

            end = time.time()

            # ------------------
            # Write your code here
            # Compuate loss and accuracy on validation set


            # ------------------

            accuracy = accuracy_score(
                preds.data.numpy(), val_targets.data.numpy())

            self.writer.add_scalar("Loss/val", loss.item(), total_step)
            self.writer.add_scalar("Accuracy", accuracy, total_step)

            torch.save(self.model.state_dict(),
                       f"{self.ckpt_dir}/epoch_{epoch}.pth")
            print(
                f"Epoch: {epoch:02d}\ttime: {end-start:.4f}s\tTrain Loss: {epoch_loss:.4f}\tValid Loss: {loss.item():.4f}\tAccuracy: {accuracy:.4f}")
        end_t = time.time()
        print(f"Total training time: {end_t - start_t :.4f}")

    def predict(self, feats: np.array) -> torch.Tensor:
        '''Predict binary labels for input samples'''
        preds = torch.Tensor()

        # ------------------
        # Write your code here


        # ------------------

        return preds.detach()


class BetterDeepAveragingNetwork(nn.Module):
    '''Network architecture of Better Deep Averaging Network with dropout'''

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, dropout_rate: float = 0.2) -> None:
        '''
        params:
            input_dim: The input dimension of embedding layer
            embedding_dim: The output_dimension of embedding layer
            hidden_dim: The hidden size of FFNN
            dropout_rate: Dropout probability
        '''
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        self.model = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ''' Feedforward of Better Deep Average Network'''
        
        # Generate embeddings
        embeddings = self.embedding(input_ids)

        # Compute average embeddings
        sum_embeddings = torch.sum(embeddings * attention_mask.unsqueeze(-1), dim=1)
        seq_len = torch.sum(attention_mask, dim=1, keepdim=True)
        avg_embeddings = sum_embeddings / seq_len
        
        # Return probabilities
        outputs = self.model(avg_embeddings)

        return outputs


class BetterClassifier(SentimentClassifier):
    '''A better version of DAN sentiment classifier with dropout layers and additional hidden layer'''

    def __init__(self,
                 num_classes: int,
                 vocab: dict,
                 feature_dim: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 batch_size: int,
                 learning_rate: float,
                 dropout_rate: float = 0.2,
                 epochs: int = 10,
                 optimizer: str = "adam",
                 ckpt_dir: str = "ckpts_better",
                 tb_dir: str = "tb_logger_better") -> None:

        self.num_classes = num_classes
        # Vocabulary of training words
        self.vocab = vocab
        # The index of pad token
        self.pad_idx = self.vocab["<pad>"]
        self.model = BetterDeepAveragingNetwork(
            input_dim=feature_dim, 
            embedding_dim=embedding_dim, 
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        self.batch_size = batch_size
        self.epochs = epochs
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(tb_dir)

        # Implement optimizers with weight decay for regularization
        if optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=1e-4
            )
        elif optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=1e-4
            )

        # Implement loss function
        self.loss = nn.BCELoss()

    def fit(self, train_feats: np.array, train_targets: np.array,
            val_feats: np.array, val_targets: np.array) -> None:

        num_samples = len(train_feats)

        # convert numpy into torch
        train_input_ids = torch.from_numpy(train_feats).long()
        train_targets = torch.from_numpy(train_targets[:, None]).float()
        val_input_ids = torch.from_numpy(val_feats).long()
        val_targets = torch.from_numpy(val_targets[:, None]).float()

        loss_values = []
        total_step = 0
        best_val_acc = 0.0
        
        start_t = time.time()
        for epoch in range(self.epochs):
            # Set model to training mode
            self.model.train()
            running_loss = 0.0
            running_nsample = 0.0

            start = time.time()
            # shuffle current data
            cur_idxs = torch.randperm(num_samples)
            train_input_ids = train_input_ids[cur_idxs]
            train_targets = train_targets[cur_idxs]
            
            for step in tqdm(range(num_samples // self.batch_size + 1), desc=f"Epoch {epoch}", leave=False):
                left = step * self.batch_size
                right = min((step+1) * self.batch_size, num_samples)

                batch_ids = train_input_ids[left:right]
                batch_y = train_targets[left:right]

                mask = batch_ids != self.pad_idx
                outputs = self.model(batch_ids, mask)

                # compute loss
                loss = self.loss(outputs, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * (right - left)
                running_nsample += (right - left)

                self.writer.add_scalar("Loss/Train", loss.item(), total_step)
                total_step += 1

            epoch_loss = running_loss/running_nsample
            loss_values.append(epoch_loss)

            end = time.time()

            # Evaluation mode
            self.model.eval()
            with torch.no_grad():
                mask = val_input_ids != self.pad_idx
                outputs = self.model(val_input_ids, mask)
                val_loss = self.loss(outputs, val_targets)
                preds = torch.round(outputs).squeeze().long()

            accuracy = accuracy_score(
                preds.data.numpy(), val_targets.data.numpy())

            # Save best model
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                torch.save(self.model.state_dict(),
                       f"{self.ckpt_dir}/best_model.pth")

            self.writer.add_scalar("Loss/val", val_loss.item(), total_step)
            self.writer.add_scalar("Accuracy", accuracy, total_step)

            print(
                f"Epoch: {epoch:02d}\ttime: {end-start:.4f}s\tTrain Loss: {epoch_loss:.4f}\tValid Loss: {val_loss.item():.4f}\tAccuracy: {accuracy:.4f}")
        end_t = time.time()
        print(f"Total training time: {end_t - start_t:.4f}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")

    def predict(self, feats: np.array) -> torch.Tensor:
        '''Predict binary labels for input samples'''
        # Set model to evaluation mode
        self.model.eval()
        
        input_ids = torch.from_numpy(feats).long()
        mask = input_ids != self.pad_idx

        with torch.no_grad():
            outputs = self.model(input_ids, mask)
            preds = torch.round(outputs).squeeze().long()

        return preds.detach()