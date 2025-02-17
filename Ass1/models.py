from abc import abstractmethod
from turtle import ycor
from typing import List
import numpy as np
from sentiment_data import SentimentExample
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

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

    def fit(self, feat: np.array, y: np.array) -> None:
        pass

    def predict(self, feat: np.array) -> int:
        return np.random.randint(self.num_classes)


class NaiveBayesClassifier(SentimentClassifier):
    '''Naive Bayes classifier for sentiment classification'''

    def __init__(self, num_classes: int, alpha: float = 1.):
        # ------------------
        # Write your code here
        self.num_classes = num_classes
        self.y_prior = np.zeros(self.num_classes) + alpha
        self.word_given_class_count = None
        self.word_in_label = np.zeros(self.num_classes)
        self.alpha = alpha

        # ------------------

    def fit(self, feat: np.array, y: np.array) -> None:
        # ------------------
        # Write your code here
        n = feat.shape[1]
        self.word_given_class_count = np.zeros((self.num_classes, n)) + self.alpha
        for i in range(self.num_classes):
            self.y_prior[i] = np.log(np.sum(y == i) / len(y))
        for index, f in enumerate(feat):
            nonzero_indices = np.nonzero(f)[0]
            self.word_given_class_count[int(y[index])][nonzero_indices] += f[nonzero_indices]
        for i in range(self.num_classes):
            self.word_in_label[i] = np.sum(self.word_given_class_count[i]) + self.alpha * n
        # ------------------

    def predict(self, feat: np.array) -> int:

        # ------------------
        # Write your code here
        # feat is the feature vector for one example
        # pred is the label (0 or 1)
        post_p = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            feat_p = np.zeros(feat.shape)
            nonzero_indices = np.nonzero(feat)[0]
            for i in nonzero_indices:
                feat_p[i] = feat[i] * np.log(
                    self.word_given_class_count[c][i] / self.word_in_label[c])
            post_p[c] = self.y_prior[c] + np.sum(feat_p)
        pred = int(np.argmax(post_p))
        # ------------------

        return pred


class LogisticRegressionClassifier(SentimentClassifier):
    '''Logistic regression classifier for sentiment classification'''

    def __init__(self,
                 num_classes: int,
                 lr: float = 1,
                 batch_size: int = 64,
                 num_epochs: int = 20):

        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.theta = np.array([])

    @staticmethod
    def sigmoid(logits: np.array) -> np.array:

        # ------------------
        # Write your code here
        probs = 1 / (1 + np.exp(-logits))
        # ------------------

        return probs

    def fit(self, feat: np.array, y: np.array) -> None:

        num_samples, feature_dim = feat.shape
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # ------------------
        # Write your code here
        # initialize the parameters
        feat = np.hstack([feat, np.ones((num_samples, 1))])
        self.theta = np.zeros(feature_dim + 1)
        # ------------------

        loss_values = []
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            running_nsample = 0.0
            for i in range(num_samples // self.batch_size):
                left = i * self.batch_size
                right = min((i + 1) * self.batch_size, num_samples)

                # ------------------
                # Write your code here
                # calculate the loss and gradient for each batch and update the parameters
                batch_indices = indices[left:right]
                X_batch = feat[batch_indices]
                y_batch = y[batch_indices]
                logits = np.dot(X_batch, self.theta)
                predictions = self.sigmoid(logits)

                loss = -np.mean(
                    y_batch * np.log(predictions + 1e-8) + (1 - y_batch) * np.log(1 - predictions + 1e-8)
                )

                error = predictions - y_batch
                gradient_theta = np.dot(X_batch.T, error) / len(y_batch)
                self.theta -= self.lr * gradient_theta
                # ------------------

                running_nsample += (right - left)
                running_loss += loss * (right - left)
                if i % 200 == 0:
                    print(
                        f"Epoch {epoch:02d}\t Step: {i:03d}\t Loss: {running_loss / running_nsample:.4f}")
            loss_values.append(running_loss / running_nsample)

        # draw loss
        plt.plot(loss_values)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training loss")
        # show
        # plt.show()

    def predict(self, feat: np.array) -> int:

        # ------------------
        # Write your code here
        # feat is the feature vector for one example
        # pred is the label (0 or 1)
        feat = np.hstack([feat, np.ones(1)])
        pred = 1 if self.sigmoid(np.dot(self.theta, feat)) > 0.5 else 0
        # ------------------

        return pred


class BetterClassifier(SentimentClassifier):
    def __init__(self, num_classes: int) -> None:
        '''Feel free to define other hyperparameters.'''
        self.num_classes = num_classes

    def fit(self, feat: np.array, y: np.array) -> None:
        # ------------------
        # (Optional) Write your code here

        # ------------------

        return

    def predict(self, feat: np.array) -> int:
        pred = 0

        # ------------------
        # (Optional) Write your code here

        # ------------------

        return pred
