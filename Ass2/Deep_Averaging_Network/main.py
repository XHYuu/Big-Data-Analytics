import argparse
from argparse import ArgumentParser
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from feature_extractor import FeatureExtractor, RawTextFeatureExtractor
from models import (DANSentimentClassifier, SentimentClassifier,
                    TrivialSentimentClassifier, BetterClassifier)
from sentiment_data import SentimentExample, load_sentiment_examples

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


###############################
# Please do no change this file
###############################


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="dan",
                        help="model to run (trivial, dan, or better)")
    parser.add_argument("--feats", type=str, default="raw",
                        help="feature extractor to use (raw, or better)")
    parser.add_argument("--train_file", type=str,
                        default="data/train.csv", help="path to training file")
    parser.add_argument("--val_file", type=str,
                        default="data/valid.csv", help="path to validation file")
    parser.add_argument("--test_file", type=str,
                        default="data/test.csv", help="path to test file")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    return args


def build_feat_extractor(args, train_exs: List[SentimentExample]):
    if args.feats == "trivial":
        feat_extractor = None
    elif args.feats == "raw":
        feat_extractor = RawTextFeatureExtractor(train_exs)
    else:
        raise Exception(
            "Pass in raw or better to run the appropriate algorithm")

    return feat_extractor


def train(args: ArgumentParser, train_exs: List[SentimentExample], val_exs: List[SentimentExample],
          feat_extractor: FeatureExtractor, num_classes: int = 2):

    # Create the model
    if args.model == "trivial":
        model = TrivialSentimentClassifier(num_classes=num_classes)
    elif args.model == "dan":
        model = DANSentimentClassifier(
            num_classes=num_classes,
            vocab=feat_extractor.vocab,
            feature_dim=len(feat_extractor.vocab),
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            optimizer=args.optimizer)
    elif args.model == "better":
        model = BetterClassifier(
            num_classes=num_classes,
            vocab=feat_extractor.vocab,
            feature_dim=len(feat_extractor.vocab),
            embedding_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            dropout_rate=0.4,
            epochs=20,
            optimizer=args.optimizer
        )
    else:
        raise Exception(
            "Pass in trivial, dan or better to run the appropriate algorithm")

    if feat_extractor:
        # Extract train features
        train_feats = feat_extractor.extract_features(train_exs)
        train_targets = np.zeros(len(train_feats))
        for i, ex in enumerate(train_exs):
            train_targets[i] = ex.label
        
        # Extract valid features
        val_feats = feat_extractor.extract_features(val_exs)
        val_targets = np.zeros(len(val_feats))
        for i, ex in enumerate(val_exs):
            val_targets[i] = ex.label

    else:
        train_feats = None
        train_targets = None
        val_feats = None
        val_targets = None

    # Train model
    model.fit(train_feats, train_targets, val_feats, val_targets)

    return model


def evaluate(model, exs: List[SentimentExample], feat_extractor: FeatureExtractor):
    ground_truths = np.array([ex.label for ex in exs])
    feats = feat_extractor.extract_features(exs)
    predictions = model.predict(feats)

    accuracy = accuracy_score(ground_truths, predictions)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        ground_truths, predictions, average="macro")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F-score: {f_score:.4f}")


def export_submission_file(model: SentimentClassifier, test_exs: List[SentimentExample], feat_extractor: FeatureExtractor):
    feats = feat_extractor.extract_features(test_exs)
    predictions = model.predict(feats)

    with open("data/test_predictions.csv", "w") as f:
        f.write("Prediction\n")
        for pred in predictions:
            f.write(f"{pred}\n")


def main():
    args = parse_args()

    # Load train data
    train_exs = load_sentiment_examples(args.train_file, "train")

    # Load valid data
    val_exs = load_sentiment_examples(args.val_file, "valid")
    
    # build feat extractor
    feat_extractor = build_feat_extractor(args, train_exs)

    # Train a sentiment classifier
    model = train(args, train_exs, val_exs, feat_extractor, args.num_classes)

    print("======== Valid accuracy =========")
    # Evaluate the performance of our classifier
    evaluate(model, val_exs, feat_extractor)

    test_exs = load_sentiment_examples(args.test_file, "test")
    export_submission_file(model, test_exs, feat_extractor)


if __name__ == "__main__":
    main()
