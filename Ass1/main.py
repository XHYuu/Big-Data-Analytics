import argparse
from typing import List
import numpy as np
from sentiment_data import load_sentiment_examples, SentimentExample
from models import SentimentClassifier, NaiveBayesClassifier, LogisticRegressionClassifier, TrivialSentimentClassifier, BetterClassifier
from feature_extractor import BoWFeatureExtractor, FeatureExtractor, BetterFeatureExtractor
from metrics import ClassificationMetrics


np.random.seed(42)


###############################
# Please do no change this file
###############################


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, default="trivial",
                        help="model to run (trivial, nb, lr, or better)")
    parser.add_argument("--feats", type=str, default="bow",
                        help="feature extractor to use (bow, or better)")
    parser.add_argument("--train_file", type=str,
                        default="data/train.csv", help="path to training file")
    parser.add_argument("--val_file", type=str,
                        default="data/valid.csv", help="path to validation file")
    parser.add_argument("--test_file", type=str,
                        default="data/test.csv", help="path to test file")
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()
    return args


def build_feat_extractor(args, train_exs: List[SentimentExample]):
    if args.model == "trivial":
        feat_extractor = None
    elif args.feats == "bow":
        feat_extractor = BoWFeatureExtractor(train_exs)
    elif args.feats == "better":
        feat_extractor = BetterFeatureExtractor(train_exs)
    else:
        raise Exception(
            "Pass in trivial, bow, or better to run the appropriate algorithm")

    return feat_extractor


def train(args, train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_classes: int = 2):
    # Create the model
    if args.model == "trivial":
        model = TrivialSentimentClassifier(num_classes=num_classes)
    elif args.model == "nb":
        model = NaiveBayesClassifier(num_classes=num_classes)
    elif args.model == "lr":
        model = LogisticRegressionClassifier(num_classes=num_classes)
    elif args.model == "better":
        model = BetterClassifier(num_classes=num_classes)
    else:
        raise Exception(
            "Pass in trivial, nb, lr or better to run the appropriate algorithm")

    # Extract feature
    if feat_extractor:
        train_feat = feat_extractor.extract_feature_for_multiple_exs(train_exs)
        targets = np.zeros(len(train_feat))
        for i, ex in enumerate(train_exs):
            targets[i] = ex.label
    else:
        train_feat = None
        targets = None
    # Train model
    model.fit(train_feat, targets)

    return model


def evaluate(model, exs: List[SentimentExample], feat_extractor: FeatureExtractor):
    ground_truths = np.array([ex.label for ex in exs])
    predictions = np.zeros(len(exs))
    for i, ex in enumerate(exs):
        if feat_extractor:
            feat = feat_extractor.extract_feature_per_ex(ex)
        else:
            feat = None
        predictions[i] = model.predict(feat)
    metrics = ClassificationMetrics(
        predictions, ground_truths, model.num_classes)
    print(f"Accuracy: {metrics.compute_accuracy():.4f}")
    print(f"Precision: {metrics.compute_precision():.4f}")
    print(f"Recall: {metrics.compute_recall():.4f}")
    print(f"F-score: {metrics.compute_f_score():.4f}")


def export_submission_file(model: SentimentClassifier, test_exs: List[SentimentExample], feat_extractor: FeatureExtractor):
    with open("data/test_predictions.csv", "w") as f:
        f.write("Prediction\n")
        for ex in test_exs:
            if feat_extractor:
                feat = feat_extractor.extract_feature_per_ex(ex)
            else:
                feat = None
            label = model.predict(feat)
            f.write(f"{label}\n")


def main():
    args = parse_args()

    # Load train data
    train_exs = load_sentiment_examples(args.train_file, "train")

    # Load valid data
    val_exs = load_sentiment_examples(args.val_file, "valid")

    # build feat extractor
    feat_extractor = build_feat_extractor(args, train_exs)

    # Train a sentiment classifier
    model = train(args, train_exs, feat_extractor, args.num_classes)

    print("======== Valid accuracy =========")
    # Evaluate the performance of our classifier
    evaluate(model, val_exs, feat_extractor)

    # export model predictions to `data/test_predictions.csv`
    # please submit this file if you have done bonus.
    test_exs = load_sentiment_examples(args.test_file, "test")
    export_submission_file(model, test_exs, feat_extractor)


if __name__ == "__main__":
    main()
