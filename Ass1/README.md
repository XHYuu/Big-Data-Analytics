# Sentiment Classification on Movie Reviews

### Main Goal

*This project try to implement basic feature extraction and classification algorithms for text classification.*

#### Use basic random way to implement Sentiment Classifier

```bash
python main.py --model trivial
```

#### Use Bayes Classifier to make prediction

```bash
python main.py --model nb --feats bow
```

##### Result

| Valid accuracy    |
|-------------------|
| Accuracy: 0.8519  |
| Precision: 0.8520 |
| Recall: 0.8485    |
| F-score: 0.8499   |

#### Use Logistical Classifier to make prediction *(Combine bias in theta)*

```bash
python main.py --model lr --feats bow
```

##### Result

| Valid accuracy    |
|-------------------|
| Accuracy: 0.8554  |
| Precision: 0.8543 |
| Recall: 0.8536    |
| F-score: 0.8539   |

#### Use n-grams to make Classifier

```bash
python main.py --model nb --feats better
```

##### Result

| Valid accuracy    |
|-------------------|
| Accuracy: 0.8499  |
| Precision: 0.8487 |
| Recall: 0.8481    |
| F-score: 0.8484   |