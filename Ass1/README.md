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
| Accuracy: 0.8511  |
| Precision: 0.8495 |
| Recall: 0.8502    |
| F-score: 0.8498   |

#### Use n-grams and Logistic to make Classifier
```bash
python main.py --model better --feats better
```
*with learning rate = 1 & batch size = 64.*  

| Valid accuracy    |
|-------------------|
| Accuracy: 0.8835  |
| Precision: 0.8868 |
| Recall: 0.8786    |
| F-score: 0.8813   |

*with learning rate = 0.5 & batch size = 32.* 

| Valid accuracy    |
|-------------------|
| Accuracy: 0.8841  |
| Precision: 0.8876 |
| Recall: 0.8792    |
| F-score: 0.8819   |
