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

#### Use Logistical Classifier to make prediction *(Combine bias in theta)*
```bash
python main.py --model lr --feats bow
```