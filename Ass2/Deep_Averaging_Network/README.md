## Use basic method to train a DAN

**record different hyperparameter for same model**

```bash
python main.py --model dan --lr 0.001
```

```bash
python main.py --model dan --lr 0.1
python main.py --model dan --lr 0.01
python main.py --model dan --lr 0.001
python main.py --model dan --lr 0.0001
python main.py --model dan --lr 0.00001
```

| Learning rate | Accuracy |  time   |
|:-------------:|:--------:|:-------:|
|     1e-1      |  0.8773  | 35.8079 |
|     1e-2      |  0.8902  | 35.0313 |
|     1e-3      |  0.8839  | 34.1291 |
|     1e-4      |  0.7684  | 34.5045 |
|     1e-5      |  0.5998  | 35.8881 |

```bash
python main.py --model dan --lr 0.001 --embed_dim 50
python main.py --model dan --lr 0.001 --embed_dim 100
python main.py --model dan --lr 0.001 --embed_dim 400
```


| Embedding dimension | Accuracy | 
|:-------------------:|:--------:| 
|         50          |  0.8777  |
|         100         |  0.8839  |
|         400         |  0.8876  | 
