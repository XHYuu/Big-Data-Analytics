import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import create_cls_dataloader, vocab
from models import RNNClsModel

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

def train_rnn_cls(args, train_dataloader, valid_dataloader):
    model = RNNClsModel(
        len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)

    # ------------------
    # Write your code here
    # 1. Define loss_func (Binary cross entropy loss)
    # 2. Define optimizer (Recommend to use Adam)


    # ------------------

    loss_list = []
    acc_list = []

    start = time.time()
    for epoch in range(args.epochs):
        total_loss = 0.
        model.train()

        for _, (sent, targets) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch:02d}", leave=False):

            # ------------------
            # Write your code here
            # The shape of sent is [batch_size, fix_length_of_sequence]
            # The shape of targets is [batch_size, 1]
            # The procedure of this part:
            # 1. Forward
            # 2. Compute loss
            # 3. Zero gradients
            # 4. Backward
            # 5. Updata network parameters


            # ------------------

            total_loss += loss.item()

        # validation part
        all_preds = []
        all_targets = []
        model.eval()
        with torch.no_grad():
            for _, (val_sent, val_y) in enumerate(valid_dataloader):
                preds = torch.round(model(val_sent))
                all_preds.append(preds.cpu().data.numpy())
                all_targets.append(val_y.cpu().data.numpy())

        val_preds = np.vstack(all_preds)
        val_targets = np.vstack(all_targets)

        val_acc = np.mean(val_preds.squeeze() == val_targets.squeeze())

        avg_loss = total_loss / len(train_dataloader)
        loss_list.append(avg_loss)
        acc_list.append(val_acc)
        print(
            f"Epoch: {epoch:02d}\tLoss: {avg_loss:.4f}\tVal acc: {val_acc:.4f}")

    end = time.time()
    print(f"Total training time: {end-start:.4f}s")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    ax1.plot(loss_list)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax2.plot(acc_list)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("valid Accuracy")
    plt.savefig("rnn_cls.jpg")
    plt.show()

    return model


def evaluate_your_model(model, valid_dataloader):
    model.eval()
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for _, (val_sent, val_y) in enumerate(valid_dataloader):
            preds = torch.round(model(val_sent))
            all_preds.append(preds.cpu().data.numpy())
            all_targets.append(val_y.cpu().data.numpy())

    val_preds = np.vstack(all_preds)
    val_targets = np.vstack(all_targets)

    val_acc = np.mean(val_preds.squeeze() == val_targets.squeeze())

    print("-"*40)
    print(f"Valid Accuracy: {val_acc:.4f}")
    print("-"*40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/train_cls.csv")
    parser.add_argument("--valid_file", type=str, default="data/valid_cls.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.002)
    args = parser.parse_args()

    # Load data
    train_dataloader = create_cls_dataloader(
        args.batch_size, args.train_file, "train")
    valid_dataloader = create_cls_dataloader(
        args.batch_size, args.valid_file, "valid")

    model = train_rnn_cls(args, train_dataloader, valid_dataloader)

    evaluate_your_model(model, valid_dataloader)


if __name__ == "__main__":
    main()
