import numpy as np
from scipy.sparse import coo_matrix

np.random.seed(42)


###############################
# Please complete this file
###############################


class ClassificationMetrics(object):
    def __init__(self,
                 pred: np.array,
                 target: np.array,
                 num_classes: int) -> None:
        ''' Calculate classification metrics according to predictions and ground truth
            Params:
                pred (1D np.array): predicted label of each sample
                target (1D np.array): ground truth label of each sample
                num_classes (int): number of classes
        '''

        # ------------------
        # Write your code here
        self.pred = pred
        self.target = target
        self.num_classes = num_classes
        self.cm = self.compute_confusion_matrix()
        # ------------------

    def compute_confusion_matrix(self):
        # ------------------
        # Write your code here
        # Multiclass Classifiers
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for t, p in zip(self.target, self.pred):
            cm[t, p] += 1
        # ------------------

        return cm

    def compute_accuracy(self):
        accuracy = 0.0

        # ------------------
        # Write your code here
        total = len(self.target)
        count_true = np.trace(self.cm)
        accuracy = count_true / total
        # ------------------

        return accuracy

    def compute_precision(self, average: str = "macro"):
        precision = 0.0

        # ------------------
        # Write your code here
        if average == "macro":
            cls = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                total_pred = np.sum(self.cm, axis=0)[i]
                if total_pred == 0:
                    continue
                cls[i] = self.cm[i][i] / total_pred

            precision = np.mean(cls)
        else:
            total = len(self.target)
            pred_true = np.trace(self.cm)
            precision = pred_true / total

        # ------------------

        return precision

    def compute_recall(self, average: str = "macro"):
        recall = 0.0

        # ------------------
        # Write your code here
        if average == "macro":
            cls = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                total_pred = np.sum(self.cm, axis=1)[i]
                if total_pred == 0:
                    continue
                cls[i] = self.cm[i][i] / total_pred

            recall = np.mean(cls)
        else:
            total = len(self.target)
            pred_true = np.trace(self.cm)
            recall = pred_true / total
        # ------------------

        return recall

    def compute_f_score(self, average="macro", beta=1.0):
        f_score = 0.0

        # ------------------
        # Write your code here
        if average == "macro":
            cls = np.zeros(self.num_classes)
            for i in range(self.num_classes):
                # precision
                total_pred_precision = np.sum(self.cm, axis=0)[i]
                if total_pred_precision == 0:
                    cls_precision = 0
                else:
                    cls_precision = self.cm[i][i] / total_pred_precision
                # recall
                total_pred_recall = np.sum(self.cm, axis=1)[i]
                if total_pred_recall == 0:
                    cls_recall = 0
                else:
                    cls_recall = self.cm[i][i] / total_pred_recall
                cls[i] = (1 + (beta ** 2)) * (cls_precision * cls_recall) / ((beta ** 2) * cls_precision + cls_recall)
            f_score = np.mean(cls)
        else:
            precision = np.trace(self.cm) / len(self.target)
            recall = np.trace(self.cm) / len(self.target)
            f_score = (1 + (beta ** 2)) * (precision * recall) / ((beta ** 2) * precision + recall)
        # ------------------

        return f_score


def test_metrics():
    # test code of metrics
    num_classes = 5

    # generate data
    pred = np.random.randint(num_classes, size=1000)
    target = np.random.randint(num_classes, size=1000)
    random_idx = np.random.randint(1000, size=600)
    pred[random_idx] = target[random_idx] = 0
    random_idx = np.random.randint(1000, size=200)
    pred[random_idx] = target[random_idx] = 1

    metric = ClassificationMetrics(pred, target, num_classes)
    print("-" * 40)
    print("Test for macro average: ")
    accuracy = metric.compute_accuracy()
    print(f"Accuracy: {accuracy:.4f}")
    precision = metric.compute_precision(average="macro")
    print(f"Precision: {precision:.4f}")
    recall = metric.compute_recall(average="macro")
    print(f"recall: {recall:.4f}")
    f_score = metric.compute_f_score(average="macro")
    print(f"F score: {f_score:.4f}")

    print("-" * 40)
    print("Test for micro average: ")
    metric = ClassificationMetrics(pred, target, num_classes)
    accuracy = metric.compute_accuracy()
    print(f"Accuracy: {accuracy:.4f}")
    precision = metric.compute_precision(average="micro")
    print(f"Precision: {precision:.4f}")
    recall = metric.compute_recall(average="micro")
    print(f"recall: {recall:.4f}")
    f_score = metric.compute_f_score(average="micro")
    print(f"F score: {f_score:.4f}")
    print("-" * 40)

    from sklearn.metrics import precision_recall_fscore_support
    # You may verify your code by comparing with sklearn package 
    print(precision_recall_fscore_support(target, pred, average="macro"))
    print(precision_recall_fscore_support(target, pred, average="micro"))


if __name__ == "__main__":
    test_metrics()
