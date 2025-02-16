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
        


        # ------------------

    def compute_confusion_matrix(self):
        cm = np.array([])
        
        # ------------------
        # Write your code here



        # ------------------

        return cm

    def compute_accuracy(self):
        accuracy = 0.0

        # ------------------
        # Write your code here


        # ------------------

        return accuracy

    def compute_precision(self, average: str = "macro"):
        precision = 0.0

        # ------------------
        # Write your code here



        # ------------------

        return precision

    def compute_recall(self, average: str = "macro"):
        recall = 0.0

        # ------------------
        # Write your code here



        # ------------------

        return recall

    def compute_f_score(self, average="macro", beta=1.0):
        f_score = 0.0

        # ------------------
        # Write your code here



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
    print("-"*40)
    print("Test for macro average: ")
    accuracy = metric.compute_accuracy()
    print(f"Accuracy: {accuracy:.4f}")
    precision = metric.compute_precision(average="macro")
    print(f"Precision: {precision:.4f}")
    recall = metric.compute_recall(average="macro")
    print(f"recall: {recall:.4f}")
    f_score = metric.compute_f_score(average="macro")
    print(f"F score: {f_score:.4f}")

    print("-"*40)
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
    print("-"*40)

    from sklearn.metrics import precision_recall_fscore_support
    # You may verify your code by comparing with sklearn package 
    print(precision_recall_fscore_support(target, pred, average="macro"))
    print(precision_recall_fscore_support(target, pred, average="micro"))


if __name__ == "__main__":
    test_metrics()
