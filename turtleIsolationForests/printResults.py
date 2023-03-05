from sklearn.metrics import roc_curve, auc
from pandas import DataFrame
import numpy as np
from time import time
import typing

def notebook_visual_printout(X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, model: any) -> None:
    train_labels_np = train_labels.to_numpy()
    test_labels_np = test_labels.to_numpy()
    start_time = time()
    model.fit(X_train, train_labels)
    fit_time = time() - start_time
    print("Time to fit model: " + str(fit_time))
    print("Threshold: " + str(model.threshold))
    print("\nTraining set results:")
    train_scores = model.train_scores
    train_predictions = train_scores > model.threshold
    print_results(train_predictions, train_labels_np)
    print("auroc: " + str(get_auroc_value(train_scores, train_labels_np)))
    start_time = time()
    scores, predictions = model.predict(X_test, test_labels)
    test_predict_time = time() - start_time
    print("\nTime to predict test data: " + str(test_predict_time))
    print("Test set results:")
    print_results(predictions, test_labels_np)
    print("auroc: " + str(get_auroc_value(scores, test_labels_np)))
    print("\n")

def csv_printout(runs: int, X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, model: any) -> None:
    print("run,precision,recall,f1,auroc,test_predict_time")
    for i in range(runs):
        test_labels_np = test_labels.to_numpy()
        model.fit(X_train, train_labels)
        start_time = time()
        scores, predictions = model.predict(X_test, test_labels_np)
        test_predict_time = time() - start_time
        TA, FA, FN, TN = calc_confusion(predictions, test_labels_np)
        precision, recall, f1 = calc_f1(TA, FA, FN, TN)
        auroc = get_auroc_value(scores, test_labels_np)
        print(str(i) + "," + str(precision) + "," + str(recall) + "," + str(f1) + "," + str(auroc) + "," + str(test_predict_time))

def get_auroc_value(scores: np.ndarray[np.float64], labels: np.ndarray[np.bool8]) -> float:
    (fpr, tpr, _) = get_auroc_points(scores, labels)
    return get_auc(fpr, tpr)

def get_auroc_points(scores: np.ndarray[np.float64], labels: np.ndarray[np.bool8]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return roc_curve(labels, scores, pos_label=True)

def get_auc(x: np.ndarray, y: np.ndarray) -> float:
    return auc(x, y)

def print_results(predictions: np.ndarray[np.bool8], labels: np.ndarray[np.bool8]) -> None:
    TA, FA, FN, TN = calc_confusion(predictions, labels)

    precision, recall, f1 = calc_f1(TA, FA, FN, TN)

    print_by_result(TA, FA, FN, TN, precision, recall, f1)

def print_by_result(TA: int, FA: int, FN: int, TN: int, precision: float, recall: float, f1: float):
    print("true anomalies: " + str(TA))
    print("false anomalies: " + str(FA))
    print("false normals: " + str(FN))
    print("true normals: " + str(TN))
    print("precision: " + str(precision))
    print("recall: " + str(recall))
    print("f1-score: " + str(f1))

def calc_confusion(predictions: np.ndarray[np.bool8], labels: np.ndarray[np.bool8]) -> np.ndarray[np.float64]:
    TA = sum(labels & predictions)
    FA = sum(~labels & predictions)
    FN = sum(labels & ~predictions)
    TN = sum(~labels & ~predictions)
    return TA, FA, FN, TN

def calc_f1(TA: int, FA: int, FN: int, TN: int) -> (float, float, float):
    if TA == 0:
        return 0, 0, 0
    else:
        precision = TA / (TA + FA)
        recall = TA / (TA + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
