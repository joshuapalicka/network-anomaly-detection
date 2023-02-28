from sklearn.metrics import roc_curve, auc
from pandas import DataFrame
import numpy as np
import typing

def notebook_visual_printout(X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, model: any) -> None:
    model.fit(X_train, train_labels)
    print("Threshold: " + str(model.threshold))
    print("\nTraining set results:")
    train_predictions = model.train_scores
    train_predictions['is_normal'] = train_labels
    print_results(train_predictions)
    print("auroc: " + str(get_auroc_value(train_predictions)))
    print("\nTest set results:")
    predictions = model.predict(X_test)
    predictions['is_normal'] = test_labels
    print_results(predictions)
    print("auroc: " + str(get_auroc_value(predictions)))
    print("\n")

def csv_printout(runs: int, X_train: DataFrame, X_test: DataFrame, train_labels: DataFrame, test_labels: DataFrame, model: any) -> None:
    print("run,precision,recall,f1,auroc")
    for i in range(runs):
        model.fit(X_train, train_labels)
        predictions = model.predict(X_test)
        predictions['is_normal'] = test_labels
        TA, FA, FN, TN = return_results(predictions)
        precision, recall, f1 = calc_f1(TA, FA, FN, TN)
        auroc = get_auroc_value(predictions)
        print(str(i) + "," + str(precision) + "," + str(recall) + "," + str(f1) + ", " + str(auroc))

def get_auroc_value(predictions: DataFrame) -> float:
    (fpr, tpr, _) = get_auroc_points(predictions)
    return get_auc(fpr, tpr)

def get_auroc_points(predictions: DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = predictions['is_normal']
    y_score = predictions['anomaly_score']
    return roc_curve(y_true, y_score, pos_label=0)

def get_auc(x: np.ndarray, y: np.ndarray) -> float:
    return auc(x, y)
    
def storage():
    sorted_predictions = predictions.sort_values('anomaly_score')
    anomalies = predictions.loc[predictions['is_normal'] == 0]
    normals = predictions.loc[predictions['is_normal'] != 0]
    TA = len(anomalies)
    FA = 0
    FN = len(normals)
    TN = 0

    aurocPoints = []
    for i in range(len(sorted_predictions)):
        if sorted_predictions.iloc[i]['is_normal'] == 0:
            FA += 1
            TA -= 1
        else:
            FN -= 1
            TN += 1
        if TN + FA != 0 and TA + FN != 0:
            recall_tpr = TA / (TA + FN)
            fallout_fpr = FA / (FA + TN)
            aurocPoints.append((sorted_predictions.iloc[i]['anomaly_score'], recall_tpr, fallout_fpr))
    return aurocPoints

def print_results(predictions: DataFrame) -> None:
    TA, FA, FN, TN = return_results(predictions)

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

def return_results(predictions: DataFrame) -> (int, int, int, int):
    anomalies = predictions.loc[predictions['is_normal'] == 0]
    normals = predictions.loc[predictions['is_normal'] != 0]

    true_anomalies = anomalies.loc[predictions['predicted_as_anomaly'] == True]
    false_anomalies = normals.loc[predictions['predicted_as_anomaly'] == True]
    false_normals = anomalies.loc[predictions['predicted_as_anomaly'] == False]
    true_normals = normals.loc[predictions['predicted_as_anomaly'] == False]

    TA = len(true_anomalies)
    FA = len(false_anomalies)
    FN = len(false_normals)
    TN = len(true_normals)

    return TA, FA, FN, TN

def calc_f1(TA: int, FA: int, FN: int, TN: int) -> (float, float, float):
    if TA == 0:
        return 0, 0, 0
    else:
        precision = TA / (TA + FA)
        recall = TA / (TA + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
