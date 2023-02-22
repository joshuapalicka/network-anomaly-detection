from pandas import DataFrame
import typing

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
    true_anomalies = anomalies.loc[predictions['predicted_as_anomaly'] == True]
    false_anomalies = anomalies.loc[predictions['predicted_as_anomaly'] == False]

    normals = predictions.loc[predictions['is_normal'] != 0]
    true_normals = normals.loc[predictions['predicted_as_anomaly'] == True]
    false_normals = normals.loc[predictions['predicted_as_anomaly'] == False]

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
