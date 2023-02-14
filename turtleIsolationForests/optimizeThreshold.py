from turtleIsolationForests.printResults import return_results, calc_f1
import pandas as pd
import typing

def optimize_threshold(predictions: pd.DataFrame) -> float:
    TA = len(predictions.loc[predictions['is_normal'] == 0])
    FA = len(predictions.loc[predictions['is_normal'] != 0])
    FN = 0
    TN = 0
    limit = len(predictions)
    i = 0
    best_f1_so_far = 0.0
    index_of_best_f1 = 0
    while i < limit:
        if predictions.iloc[i]['is_normal'] == 0:
            TA -= 1
            FN += 1
        else:
            FA -= 1
            TN += 1
        _, _, f1 = calc_f1(TA, FA, FN, TN)
        if f1 > best_f1_so_far:
            best_f1_so_far = f1
            index_of_best_f1 = i
        i += 1
    return predictions.iloc[index_of_best_f1]['anomaly_score']
