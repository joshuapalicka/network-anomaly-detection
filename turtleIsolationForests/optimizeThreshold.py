from turtleIsolationForests.printResults import calc_f1
import numpy as np
import pandas as pd
import typing

def optimal_threshold_index(labels: np.ndarray[np.bool8], ordered_score_indices: np.ndarray[np.intc]) -> np.intc:
    TA = len(labels == True)
    FA = len(labels) - TA
    FN = 0
    TN = 0
    i = 0
    best_f1_so_far = 0.0
    index_of_best_f1 = 0
    while i < len(labels):
        if labels[ordered_score_indices[i]]:
            TA -= 1
            FN += 1
        else:
            FA -= 1
            TN += 1
        if TA + FA != 0:
            _, _, f1 = calc_f1(TA, FA, FN, TN)
            if f1 > best_f1_so_far:
                best_f1_so_far = f1
                index_of_best_f1 = ordered_score_indices[i]
        i += 1
    return index_of_best_f1
