from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree, c
from turtleIsolationForests.extendedIsolationForest import rng
import math
import numpy as np
import pandas as pd
import random
import typing

c1 = 1.0
c2 = 1.0 #These constants affect radius selection of HypersphereDecisions

def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> np.float64:
    return np.linalg.norm(point1 - point2, ord=2)

class HypersphereDecision:

    def __init__(self, center: np.ndarray[np.float64], radius: np.ndarray[np.float64]):
        self.center = center
        self.radius = radius
    
    def go_left(self, point_of_interest: pd.Series) -> bool:
        return euclidean_distance(np.array(point_of_interest), self.center) <= self.radius

class FBIsolationTree(IsolationTree):
    
    #override used by overridden FBIsolationForest._make_tree
    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.decision = self._decide_split()
    
    def _decide_split(self) -> HypersphereDecision:
        min_bounds = self.data.apply(min)
        max_bounds = self.data.apply(max)
        center = rng.uniform(low=min_bounds, high=max_bounds)

        minVal = center - min_bounds
        maxVal = max_bounds - center
        rMin = min(maxVal.combine(minVal, min))
        rMax = max(maxVal.combine(minVal, max)) 
        radius = rng.uniform(low=rMin, high=rMax)

        return HypersphereDecision(center, radius)

class FBIsolationForest(IsolationForest):

    def _make_tree(self, sample_data: pd.DataFrame, depth = 0) -> IsolationTree:
        tree = FBIsolationTree(sample_data)
        if len(sample_data) > 1 and depth < self.max_depth:
            tree.split()
            left_indices = sample_data.apply(tree.decision.go_left, axis=1, result_type='reduce')
            left_data = sample_data.loc[left_indices]
            right_data = sample_data.loc[~left_indices]
            tree.left = self._make_tree(left_data, depth + 1)
            tree.right = self._make_tree(right_data, depth + 1)
        return tree