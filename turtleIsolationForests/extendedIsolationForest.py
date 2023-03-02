from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree
import numpy as np
from numpy.random import default_rng
import pandas as pd
import random
import typing

rng = default_rng()

class MultivariateDecision:

    def __init__(self, vector: np.ndarray[np.float64], intercept: np.ndarray[np.float64]):
        self.weights = vector
        self.intercept = intercept
    
    def go_left(self, point_of_interest: np.ndarray[np.float64]) -> bool:
        return np.dot((point_of_interest - self.intercept), self.weights) <= 0

class ExtendedIsolationTree(IsolationTree):
    
    #override
    def split(self):
        self.decision = self._decide_split()
        left_indices = self.data.apply(self.decision.go_left, axis=1, raw=True, result_type='reduce')
        left_data = self.data.loc[left_indices]
        right_data = self.data.loc[~left_indices]
        self.left = IsolationTree(left_data)
        self.right = IsolationTree(right_data)

    #override
    def _decide_split(self) -> MultivariateDecision:
        return MultivariateDecision(self._random_vector_on_unit_sphere(), self._random_intercept())

    def _random_vector_on_unit_sphere(self) -> np.ndarray[np.float64]:
        return rng.standard_normal(len(self.data.columns))
    
    def _random_intercept(self) -> np.ndarray[np.float64]:
        #return [self._random_bounded_attribute_value(attribute) for attribute in list(self.data.columns)]
        return rng.uniform(low=self.data.apply(min), high=self.data.apply(max))
    
    def _random_bounded_attribute_value(self, attribute: str) -> float:
        lower_bound = self.data[attribute].min()
        upper_bound = self.data[attribute].max()
        return random.uniform(lower_bound, upper_bound)

class ExtendedIsolationForest(IsolationForest):

    def _make_tree(self, sample_data: pd.DataFrame) -> ExtendedIsolationTree:
        point_to_isolate = self._random_point(sample_data)
        root = ExtendedIsolationTree(sample_data)
        tree_pointer = root
        depth = 0
        while (depth < self.max_depth and len(tree_pointer.data) > 1):
            tree_pointer.split()
            if (tree_pointer.decision.go_left(point_to_isolate)):
                tree_pointer = tree_pointer.left
            else:
                tree_pointer = tree_pointer.right
            depth += 1
        return root
