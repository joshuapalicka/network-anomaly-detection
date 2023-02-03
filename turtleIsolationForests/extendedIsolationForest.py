from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree
import numpy as np
import pandas as pd
import random

class ExtendedIsolationForest(IsolationForest):

    def _make_tree(self, sample_data):
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

class ExtendedIsolationTree(IsolationTree):
    
    #override
    def split(self):
        self.decision = self._decide_split()
        left_indices = self.data.apply(self.decision.go_left, axis=1, result_type='reduce')
        left_data = self.data.loc[left_indices]
        right_data = self.data.loc[~left_indices]
        self.left = IsolationTree(left_data)
        self.right = IsolationTree(right_data)

    #override
    def _decide_split(self):
        return MultivariateDecision(self._random_vector_on_unit_sphere(), self._random_intercept())

    def _random_vector_on_unit_sphere(self): #note to self, EIF authors say this is an even dist of slopes ... but wouldn't that be an exponential distribution?
        return [random.gauss(mu = 0.0, sigma = 1.0) for _ in list(self.data.columns)]
    
    def _random_intercept(self):
        return [self._random_bounded_attribute_value(attribute) for attribute in list(self.data.columns)]
    
    def _random_bounded_attribute_value(self, attribute):
        lower_bound = self.data[attribute].min()
        upper_bound = self.data[attribute].max()
        return random.uniform(lower_bound, upper_bound)

class MultivariateDecision:

    def __init__(self, vector, intercept):
        self.weights = np.array(vector)
        self.bias = -1 * np.dot(np.array(intercept), self.weights)
    
    def go_left(self, point_of_interest):
        return (np.dot(np.array(point_of_interest), np.array(self.weights)) + self.bias) < 0
