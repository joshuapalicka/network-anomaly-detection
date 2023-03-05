from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree, c
from turtleIsolationForests.extendedIsolationForest import rng
from turtleIsolationForests.FBIF import HypersphereDecision
import math
import numpy as np
import pandas as pd
import random
import typing

class FBSCiTree(IsolationTree):

    def __init__(self, 
                 num_hyperspheres_per_split,        # the number of hyperspheres generated at each tree split, from which the best is chosen
                 data: pd.DataFrame, 
                 c1 = 1.0,                          # c1 and c2 are hyperparameters that control the range of values from which
                 c2 = 1.0):                         # hypersphere radii are chosen):
        super().__init__(data)
        self.num_hyperspheres_per_split = num_hyperspheres_per_split
        self.c1 = c1
        self.c2 = c2

class FBSCiForest(IsolationForest):

    def __init__(self,
                 num_hyperspheres_per_split,        # the number of hyperspheres generated at each tree split, from which the best is chosen
                 c1 = 1.0,                          # c1 and c2 are hyperparameters that control the range of values from which
                 c2 = 1.0,                          # hypersphere radii are chosen
                 contamination = 'auto',            # if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees = 100,                   # default forest size as presented in original paper.
                 subsample_size = 256,              # default subsample size as presented in original paper.
                 random_state = None):
        super().__init__(contamination, num_trees, subsample_size, random_state)
        self.num_hyperspheres_per_split = num_hyperspheres_per_split
        self.c1 = c1
        self.c2 = c2

    def _make_tree(self, sample_data: pd.DataFrame, depth = 0) -> IsolationTree:
        tree = FBSCiTree(sample_data, self.c1, self.c2)
        if len(sample_data) > 1 and depth < self.max_depth:
            tree.split(self.num_hyperspheres_per_split)
            left_indices = sample_data.apply(tree.decision.go_left, axis=1, result_type='reduce')
            left_data = sample_data.loc[left_indices]
            right_data = sample_data.loc[~left_indices]
            tree.left = self._make_tree(left_data, depth + 1)
            tree.right = self._make_tree(right_data, depth + 1)
        return tree
