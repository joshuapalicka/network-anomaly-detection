from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree, c
from turtleIsolationForests.extendedIsolationForest import rng
import math
import numpy as np
import pandas as pd
import random
import typing

def euclidean_distance_sq(point1: np.ndarray, point2: np.ndarray) -> np.float64:
    diff = point1 - point2
    return np.dot(diff, diff)

class HypersphereDecision:

    def __init__(self, center: np.ndarray[np.float64], radius: np.float64):
        self.center = center
        self.radiusSq = radius * radius
    
    def go_left(self, point_of_interest: pd.Series) -> bool:
        return euclidean_distance_sq(np.array(point_of_interest), self.center) <= self.radiusSq

class FBIsolationTree(IsolationTree):

    def __init__(self, 
                 data: pd.DataFrame, 
                 c1 = 1.0,                          # c1 and c2 are hyperparameters that control the range of values from which
                 c2 = 1.0):                         # hypersphere radii are chosen):
        super().__init__(data)
        self.c1 = c1
        self.c2 = c2
    
    #override used by overridden FBIsolationForest._make_tree
    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.decision = self._decide_split()
    
    def _decide_split(self) -> HypersphereDecision:
        min_bounds = self.data.apply(min)
        max_bounds = self.data.apply(max)
        
        # paper center
        #center = rng.uniform(low=min_bounds, high=max_bounds)
        
        # the paper selects a center from the range of values, but I judge this biases the hyperspheres to center uniformly 
        # in hypercubic space and risks a sphere containing no data
        # Testing out using a random point as center - result is forest takes 3x as long to build, 30sec instead of 10sec,
        # but has massive improvement in scores. Up to more reasonable ~.6 f1, from ~.3
        center = self.data.sample(1, axis=0).to_numpy()[0]

        minVal = center - min_bounds
        maxVal = max_bounds - center

        # best guess at paper's poorly-explained rMin and rMax. Produces rMin of zero at every decision
        #rMin = self.c1 * min(maxVal.combine(minVal, min))
        #rMax = self.c2 * max(maxVal.combine(minVal, max))

        # New rMin, rMax norms the combined bounds rather than min/maxing them. This has superior scores.
        rMin = self.c1 * np.linalg.norm(maxVal.combine(minVal, min), ord=2)
        rMax = self.c2 * np.linalg.norm(maxVal.combine(minVal, max), ord=2)
        # Note: even small opposing changes in c1, c2 like 1.1,0.9 cause crash failures when rMin and rMax would be very close to each other
        # i.e. when the decision data contains few, close points. I also don't see much point in lowering c1 or increasing c2.
        # Possibly I have automated c1,c2 tuning by introducing the normed rMin, rMax calculation.
 
        radius = rng.uniform(low=rMin, high=rMax)
        return HypersphereDecision(center, radius)

class FBIsolationForest(IsolationForest):

    def __init__(self,
                 c1 = 1.0,                          # c1 and c2 are hyperparameters that control the range of values from which
                 c2 = 1.0,                          # hypersphere radii are chosen
                 contamination = 'auto',            #if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees = 100,                   #default forest size as presented in original paper.
                 subsample_size = 256,              #default subsample size as presented in original paper.
                 random_state = None):
        super().__init__(contamination, num_trees, subsample_size, random_state)
        self.c1 = c1
        self.c2 = c2

    def _make_tree(self, sample_data: pd.DataFrame, depth = 0) -> IsolationTree:
        tree = FBIsolationTree(sample_data, self.c1, self.c2)
        if len(sample_data) > 1 and depth < self.max_depth:
            tree.split()
            left_indices = sample_data.apply(tree.decision.go_left, axis=1, result_type='reduce')
            left_data = sample_data.loc[left_indices]
            right_data = sample_data.loc[~left_indices]
            tree.left = self._make_tree(left_data, depth + 1)
            tree.right = self._make_tree(right_data, depth + 1)
        return tree