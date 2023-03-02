from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree, c
import math
import numpy as np
import pandas as pd
import random
import typing

rng = np.random.default_rng()

class Hyperplane:

    def __init__(self, vector: np.ndarray[np.float64], data: pd.DataFrame):
        self.vector = vector
        self.projected_data = data.apply(self.project, axis=1, raw=True).to_numpy()
        #print(self.projected_data)
        #print(type(self.projected_data))    
    
    def project(self, vector: np.ndarray[np.float64]) -> np.float64:
        #print("hyperplane vector shape: " + str(self.vector.shape))
        #print("input vector shape: " + str(vector.shape))
        res = np.dot(vector, self.vector)
        #print(res)
        #print(type(res))
        return res#np.dot(vector, self.vector)

class HyperplaneDecision:

    def __init__(self, hyperplane: Hyperplane, intercept: np.ndarray[np.float64]):
        self.hyperplane = hyperplane
        self.projected_intercept = self.hyperplane.project(intercept)
    
    def go_left(self, point_of_interest: np.ndarray[np.float64]) -> bool:
        return self.go_left_projected(self.hyperplane.project(point_of_interest))
    
    def go_left_projected(self, projected_point_of_interest: np.float64) -> bool:
        goleft = projected_point_of_interest < self.projected_intercept
        #print(projected_point_of_interest)
        #print(type(projected_point_of_interest))
        return goleft

class SCIsolationTree(IsolationTree):

    def __init__(self, data: pd.DataFrame, num_hyperplanes_per_split: int, num_attrs_per_split: int):
        super().__init__(data)
        self.num_hyperplanes_per_split = num_hyperplanes_per_split
        self.num_attrs_per_split = num_attrs_per_split
    
    #override used by overridden SCIsolationForest._make_tree
    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.decision = self._decide_split()
        return (self.decision.left_data, self.decision.right_data)
    
    #SCiForest generates multiple hyperplanes and chooses the best
    #They have a method for generating hyperplanes in their paper that does not require scaling in preprocessing
    #But since we will scale, I do not need to account for non-1 stdevs, and have simplified code accordingly.
    #
    # the authors generate tau hyperplanes
    # for each hyperplane, they generate coefficients and test all available points as possible split points
    # the split point that produces the highest gain is the best version of that hyperplane
    # the best version of the best of the tau planes is the chosen decision
    def _decide_split(self) -> HyperplaneDecision:
        hyperplane = Hyperplane(self._random_vector_on_unit_sphere(self.num_attrs_per_split), self.data)
        decision = self._best_decision_for_hyperplane(hyperplane)
        num_hyperplanes = self.num_hyperplanes_per_split - 1
        while num_hyperplanes > 0:
            next_hyperplane = Hyperplane(self._random_vector_on_unit_sphere(self.num_attrs_per_split), self.data)
            next_decision = self._best_decision_for_hyperplane(next_hyperplane)
            if next_decision.gain >= decision.gain:
                decision = next_decision
            num_hyperplanes -= 1
        return decision
    
    def _best_decision_for_hyperplane(self, hyperplane: Hyperplane) -> HyperplaneDecision:
        split = self.data.iloc[0].to_numpy()
        best_decision = HyperplaneDecision(hyperplane, split)
        best_decision.gain = self._gain(best_decision)
        best_gain = best_decision.gain
        i = 1
        while i < len(self.data.index):
            split = self.data.iloc[i].to_numpy()
            decision = HyperplaneDecision(hyperplane, split)
            decision.gain = self._gain(decision)
            if decision.gain > best_gain:
                best_gain = decision.gain
                best_decision = decision
            i += 1
        return best_decision
    
    def _gain(self, decision: HyperplaneDecision) -> float:
        projected_data = decision.hyperplane.projected_data
        #print(projected_data)
        #print(type(projected_data))
        #print(projected_data.shape)
        left_indices = decision.go_left_projected(projected_data)
        decision.left_data = self.data.loc[left_indices]
        decision.right_data = self.data.loc[~left_indices]
        projected_left = projected_data[left_indices]                                                  # filtered series
        projected_right = projected_data[~left_indices]                                                # filtered series
        sigma_y = np.std(projected_data)
        sigma_yl = np.std(projected_left)
        sigma_yr = np.std(projected_right)
        sigma_avg = (sigma_yl + sigma_yr) / 2 # paper cites one-pass solution for stdev calculation in Knuth book.
        gain = (sigma_y - sigma_avg) / sigma_y
        if math.isnan(gain):
            return -1
        else:
            return gain

    def _random_vector_on_unit_sphere(self, num_attributes: int) -> np.ndarray[np.float64]:
        total_attrs = len(self.data.columns)
        column_indices = rng.choice(total_attrs, num_attributes, replace=False)
        vector = np.zeros(total_attrs)
        for index in column_indices:
            vector[index] = rng.standard_normal(1)[0]
        return vector
        #return rng.standard_normal(len(self.data.columns)) #for simplicity, not implementing num_attributes_per_hyperplane parameter
                                                           #This SCiForest will always use all columns for every hyperplane.
    
    def _random_intercept(self) -> np.ndarray[np.float64]:
        return rng.uniform(low=self.data.apply(min), high=self.data.apply(max))

class SCIsolationForest(IsolationForest):

    def __init__(self,
                 num_hyperplanes_per_split,
                 num_attributes_per_split,
                 contamination = 'auto',            #if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees = 100,                   #default forest size as presented in original paper.
                 subsample_size = 256,              #default subsample size as presented in original paper.
                 random_state = None,
                 verbose = False):
        super().__init__(contamination, num_trees, subsample_size, random_state, verbose)
        self.num_hyperplanes_per_split = num_hyperplanes_per_split
        self.num_attributes_per_split = num_attributes_per_split

    def _make_tree(self, sample_data: pd.DataFrame, depth: int = 0) -> SCIsolationTree:
        tree = SCIsolationTree(sample_data, self.num_hyperplanes_per_split, self.num_attributes_per_split)
        if len(sample_data) > 2 and depth < self.max_depth:
            (left_data, right_data) = tree.split()
            tree.left = self._make_tree(left_data, depth + 1)
            tree.right = self._make_tree(right_data, depth + 1)
        return tree
