from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree, c
import math
import numpy as np
from numba import prange, jit, int32, float64, void
import pandas as pd
import typing

rng = np.random.default_rng()

@jit(void(int32, int32, float64[:,:], float64[:,:]), parallel=True)
def _update_split_stats(i: int, xi: np.float64, split_means: np.ndarray[np.float64], split_vars: np.ndarray[np.float64]):
    limit = len(split_means)
    for j in prange(limit):
        if j < i: #split point itself goes to right statistics
            prev_mean_j = split_means[j, 0]
            split_means[j, 0] += (xi - split_means[j, 0]) / i
            split_vars[j, 0] += (xi - prev_mean_j) * (xi - split_means[j, 0])
        else:
            prev_mean_j = split_means[j, 1]
            split_means[j, 1] += (xi - split_means[j, 1]) / i
            split_vars[j, 1] += (xi - prev_mean_j) * (xi - split_means[j, 1])

# Calculates all left/right split standard deviations (and the global st.dev) in one pass over the data
# The result is meaningful only if the projected data passed in is sorted
# This is an implementation of Welford's online algorithm as described at https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def _split_stdevs_one_pass(projected_data: np.ndarray[np.float64], sorted_indices: np.ndarray[int]) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.float64]:
    len_data = len(projected_data)
    split_vars = np.zeros((len_data, 2), dtype=np.float64)
    split_means = np.zeros((len_data, 2), dtype=np.float64)
    all_var = 0
    all_mean = 0
    i = 0
    while i < len_data:
        xi = projected_data[sorted_indices[i]] # alias the current datum for brevity
        i += 1 # it is convenient to increment i here since all further uses refer to the number of data seen so far

        # update global mean/variance
        prev_all_mean = all_mean
        all_mean += (xi - all_mean) / i
        all_var += (xi - prev_all_mean) * (xi - all_mean)

        # update split means/variances
        _update_split_stats(i, xi, split_means, split_vars)
    
    # deferred divisions in the variances executed here for speed and floating point error reduction
    all_var /= len_data
    split_vars /= len_data
    
    # sqrt to get standard deviations
    all_stdev = np.sqrt(all_var)
    split_stdevs = np.sqrt(split_vars)

    return split_stdevs, all_stdev

class Hyperplane:

    def __init__(self, vector_coefs: np.ndarray[np.float64], column_indices: np.ndarray[np.float64], data: pd.DataFrame):
        self.vector_coefs = vector_coefs
        self.column_indices = column_indices
        self.projected_data = data.apply(self.project, axis=1, raw=True).to_numpy()
        projection_range = np.ptp(self.projected_data)
        self.lower_bound = -1 * projection_range
        self.upper_bound = projection_range
    
    def project(self, vector: np.ndarray[np.float64]) -> np.float64:
        return np.dot(vector[self.column_indices], self.vector_coefs)

class HyperplaneDecision:

    def __init__(self, hyperplane: Hyperplane, projected_intercept: np.float64):
        self.hyperplane = hyperplane
        self.projected_intercept = projected_intercept
    
    def go_left(self, point_of_interest: np.ndarray[np.float64]) -> bool:
        return self.go_left_projected(self.hyperplane.project(point_of_interest))
    
    def go_left_projected(self, projected_point_of_interest: np.float64) -> bool:
        return projected_point_of_interest < self.projected_intercept

class SCIsolationTree(IsolationTree):

    def __init__(self, data: pd.DataFrame, num_hyperplanes_per_split: int, num_attrs_per_split: int):
        super().__init__(data)
        self.num_hyperplanes_per_split = num_hyperplanes_per_split
        self.num_attrs_per_split = num_attrs_per_split
    
    #override used by overridden SCIsolationForest._make_tree
    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.decision = self._decide_split()
    
    #SCiForest generates multiple hyperplanes and chooses the best
    #They have a method for generating hyperplanes in their paper that does not require scaling in preprocessing
    #But since we will scale, I do not need to account for non-1 stdevs, and have simplified code accordingly.
    def _decide_split(self) -> HyperplaneDecision:
        best_decision = None
        best_gain = -16
        i = 0
        while i < self.num_hyperplanes_per_split:
            vector_coefs, column_indices = self._random_vector_on_unit_sphere()
            next_hyperplane = Hyperplane(vector_coefs, column_indices, self.data)
            next_decision = self._best_decision_for_hyperplane(next_hyperplane)
            next_gain = next_decision.gain
            if next_gain >= best_gain:
                best_gain = next_gain
                best_decision = next_decision
            i += 1
        return best_decision
    
    def _best_decision_for_hyperplane(self, hyperplane: Hyperplane) -> HyperplaneDecision:
        sorted_indices = np.argsort(hyperplane.projected_data)
        split_stdevs, all_stdev = _split_stdevs_one_pass(hyperplane.projected_data, sorted_indices)

        if all_stdev == 0: # case when all data are identical in relevant attributes, ex. if the attributes are all uncommon flags
            midpoint = len(sorted_indices) // 2
            decision = HyperplaneDecision(hyperplane, hyperplane.projected_data[sorted_indices[midpoint]])
            decision.gain = 0
            return decision

        i = 0
        best_gain = -16
        best_i = -1
        while i < len(sorted_indices):
            index = sorted_indices[i]
            avg_stdev = (split_stdevs[index, 0] + split_stdevs[index, 1]) / 2
            gain = (all_stdev - avg_stdev) / all_stdev
            if gain > best_gain:
                best_gain = gain
                best_i = i
            i += 1

        decision = HyperplaneDecision(hyperplane, hyperplane.projected_data[sorted_indices[best_i]])
        decision.gain = best_gain
        return decision

    def _random_vector_on_unit_sphere(self) -> np.ndarray[np.float64]:
        num_attributes = self.num_attrs_per_split
        total_attributes = len(self.data.columns)
        if num_attributes > total_attributes or num_attributes <= 0:
            raise AttributeError("num_attributes_per_split = " + str(num_attributes) + " is either nonpositive or larger than the number of columns")
        column_indices = rng.choice(total_attributes, num_attributes, replace=False)
        vector_coefs = rng.standard_normal(num_attributes)
        #vector = np.zeros(total_attributes)
        #vector[column_indices] = rng.standard_normal()
        #return vector
        return vector_coefs, column_indices
    
    # SCiForest adds a range boundary per-node outside of which a point's depth will not increment.
    def path_length(self, point: np.ndarray[np.float64]) -> float:
        tree_pointer = self
        path_length = 0
        while (tree_pointer.decision is not None):  #only leaves have None decision
            hyperplane = tree_pointer.decision.hyperplane
            projected_point = hyperplane.project(point)
            if projected_point >= hyperplane.lower_bound and projected_point <= hyperplane.upper_bound:
                path_length += 1
            if (tree_pointer.decision.go_left_projected(projected_point)):
                tree_pointer = tree_pointer.left
            else:
                tree_pointer = tree_pointer.right
        return path_length + c(tree_pointer.size)   #As in IF paper, returned path length is adjusted up by c(|tree_data|) at the terminal node

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
            tree.split()
            left_indices = tree.data.apply(tree.decision.go_left, axis=1, raw=True)
            left_data = tree.data.loc[left_indices]
            right_data = tree.data.loc[~left_indices]
            tree.left = self._make_tree(left_data, depth + 1)
            tree.right = self._make_tree(right_data, depth + 1)
        return tree
