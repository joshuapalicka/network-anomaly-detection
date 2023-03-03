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
    
    def project(self, vector: np.ndarray[np.float64]) -> np.float64:
        return np.dot(vector, self.vector)

class HyperplaneDecision:

    def __init__(self, hyperplane: Hyperplane, intercept: np.ndarray[np.float64]):
        self.hyperplane = hyperplane
        self.projected_intercept = self.hyperplane.project(intercept)
    
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
        return (self.decision.left_data, self.decision.right_data)
    
    #SCiForest generates multiple hyperplanes and chooses the best
    #They have a method for generating hyperplanes in their paper that does not require scaling in preprocessing
    #But since we will scale, I do not need to account for non-1 stdevs, and have simplified code accordingly.
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
        projected_data = hyperplane.projected_data
        projected_data.sort()
        left_stdevs, right_stdevs, all_stdev = self._split_stdevs_one_pass(projected_data)

        i = 0
        best_gain = -16
        best_index = -1
        while i < len(projected_data):
            avg_stdev = (left_stdevs[i] + right_stdevs[i]) / 2 # paper cites one-pass solution for stdev calculation in Knuth book.
            gain = (all_stdev - avg_stdev) / all_stdev
            if gain > best_gain:
                best_gain = gain
                best_index = i
            i += 1

        decision = HyperplaneDecision(hyperplane, hyperplane.projected_data[best_index])
        decision.gain = best_gain
        decision.left_data = None #how to preserve index information to efficiently assign left and right data?
        return decision
    
    # Calculates all left/right split standard deviations (and the global st.dev) in one pass over the data
    # The result is meaningful only if the projected data passed in is sorted
    # This is an implementation of Welford's online algorithm as described at https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def _split_stdevs_one_pass(self, projected_data: np.ndarray[np.float64]) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.float64]:
        len_data = len(projected_data)
        left_vars = np.zeros(len_data)
        left_means = np.zeros(len_data)
        right_vars = np.zeros(len_data)
        right_means = np.zeros(len_data)
        all_var = 0
        all_mean = 0
        i = 0
        while i < len_data:
            xi = projected_data[i] # alias the current datum for brevity
            i += 1 # it is convenient to increment i here since all further uses refer to the number of data seen so far

            # update global mean/variance
            prev_all_mean = all_mean
            all_mean += (xi - all_mean) / i
            all_var += (xi - prev_all_mean) * (xi - all_mean)

            # update split means/variances
            self._update_split_means(i, xi, left_means, left_vars, right_means, right_vars)
        
        # deferred divisions in the variances executed here for speed and floating point error reduction
        all_var /= len_data
        i = 0
        while i < len_data:
            left_vars[i] /= len_data
            right_vars[i] /= len_data
            i += 1
        
        # sqrt to get standard deviations
        all_stdev = np.sqrt(all_var)
        left_stdevs = np.sqrt(left_vars)
        right_stdevs = np.sqrt(right_vars)

        return left_stdevs, right_stdevs, all_stdev

    # I wonder if I can get numpy to parallelize this for me... gotta go fast
    def _update_split_means(self, i, xi, left_means, left_vars, right_means, right_vars):
        j = 0
        while j < len(left_means):
            self._update_split_mean(i, j, xi, left_means, left_vars, right_means, right_vars)
            j += 1
    
    def _update_split_mean(self, i, j, xi, left_means, left_vars, right_means, right_vars):
        if j < i: #split point itself goes to right statistics
            prev_left_mean_j = left_means[j]
            left_means[j] += (xi - left_means[j]) / i
            left_vars[j] += (xi - prev_left_mean_j) * (xi - left_means[j])
        else:
            prev_right_mean_j = right_means[j]
            right_means[j] += (xi - right_means[j]) / i
            right_vars[j] += (xi - prev_right_mean_j) * (xi - right_means[j])
    
    def _gain(self, decision: HyperplaneDecision) -> float:
        projected_data = decision.hyperplane.projected_data
        left_indices = decision.go_left_projected(projected_data)
        decision.left_data = self.data.loc[left_indices]
        decision.right_data = self.data.loc[~left_indices]
        projected_left = projected_data[left_indices]
        projected_right = projected_data[~left_indices]
        print('projected_data: ' + str(type(projected_data)) + str(projected_data.shape))
        print('projected_left: ' + str(type(projected_left)) + str(projected_left.shape))
        print('projected_right: ' + str(type(projected_right)) + str(projected_right.shape))
        return _gain(projected_data, projected_left, projected_right)
    
    def _gain(self, projected_data, projected_left, projected_right):
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
