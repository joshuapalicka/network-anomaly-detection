from turtleIsolationForests.isolationForest import IsolationForest, IsolationTree, c
from turtleIsolationForests.extendedIsolationForest import MultivariateDecision, rng
import math
import numpy as np
import pandas as pd
import random
import typing

class Projection:

    def __init__(self, hyperplane: MultivariateDecision):
        self.hyperplane = hyperplane
    
    def project_onto_hyperplane(self, vector: pd.Series) -> np.float64:
        # WHAT DO THE AUTHORS MEAN BY PROJECTING A VECTOR ONTO A HYPERPLANE AND ENDING UP WITH A REAL

        # this is the vector result of vector projection onto the hyperplane
        # projected = vector - (np.dot(vector, self.hyperplane.weights) / np.dot(self.hyperplane.weights, self.hyperplane.weights) * self.hyperplane.weights)
        
        # this is plugging the vector into the hyperplane equation as my best guess for what they did
        return np.dot(np.array(vector), self.hyperplane.weights) + self.hyperplane.bias

        # but perhaps they took the distance between the point and the plane? 
        # or the distance from the origin to the point's projection onto the hyperplane?
        # or the magnitude of the projected vector?

class SCIsolationTree(IsolationTree):
    
    #override used by overridden SCIsolationForest._make_tree
    def split(self, num_hyperplanes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.decision = self._decide_split(num_hyperplanes)
        return (self.decision.left_data, self.decision.right_data)
    
    #SCiForest generates multiple hyperplanes and chooses the best
    #They have a method for generating hyperplanes in their paper that does not require scaling in preprocessing
    #But since we will scale, I am borrowing code from EIF. They are otherwise the same.
    def _decide_split(self, num_hyperplanes: int) -> MultivariateDecision:
        hyperplane = MultivariateDecision(self._random_vector_on_unit_sphere(), self._random_intercept())
        num_hyperplanes -= 1
        while num_hyperplanes > 0:
            next_hyperplane = MultivariateDecision(self._random_vector_on_unit_sphere(), self._random_intercept())
            # currently generating random intercept
            # paper generates only random coeffs and tests every possible point as intercept
            # cites one-pass solution for stdev calculation in Knuth book.
            hyperplane = self._better_hyperplane(hyperplane, next_hyperplane)
            num_hyperplanes -= 1
        return hyperplane

    def _better_hyperplane(self, first: MultivariateDecision, second: MultivariateDecision) -> MultivariateDecision:
        gain1 = self._gain(first)
        gain2 = self._gain(second)
        if gain1 >= gain2:
            return first
        return second
    
    def _gain(self, hyperplane: MultivariateDecision) -> float:
        left_indices = self.data.apply(hyperplane.go_left, axis=1, result_type='reduce')
        hyperplane.left_data = self.data.loc[left_indices]
        hyperplane.right_data = self.data.loc[~left_indices]
        projection = Projection(hyperplane)
        projected_data = self.data.apply(projection.project_onto_hyperplane, axis=1, result_type='reduce') # a series
        projected_left = projected_data.loc[left_indices]                                                  # filtered series
        projected_right = projected_data.loc[~left_indices]                                                # filtered series
        sigma_y = np.std(projected_data)
        sigma_yl = np.std(projected_left)
        sigma_yr = np.std(projected_right)
        sigma_avg = (sigma_yl + sigma_yr) / 2
        gain = (sigma_y - sigma_avg) / sigma_y
        return gain

    def _random_vector_on_unit_sphere(self) -> np.ndarray[np.float64]:
        return rng.standard_normal(len(self.data.columns)) #for simplicity, not implementing num_attributes_per_hyperplane parameter
                                                           #This SCiForest will always use all columns for every hyperplane.
    
    def _random_intercept(self) -> np.ndarray[np.float64]:
        return rng.uniform(low=self.data.apply(min), high=self.data.apply(max))
    
    def _random_bounded_attribute_value(self, attribute: str) -> float:
        lower_bound = self.data[attribute].min()
        upper_bound = self.data[attribute].max()
        return random.uniform(lower_bound, upper_bound)

class SCIsolationForest(IsolationForest):

    def __init__(self,
                 num_hyperplanes_per_split,
                 contamination = 'auto',            #if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees = 100,                   #default forest size as presented in original paper.
                 subsample_size = 256,              #default subsample size as presented in original paper.
                 random_state = None):
        super().__init__(contamination, num_trees, subsample_size, random_state)
        self.num_hyperplanes_per_split = num_hyperplanes_per_split

    #SCIsolationForest splits fully, rather than isolating a random point.
    #The paper also doesn't use max_depth to truncate tree growth, but I have added it in for consistency with the IsolationForest base
    def _make_tree(self, sample_data: pd.DataFrame, depth: int = 0) -> SCIsolationTree:
        tree = SCIsolationTree(sample_data)
        if len(sample_data) > 2 and depth < self.max_depth:
            (left_data, right_data) = tree.split(self.num_hyperplanes_per_split)
            tree.left = self._make_tree(left_data, depth + 1)
            tree.right = self._make_tree(right_data, depth + 1)
        return tree
