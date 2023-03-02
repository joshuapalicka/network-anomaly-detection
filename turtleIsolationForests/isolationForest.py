from turtleIsolationForests.optimizeThreshold import optimize_threshold
import math
import numpy as np
from numpy.random import default_rng
import pandas as pd
import random
import typing

rng = default_rng()

class Decision:

    def __init__(self, attribute: str, attribute_index: int, value: float):
        self.attribute = attribute
        self.attribute_index = attribute_index
        self.value = value
    
    def go_left(self, point_of_interest: np.ndarray[np.float64]) -> bool:
        return (point_of_interest[self.attribute_index] < self.value)

class IsolationTree:

    def __init__(self, data: pd.DataFrame):
        self.data = data                            #the data at this node, before decision if interior node.
        self.size = len(data)                       #the amount of data at this node, cached to reduce prediction calculations
        self.left = None                            #left child Isolation Tree.
        self.right = None                           #right child Isolation Tree.
        self.decision = None                        #if interior node, the decision function (set by split), if leaf, None.
    
    def path_length(self, point: np.ndarray[np.float64]) -> float:
        tree_pointer = self
        path_length = 0
        while (tree_pointer.decision is not None):  #only leaves have None decision
            path_length += 1
            left = tree_pointer.decision.go_left(point)
            if (left):
                tree_pointer = tree_pointer.left
            else:
                tree_pointer = tree_pointer.right
        return path_length + c(tree_pointer.size)   #As in IF paper, returned path length is adjusted up by c(|tree_data|) at the terminal node
        
    def split(self) -> None:
        self.decision = self._decide_split()
        left_data = self.data[self.data[self.decision.attribute] < self.decision.value]
        right_data = self.data[self.data[self.decision.attribute] >= self.decision.value]
        self.left = IsolationTree(left_data)
        self.right = IsolationTree(right_data)
    
    def _decide_split(self) -> Decision:
        attribute_index = rng.integers(len(self.data.columns))
        attribute = self.data.columns[attribute_index]
        lower_bound = self.data[attribute].min()
        upper_bound = self.data[attribute].max()
        split_value = rng.uniform(lower_bound, upper_bound)
        return Decision(attribute, attribute_index, split_value)

class IsolationForest:

    def __init__(self,
                 contamination:any = 'auto',            #if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees:int = 100,                   #default forest size as presented in original paper.
                 subsample_size:int = 256,              #default subsample size as presented in original paper.
                 random_state:int = None,
                 verbose:bool = False):                  #when true prints out status messages
        self.contamination = contamination
        self.subsample_size = subsample_size
        self.max_depth = math.floor(math.log2(subsample_size))  #Maximum tree depth. After this depth, the average depth of all points, by definition points are not anomalous.
        self.num_trees = num_trees
        self.random_state = random_state
        self.verbose = verbose
        self.c = c(subsample_size)
    
    def _advance_random_state(self) -> None:        #have to change random_state to get different trees.
        if self.random_state is not None:           #but do so in a way that preserves replicability
            self.random_state *= 31                 #this is just multiply by a small prime, mod a bigger prime
            self.random_state %= 104729
    
    def _random_subsample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        self._advance_random_state()
        return dataframe.sample(n = self.subsample_size,
                                replace = False,
                                weights = None,
                                random_state = self.random_state)
    
    def fit(self, train_data: pd.DataFrame, train_labels: pd.DataFrame) -> None:
        self.forest = [self._make_tree(self._random_subsample(train_data)) for i in range(self.num_trees)]
        if (self.verbose):
            print("Finished building forest")
        self.threshold = self._calculate_anomaly_score_threshold(train_data, train_labels)
        if (self.verbose):
            print("Finished calculating threshold")

    def _make_tree(self, sample_data: pd.DataFrame) -> IsolationTree:
        point_to_isolate = self._random_point(sample_data)
        root = IsolationTree(sample_data)
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
    
    def _random_point(self, sample_data: pd.DataFrame) -> np.ndarray[np.float64]:
        self._advance_random_state()
        return sample_data.sample(n = 1, random_state = self.random_state).iloc[0].to_numpy()
    
    def _calculate_anomaly_score_threshold(self, train_data: pd.DataFrame, train_labels: pd.DataFrame) -> float:
        self.train_scores = self._score(train_data)
        if self.contamination == 'auto':
            self.train_scores['is_normal'] = train_labels
            self.train_scores.sort_values('anomaly_score', inplace=True, ignore_index=True)
            threshold = optimize_threshold(self.train_scores)   
        else:
            percentile_contamination = 100 * self._adapt_contamination(train_data)
            threshold = np.percentile(self.train_scores, 100 - percentile_contamination)        
                                                #this takes aaaaages. how is sklearn doing it so fast, and is it during fit or predict?
                                                #It's during fit
                                                #I bet it's parallelism through cython
        self.train_scores['predicted_as_anomaly'] = self.train_scores['anomaly_score'] > threshold
        return threshold
    
    def _adapt_contamination(self, data: pd.DataFrame) -> float:       #accept int contaminations, but work internally only with float contaminations in [0,1]
        if self.contamination is float:
            return min(1.0, max(0.0, self.contamination))
        elif self.contamination is int:
            return self.contamination / len(data)
        else:
            return 0.5

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        predictions = self._score(data)
        predictions['predicted_as_anomaly'] = predictions['anomaly_score'] > self.threshold
        return predictions
    
    def _score(self, data: pd.DataFrame) -> pd.DataFrame:
        scoreframe = pd.DataFrame(index=data.index)
        scoreframe['anomaly_score'] = self._calculate_anomaly_scores(data)
        return scoreframe
    
    def _calculate_anomaly_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        self.len_data = len(data)
        anomaly_scores = data.apply(self._calculate_anomaly_score, axis=1, raw=True, result_type='reduce')
        del self.len_data
        return anomaly_scores
    
    def _calculate_anomaly_score(self, point: np.ndarray[np.float64]) -> float:
        running_total = 0
        count = 0
        for tree in self.forest:
            running_total += tree.path_length(point)
            count += 1
        path_length = running_total / count
                                                    #Equation 2 from original Isolation Forest paper
        return 2 ** (-1 * path_length / self.c)

def c(n: int) -> float:                             #c(n) as defined in Isolation Forest Paper (2012) is the average path length
                                                    #of an unsuccessful BST search and is used to normalize anomaly scores
    if n > 2:
        return 2 * harmonic_number(n - 1) - (2 * (n-1) / n)
    elif n == 2:
        return 1
    else:
        return 0
    
def harmonic_number(n: int) -> float:               #this is an approximation
    #Euler-Mascheroni constant                      #see https://en.wikipedia.org/wiki/Harmonic_number
    gamma = 0.5772156649015328606065120900824024310421
    return math.log(n) + gamma
