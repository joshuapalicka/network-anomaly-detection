import math
import numpy as np
import pandas as pd
import random
import typing

class Decision:

    def __init__(self, attribute: list[float], value: list[float]):
        self.attribute = attribute
        self.value = value
    
    def go_left(self, point_of_interest: pd.Series) -> bool:
        return (point_of_interest[self.attribute] < self.value)

class IsolationTree:

    def __init__(self, data: pd.DataFrame):
        self.data = data                            #the data at this node, before decision if interior node.
        self.size = len(data)                       #the amount of data at this node, cached to reduce prediction calculations
        self.left = None                            #left child Isolation Tree.
        self.right = None                           #right child Isolation Tree.
        self.decision = None                        #if interior node, the decision function (set by split), if leaf, None.
    
    def path_length(self, point: pd.Series) -> float:
        tree_pointer = self
        path_length = 0
        while (tree_pointer.decision is not None):  #only leaves have None decision
            path_length += 1
            if (tree_pointer.decision.go_left(point)):
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
        attribute = random.sample(list(self.data.columns), 1)[0]
        lower_bound = self.data[attribute].min()
        upper_bound = self.data[attribute].max()
        split_value = random.uniform(lower_bound, upper_bound)
        return Decision(attribute, split_value)

class IsolationForest:

    def __init__(self,
                 contamination = 'auto',            #if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees = 100,                   #default forest size as presented in original paper.
                 subsample_size = 256,              #default subsample size as presented in original paper.
                 random_state = None):
        self.contamination = contamination
        self.subsample_size = subsample_size
        self.max_depth = math.floor(math.log2(subsample_size))  #Maximum tree depth. After this depth, the average depth of all points, by definition points are not anomalous.
        self.num_trees = num_trees
        self.random_state = random_state
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
    
    def fit(self, train_data: pd.DataFrame) -> None:                      #the whole dataset, as a dataframe, processed and suitable for training
        self.forest = [self._make_tree(self._random_subsample(train_data)) for i in range(self.num_trees)]
        self.threshold = self._calculate_anomaly_score_threshold(train_data)

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
    
    def _random_point(self, sample_data: pd.DataFrame) -> pd.Series:
        self._advance_random_state()
        return sample_data.sample(n = 1, random_state = self.random_state).iloc[0]
    
    def _calculate_anomaly_score_threshold(self, train_data: pd.DataFrame) -> float:
        if self.contamination == 'auto':
            return 0.5
        else:
            percentile_contamination = 100 * self._adapt_contamination(train_data)
            return np.percentile(self._calculate_anomaly_scores(train_data), 100 - percentile_contamination)        
                                                            #this takes aaaaages. how is sklearn doing it so fast, and is it during fit or predict?
                                                            #It's during fit
                                                            #I bet it's parallelism through cython
    
    def _adapt_contamination(self, data: pd.DataFrame) -> float:       #accept int contaminations, but work internally only with float contaminations in [0,1]
        if self.contamination == 'auto':
            return 0.5
        elif self.contamination is int:
            return self.contamination / len(data)
        else:
            return min(1.0, max(0.0, self.contamination))

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
        anomaly_scores = data.apply(self._calculate_anomaly_score, axis=1, result_type='reduce')
        del self.len_data
        return anomaly_scores
    
    def _calculate_anomaly_score(self, point: pd.Series) -> float:
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
