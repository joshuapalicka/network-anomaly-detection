from heapq import nsmallest
import math
import pandas as pd
import random

class IsolationForest:

    def __init__(self,
                 contamination,                     #if integer, the number of expected anomalies. If float [0,1], the proportion of expected anomalies.
                 num_trees = 100,                   #default forest size as presented in original paper.
                 subsample_size = 256,              #default subsample size as presented in original paper.
                 random_state = None):
        self.contamination = contamination
        self.subsample_size = subsample_size
        self.max_depth = math.log2(subsample_size)  #Maximum tree depth. After this depth, which is the average depth of all points, by definition points are not anomalous.
        self.num_trees = num_trees
        self.random_state = random_state
        self.forest = list()
    
    def advance_random_state(self):                 #have to change random_state to get different trees.
        if self.random_state is not None:           #but do so in a way that preserves replicability
            self.random_state *= 31                 #this is just multiply by a small prime, mod a bigger prime
            self.random_state %= 104729
    
    def random_subsample(self, dataframe):
        self.advance_random_state()
        return dataframe.sample(n = self.subsample_size,
                                replace = False,
                                weights = None,
                                random_state = self.random_state)
    
    def random_point(self, sample_data):
        self.advance_random_state()
        return sample_data.sample(n = 1, random_state = self.random_state).iloc[0]
    
    def fit(self, train_data):                      #the whole dataset, as a dataframe, processed and suitable for training
        self.forest = [self.fit_tree(self.random_subsample(train_data)) for i in range(self.num_trees)]
        self.threshold = self.calculate_anomaly_score_threshold(train_data)

    def fit_tree(self, sample_data):
        point_to_isolate = self.random_point(sample_data)
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
    
    def calculate_anomaly_score_threshold(self, train_data):
        training_predictions = self.calculate_anomaly_scores(train_data)        #this takes aaaaages. how is sklearn doing it so fast, and do it during fit or predict?
        contamination = self.normalize_contamination(train_data)
        threshold_index = round(contamination * len(train_data))
        if contamination <= 0.2:
            threshold = nsmallest(threshold_index, training_predictions)[threshold_index]
        else:
            threshold = sorted(training_predictions)[threshold_index]
        return threshold

    def predict(self, X_test):
        predictions = pd.DataFrame(index=X_test.index)
        predictions['anomaly_score'] = self.calculate_anomaly_scores(X_test)
        predictions['predicted_as_anomaly'] = predictions['anomaly_score'] > self.threshold
        return predictions
    
    def calculate_anomaly_scores(self, data):
        self.len_data = len(data)
        anomaly_scores = data.apply(self.calculate_anomaly_score, axis=1, result_type='reduce')
        del self.len_data
        return anomaly_scores
    
    def calculate_anomaly_score(self, point):
        running_total = 0
        count = 0
        for tree in self.forest:
            running_total += tree.path_length(point)
            count += 1
        path_length = running_total / count
        return 2 ** (-1 * path_length / c(self.len_data))
    
    def normalize_contamination(self, data):        #accept int contaminations, but work internally only with float contaminations in [0,1]
        if self.contamination is int:
            return self.contamination / len(data)
        else:
            return min(1.0, max(0.0, self.contamination))
        

class IsolationTree:

    def __init__(self,
                 data,
                 parent = None):
        self.data = data                            #the data at this node, before decision if interior node.
        self.size = len(data)                       #the amount of data at this node, cached to reduce prediction calculations
        self.parent = None                          #parent Isolation Tree.
        self.left = None                            #left child Isolation Tree.
        self.right = None                           #right child Isolation Tree.
        self.decision = None                        #if interior node, the decision function, if leaf, None.
        
    def split(self):
        attribute = random.sample(list(self.data.columns), 1)[0]
        lower_bound = self.data[attribute].min()
        upper_bound = self.data[attribute].max()
        split_value = random.uniform(lower_bound, upper_bound)
        self.decision = Decision(attribute, split_value)
        left_data = self.data[self.data[attribute] < split_value]
        right_data = self.data[self.data[attribute] >= split_value]
        self.left = IsolationTree(left_data, parent = self)
        self.right = IsolationTree(right_data, parent = self)
    
    def path_length(self, point):
        tree_pointer = self
        path_length = 0
        while (tree_pointer.decision is not None):  #only leaves have None decision
            path_length += 1
            if (tree_pointer.decision.go_left(point)):
                tree_pointer = tree_pointer.left
            else:
                tree_pointer = tree_pointer.right
        return path_length + c(tree_pointer.size) #As in IF paper, returned path length is adjusted up by c(|tree_data|)

class Decision:

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
    
    def go_left(self, point_of_interest):
        return (point_of_interest[self.attribute] < self.value)

def c(n):                                           #c(n) as defined in Isolation Forest Paper is the average path length
                                                    #of an unsuccessful BST search and is used to normalize anomaly scores
    if n > 2:
        return 2 * harmonic_number(n - 1) - (2 * (n-1) / n)
    elif n == 2:
        return 1
    else:
        return 0
    
def harmonic_number(n):                             #this is an approximation that is valid for large n
    #Euler-Mascheroni constant                      #see https://en.wikipedia.org/wiki/Harmonic_number
    gamma = 0.5772156649015328606065120900824024310421
    return math.log(n) + gamma
