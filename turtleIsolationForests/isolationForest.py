import pandas as pd
import math
import random

class IsolationForest:

    def __init__(self,                          
                 num_trees = 100,                   #default forest size as presented in original paper.
                 subsample_size = 256,              #default subsample size as presented in original paper.
                 random_state = None):
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
    
    def fit_tree(self, sample_data):
        point_to_isolate = self.random_point(sample_data)
        root = IsolationTree(sample_data)
        tree_pointer = root
        while (tree_pointer.depth < self.max_depth and len(tree_pointer.data) > 1):
            tree_pointer.split()
            if (tree_pointer.decision.go_left(point_to_isolate)):
                tree_pointer = tree_pointer.left
            else:
                tree_pointer = tree_pointer.right
        return root

    def predict(self, X_test):
        predictions = pd.DataFrame(index=X_test.index)
        predictions['anomaly_score'] = X_test.apply(self.predict_point, axis=1, result_type='reduce')
        return predictions
    
    def predict_point(self, point):
        predictions = [tree.predict(point) for tree in self.forest]
        return sum(predictions) / len(predictions)
        

class IsolationTree:

    def __init__(self,
                 data,
                 parent = None):
        self.data = data                            #the data at this node, before decision, if interior node.
        self.parent = None                          #parent Isolation Tree.
        self.left = None                            #left child Isolation Tree.
        self.right = None                           #right child Isolation Tree.
        self.decision = None                        #if interior node, the decision function, if leaf, None.
        if parent is None:
            self.depth = 0                          #with no parent, this tree is a root
        else:
            self.depth = parent.depth + 1           #depth of the tree, distance from root
        
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
    
    def predict(self, point):
        tree_pointer = self
        while (tree_pointer.decision is not None):  #None decision happens only at leaves
            if (tree_pointer.decision.go_left(point)):
                tree_pointer = tree_pointer.left
            else:
                tree_pointer = tree_pointer.right
        return tree_pointer.depth

class Decision:

    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
    
    def go_left(self, point_of_interest):
        return (point_of_interest[self.attribute] < self.value)
