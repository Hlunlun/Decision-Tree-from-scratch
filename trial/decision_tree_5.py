import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import math


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, X, y,curr_depth=0):

        num_samples, num_features = np.shape(X)
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(X, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, X, num_samples, num_features):
        ''' function to find the best split '''
        
        features = self.X.columns.values.tolist()

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        for feature in features:
            fature_values = X[feature]
            possible_thresholds = np.unique(fature_values)
            for threshold in possible_thresholds:
                X_left, X_right = self.split(X, feature, threshold)

                if len(X_left) > 0 and  len(X_right) > 0:
                    y_left, y_right =  





# load csv file
data = pd.read_csv('../data/train.csv')

# prepare for the data
y = data['fake']
X = data.drop(['fake'], axis = 1) # remove 'fake' column

print(np.unique(X['#follows'].iloc[0]))

# split test and train data: 0.8 train, 0.2 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 