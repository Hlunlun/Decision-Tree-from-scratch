import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter 
import math

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, data, target, classes_, max_depth = 5, evaluationFunction=None):
        # max depth of the decision tree        
        self.max_depth = max_depth

        # classes of the decision tree
        self.classes_ = classes_

        # target of the decision tree
        self.target = target

        # prepare for the training data
        self.prepare_data(data)

        # initialize the model
        self.model = None

        # evaluation function: entropy or gini
        self.evaluationFunction = self.gini if evaluationFunction is None else evaluationFunction


    def prepare_data(self, data):
        '''
        Prepare the data for the Decision Tree.
        data: csv file
        '''
        
        # load the csv file
        self.data = pd.read_csv('data/train.csv')        
        
        # prepare for the data
        y = self.data[self.target]
        X = self.data.drop([self.target], axis = 1)

        # split test and train data: 0.8 train, 0.2 test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2) 

        # extract the features from the data
        self.features = self.X_train.columns.values.tolist()

    def fit(self):
        self.n_classes_ = len(set(self.y_train))        
        self.n_features_ = self.X_train.shape[1]        
        self.model = self._grow_tree()

    def gini(self):
                

        
        


    def _grow_tree(self, depth=5):
        '''
        Build Decision Tree.
        '''

        if len(self.data) == 0:
            return None


    def _best_split(self, X, y):
        '''
        Find the best split for the data.
        X: features
        y: target
        '''
        


    def visualization_og_data(self):
        sns.pairplot(self.data, hue='fake')
        plt.savefig('original_data_seaborn.png')

    def visualization_tree(self):
        export_graphviz(self.model, out_file = 'DT_sklearn.dot', feature_names = self.features, class_names = self.classes_ , filled=True, rounded=True)
        # dot -Tpng DT_sklearn.dot -o DT_sklearn.png


if __name__ == '__main__':

    tree = DecisionTree(data='../data/train.csv',
                        target='fake', 
                        classes_=['fake', 'real'],
                        max_depth = 100)
    # tree.visualization_og_data()
    tree.fit()

    tree.gini()
