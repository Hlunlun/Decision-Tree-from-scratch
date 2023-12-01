import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pydotplus

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None, samples=None, class_=None):
        ''' constructor ''' 
        
        # node
        # feature, threshold that used to split the node 
        self.feature_index = feature_index
        self.threshold = threshold
        # children: left, right
        self.left = left
        self.right = right
        # information gain, samples number, value of different class: to show on the graph
        self.info_gain = info_gain        
        self.samples = samples
        self.value = value
        
        # leaf: the result of the decision tree  
        self.class_ = class_



class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2, features=None, classes_=None):
        ''' constructor '''
        
        # initialize the tree 
        self.root = None
        
        # set the stop criterion
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.features = features
        self.classes_ = classes_

    def fit(self, X, y):
        ''' function to fit model '''
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.grow_tree(dataset)   

    def grow_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, y = dataset[:,:-1], dataset[:,-1]
        n_samples, n_features = X.shape[0], X.shape[1]
        
        # split until stop criterion is met
        if n_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.find_best_split(dataset, n_samples, n_features)
            # check information gain > 0
            if best_split["info_gain"]>0:
                # recur left tree
                subtree_left = self.grow_tree(best_split["dataset_left"], curr_depth+1)
                # recur right tree
                subtree_right = self.grow_tree(best_split["dataset_right"], curr_depth+1)
                # return the node with best split
                return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"], 
                            left=subtree_left, right=subtree_right, info_gain=best_split["info_gain"],
                            value=best_split["value"],samples=best_split["samples"])
        
        # compute value of leaf: the value with the highest count of samples
        leaf_class_ = max(list(y), key=list(y).count)
        return Node(class_ = leaf_class_)
    
    def find_best_split(self, dataset, n_samples, n_features):
        ''' function to find the best split '''
        # initialize the best split, and the information gain
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop all features
        for feature_index in range(n_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            # loop all values of features 
            for threshold in thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs no null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, y_left, y_right = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # get current information gain
                    curr_info_gain = self.information_gain(y, y_left, y_right, "gini")
                    # update the best split if find better information gain: bigger gain -> better split
                    if curr_info_gain > max_info_gain:
                        best_split={
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain,
                            "value": list(Counter(list(y)).values()),
                            "samples": n_samples
                        }
                        max_info_gain = curr_info_gain
                        
        # return the best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, child_left, child_right, mode="entropy"):
        ''' function to compute information gain '''
        
        # compute the weight of childs
        weight_left = len(child_left) / len(parent)
        weight_right = len(child_right) / len(parent)

        # compute the information gain
        if mode=="gini":
            gain = self.gini(parent) - (weight_left*self.gini(child_left) + weight_right*self.gini(child_right))
        if mode == "entropy":
            gain = self.entropy(parent) - (weight_left*self.entropy(child_left) + weight_right*self.entropy(child_right))
       
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        # types of classes, in this case: [0, 1]
        classes_ = np.unique(y)
        entropy = 0

        # loop all classes and compute the entropy
        for class_ in classes_:
            p_class_ = len(y[y == class_]) / len(y)
            entropy += -p_class_ * np.log2(p_class_)
        return entropy
    
    def gini(self, y):
        ''' function to compute gini index '''
        # types of classes, in this case: [0, 1]
        classes_ = np.unique(y)
        gini = 0

        # loop all classes and compute the gini
        for class_ in classes_:
            p_class_ = len(y[y == class_]) / len(y)
            gini += p_class_**2
        return 1 - gini
        
    
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.class_ is not None:
            print(tree.class_)

        else:
            print(self.features[tree.feature_index]+ " <= ", tree.threshold, " ,info_gain=", tree.info_gain, ' ,value=', tree.value, ' ,samples=', tree.samples)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions 
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        # if there's only one point
        if tree.class_!=None: return tree.class_

        # start to classify by comparing the feature value to the threshold
        feature_threshold = x[tree.feature_index]
        if feature_threshold <=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def plot_tree_dots(self, tree=None):

        if not tree:
            tree = self.root        

        # initialize the the dot data
        dot_data = ['''digraph Tree{
        node [shape=box, style="rounded", color="black", fontname="helvetica"] ;
        edge [fontname="helvetica"] ;''']     
        
        # count the dot 
        i_node = 0
        def generate_dot_data(tree, curr_node=0):
            nonlocal i_node           
            
            if tree.class_ is not None:
                dot_data.append('%d [label="class: %s"];'%(curr_node, self.classes_[round(tree.class_)])) 
                # i_node += 1 
                return                 
            else:         
                # append the split information       
                dot_data.append('%d [label= "%s <= %f \n info_gain = %f \n value= %s \n samples= %d"];'
                    %(curr_node, self.features[tree.feature_index], tree.threshold, tree.info_gain, str(tree.value), tree.samples))
                # arrow for node that as children
                if tree.left is not None:
                    if curr_node == 0:
                        dot_data.append('%d -> %d[headlabel = True];'%(curr_node, i_node+1))
                    else:
                        dot_data.append('%d -> %d;'%(curr_node, i_node+1)) 
                    i_node += 1                
                generate_dot_data(tree.left, i_node)  
                if tree.right is not None:
                    if curr_node == 0:
                        dot_data.append('%d -> %d[headlabel = False];'%(curr_node, i_node+1))
                    else:
                        dot_data.append('%d -> %d;'%(curr_node, i_node+1)) 
                    i_node += 1
                generate_dot_data(tree.right, i_node)            

        generate_dot_data(tree)

        # add '}' at the end of the dot data
        dot_data.append('}')

        # transform the dot data to a string
        dot_datas = '\n'.join(dot_data)
        return dot_datas
        

def preprocess_data(data):

    # all properties
    data_properties = data.columns.values.tolist() 
    print(data_properties)

    # data shape
    print(data.shape)
    print(data.head())

    # check missing data
    missing_count = len(data) - data.count()
    print("Missing values count:\n", missing_count)
    # if missing, drop the data
    data = data.dropna()

if __name__ == '__main__':
    
    # load csv file
    data = pd.read_csv("data/train.csv")
    
    # get to know data
    preprocess_data(data)

    # split train and test data
    X = data.drop(['fake'], axis = 1).values
    y = data['fake'].values.reshape(-1,1)   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=39)

    # fit the model
    features = data.drop(['fake'], axis = 1).columns.values.tolist()
    decision_tree = DecisionTreeClassifier(min_samples_split=2, max_depth=3, features=features, classes_=['fake', 'real'])
    decision_tree.fit(X_train,y_train)
    
    # print the tree
    decision_tree.print_tree()
    dot_datas = decision_tree.plot_tree_dots()
    graph = pydotplus.graph_from_dot_data(dot_datas)
    # graph.write_dot("DT.dot")
    graph.write_png("DT.png")

    # test model
    y_predict = decision_tree.predict(X_test)
    y_predict = decision_tree.predict(X_test) 
    print(accuracy_score(y_test, y_predict))


    # test model using new datset
    data_new = pd.read_csv('data/test.csv')
    y_new = data_new['fake']
    X_new = data_new.drop(['fake'], axis = 1).values
    y_new_predict = decision_tree.predict(X_new)
    print(accuracy_score(y_new, y_new_predict))


    # accurayc -> random_state and the max_depth
    # accu = []
    # for j in range(50):
    #     for i in range(11):
    #         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=j)
    #         decision_tree = DecisionTreeClassifier(min_samples_split=2, max_depth=i, features=features, classes_=['fake', 'real'])
    #         decision_tree.fit(X_train,y_train)
    #         y_new_predict = decision_tree.predict(X_new)
    #         accu.append(accuracy_score(y_new, y_new_predict))      

    #     fig = plt.figure(figsize=(12,6))
    #     plt.plot(accu,'-')
    #     plt.xlabel('max_depth')
    #     plt.ylabel('accuracy')
    #     plt.legend(['Train','Valid'])
    #     plt.title('random_state = %d'%(j))
    #     plt.savefig('random_state_max_depth/random_state_%d.png'%(j))
    #     accu = []

    # accuracy ->  the min_samples_split and the max_depth
    # accu = []
    # for j in range(2,12):
    #     for i in range(2,12):            
    #         decision_tree = DecisionTreeClassifier(min_samples_split=j, max_depth=i, features=features, classes_=['fake', 'real'])
    #         decision_tree.fit(X_train,y_train)
    #         y_new_predict = decision_tree.predict(X_new)
    #         accu.append(accuracy_score(y_new, y_new_predict))
    #     fig = plt.figure(figsize=(12,6))
    #     plt.plot(accu,'-')
    #     plt.xlabel('max_depth')
    #     plt.ylabel('accuracy')
    #     plt.xlim(2)
    #     plt.legend(['Train','Valid'])
    #     plt.title('min_samples_split = %d'%(j))
    #     plt.savefig('min_samples_split_max_depth/min_samples_split_%d.png'%(j))
    #     accu = []




