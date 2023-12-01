import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import re
import dtreeviz 


# load csv file
data = pd.read_csv('../data/train.csv')

# data.rename(columns=lambda x: re.sub('[ /#]', '_',x) , inplace=True)
# headers = list(data.columns)
# print(headers)

# prepare for the data
y = data['fake']
X = data.drop(['fake'], axis = 1) # remove 'fake' column

# split test and train data: 0.8 train, 0.2 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

# decision tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# test the accuracy of the model using tes dataset
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))


# test the accuracy of the model using new dataset
data_new = pd.read_csv('../data/test.csv')
y_new = data_new['fake']
X = data_new.drop(['fake'], axis = 1)
y_new_predict = model.predict(X)
print(accuracy_score(y_new, y_new_predict))



# Visualization
features = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length',
            'external URL', 'private', '#posts', '#followers', '#follows']

export_graphviz(model, out_file = 'DT_sklearn.dot', feature_names = features, class_names = ['fake', 'real'], filled=True, rounded=True)
# dot -Tpng DT_sklearn.dot -o DT_sklearn.png


# print(model.info())


