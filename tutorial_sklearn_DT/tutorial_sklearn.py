
'''prepare for the data'''
import random
from collections import defaultdict

mushrooms = defaultdict(lambda: [])

def checkType(size, length):
    if size < 10: return 'money'
    if size < 15: return 'fortunate'
    if length > 15: return 'money'
    return 'mysterious'

for i in range(100):
    size = random.randint(1,20)
    length = random.randint(1, 20)
    mushrooms['type'].append(checkType(size,length))
    mushrooms['size'].append(size)
    mushrooms['length'].append(length)

'''mushrooms:
{'type': ['mysterious', 'fortunate', 'money', 'money', ...], 'size': [19, 10, 3, 8, 19,...], 'length': [12, 1, 8, 5,...]})
'''

import pandas as pd 
data = pd.DataFrame.from_dict(mushrooms)

'''data:
          type  size  length
0        money     9      13
1    fortunate    12       2
2        money     9      20
3   mysterious    16       8
4        money    18      19
..         ...   ...     ...
95       money     4       9
96   fortunate    11       7
97   fortunate    12      18
98       money     7      18
99   fortunate    11      16
'''

y = data['type'] # for 
'''y:
0          money
1          money
2          money
         ...
99    mysterious
'''
X = data.drop(['type'], axis = 1) # for , drop the feature of type
'''X:
    size  length
0      9      13
1      1       1
..   ...     ...
99    18       6
'''


'''learn from decision tree'''

# split test and train data
from sklearn.model_selection import train_test_split
# 80%訓練集(練習小考,會自己修正錯誤)、20%測試集(真的考試，測試他有沒有認真讀)，給model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 


# train by using decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

''' test the accuracy of the model using tes dataset'''
from sklearn.metrics import accuracy_score
y_predict = model.predict(X_test)
print(accuracy_score(y_test, y_predict))


''' test the accuracy of the model using new dataset '''
answers = [] # correct answers
tests = defaultdict(lambda: [])

for i in range(100):
    size = random.randint(1, 20)
    length = random.randint(1, 20)
    tests['size'].append(size)
    tests['length'].append(length)
    answers.append(checkType(size, length))

tests = pd.DataFrame.from_dict(tests)
predicts = model.predict(tests)
print(accuracy_score(answers, predicts))




''' Visualization '''

# graphviz
from sklearn.tree import export_graphviz
export_graphviz(model, out_file = 'mushrooms.dot', feature_names = ['size', 'length'], class_names = model.classes_, filled=True, rounded=True)


# contour
import matplotlib.pyplot as plt 
import numpy as np
fig = plt.figure(figsize=(12,6))
resolution = 100
# np.min/max: find proper value, resolution: how many points to plot
# linspace: 做節點分割
dx = np.linspace(np.min(1), np.max(20), resolution)
dy = np.linspace(np.min(1), np.max(20), resolution)
dx, dy = np.meshgrid(dx, dy)

# np.c_[x,y]: make a 2D array
# flatten(): convert 2D array to 1D array
Xc = np.c_[dx.flatten(), dy.flatten()]
z = model.predict(Xc)   

# convert predict to number
kls = list(model.classes_)

# z: depth of the contourf
z = np.array([kls.index(v) for v in z])
z = z.reshape(dx.shape)

contourf = plt.contourf(dx, dy, z, alpha=0.6)
plt.xlabel('size')
plt.ylabel('length')


# plt.legend(model.classes_, prop = { "size": 8 }) #,bbox_to_anchor=(1.1, 1)
''' model.classes_:
['fortunate' 'money' 'mysterious']
'''

plt.title("fortunate:0, money:1, mysterious:2")

plt.colorbar(contourf)
plt.show()




