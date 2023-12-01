import csv
import math
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = len(X[0])  # 計算特徵數量
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(n_samples_per_class)
        node = Node(predicted_class=predicted_class)

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        best_gini = float('inf')
        idx, thr = None, None
        for i in range(self.n_features_):  # 使用範圍內的索引值
            feature_values = [row[i] for row in X]  # 獲取第 i 個特徵的所有值
            thresholds, classes = zip(*sorted(zip(feature_values, y)))
            num_left = [0] * self.n_classes_
            num_right = list(classes)
            for j in range(1, len(y)):
                c = num_right[j - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / j) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (len(y) - j)) ** 2 for x in range(self.n_classes_))
                gini = (j * gini_left + (len(y) - j) * gini_right) / len(y)
                if thresholds[j] == thresholds[j - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    idx = i
                    thr = (thresholds[j] + thresholds[j - 1]) / 2.0
        return idx, thr

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class



# read csv file
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

# transfer data to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# 處理缺失值，這裡使用平均值填充
def handle_missing_values(dataset):
    for i in range(len(dataset[0])):
        # 檢查是否為數字，如果是，則轉換為浮點數
        if all(isinstance(row[i], (int, float)) for row in dataset):
            continue  # 若是數字類型，則跳過
        column_values = [float(row[i].strip()) for row in dataset if isinstance(row[i], str) and row[i].strip()]
        mean = sum(column_values) / len(column_values)
        for row in dataset:
            if isinstance(row[i], str) and not row[i].strip():
                row[i] = mean


# 根據需要進行其他的預處理步驟，如標準化、特徵轉換等等

# load train data
train_data = load_csv('../data/train.csv')
# remove header of train data
train_data = train_data[1:]
# transform to float data
for i in range(len(train_data[0])):
    str_column_to_float(train_data, i)
# 處理缺失值
handle_missing_values(train_data)

# 處load test data
test_data = load_csv('../data/test.csv')
# remove header of test data
test_data = test_data[1:]
# 轉換為浮點數
for i in range(len(test_data[0])):
    str_column_to_float(test_data, i)
# 處理缺失值
handle_missing_values(test_data)

# 分離特徵和目標類別
X_train = [row[:-1] for row in train_data]
y_train = [row[-1] for row in train_data]
X_test = [row[:-1] for row in test_data]
y_test = [row[-1] for row in test_data]


# 初始化 Decision Tree 模型
dt = DecisionTree(10)
# 訓練模型
dt.fit(X_train, y_train)

# # 進行預測
predictions = dt.predict(X_test)

# # 計算準確率
accuracy = sum(1 for i in range(len(y_test)) if y_test[i] == predictions[i]) / float(len(y_test))
print(f"準確率：{accuracy}")
