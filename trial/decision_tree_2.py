
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import itertools



# load csv file
data = pd.read_csv('../data/train.csv')

# prepare for the data
y = data['fake']
X = data.drop(['fake'], axis = 1)

# split test and train data: 0.8 train, 0.2 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 

# gini越大，隨機性就越強，最大0.5, 最小0
def gini_impurity(y):
    ''' 
    Given a Pandas Series, compute the Gini impurity of the Series.
    y: variable wiht which calculates Gini impurity
    '''
    if isinstance(y, pd.Series):
        p = y.value_counts() / y.shape[0]
        gini = 1 - np.sum(p**2)
        return gini
    else:
        raise('Object must be a Pandas Series.')

# 熵越大，變量不確定性就越大
# Ex: 好瓜和壞括各站一半時，熵最大(1)；當所有瓜都是好瓜或是都是壞瓜時，熵最小(0)
def entropy(y):
    '''
    Given a Pandas Series, compute the entropy of the Series.
    y: variable wiht which calculates entropy
    '''
    if isinstance(y, pd.Series):
        p = y.value_counts() / y.shape[0]
        entropy = -np.sum(p * np.log2(p + 1e-9))
        return entropy
    else:
        raise('Object must be a Pandas Series.')


def variance(y):
    '''
    Function to help calculate the variance of avoiding nan 
    y: varable to calculate variance to. It should be a Pandas Series.
    '''
    if(len(y) == 1):
        return 0
    else:
        return y.var()

# if the information_gain is the highest, then it's the best split
def information_gain(y, mask, func=entropy):
    '''
    It returns the information Gain of a varaiable given a loss function.
    y: target variable
    mask: split choice
    func: function to be used to calculate the information gain in case of binary classification.
    '''
    a = sum(mask)
    b = mask.shape[0]

    if(a == 0 or b == 0):
        information_gain = 0
    else:
        if y.dtype != 'O':
            information_gain = variance(y) - (a/(a+b)) * variance(y[mask]) - (b/(a+b)) * variance(y[-mask])
        else:
            information_gain = func(y) - (a/(a+b)) * func(y[mask]) - (b/(a+b)) * func(y[-mask])
    
    return information_gain









def categorical_options(a):
    '''
    Creates all possible combinations of a Pandas Series.
    a: Pandas Series from where to get all possible combinations.
    '''
    a = a.unique()

    opciones = []
    for L in range(0, len(a)+1):
        for subset in itertools.combinations(a, L):
            subset = list(subset)
            opciones.append(subset)
    return opciones[1:-1]

def max_information_gain_split(x, y, func=entropy):
    '''
    Given a predictor & target variable, returns the best split, the error and the type of variable based on a selected cost function.
    x: predictor variable as Pandas Series.
    y: target variable as Pandas Series.
    func: function to be used to calculate the best split
    '''
    split_values = [] 
    info_gains = []

    numeric_variable = True if x.dtype!= 'O' else False

    # Create options according to variable type
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else:
        options = categorical_options(x)

    # Calculate information gains for all values of the options
    for value in options:
        mask = x < value if numeric_variable else x.isin(value) 
        val_info_gain = information_gain(y, mask, func)
        # Append the information gains to the list
        info_gains.append(val_info_gain)
        split_values.append(value)

    # Check if there are more than 1 results if not, return False
    if len(info_gains) == 0 :
        return(None, None, None, False)
    else:
        # Get results with highest information gains
        best_info_gain = max(info_gains)
        best_info_gain_index = info_gains.index(best_info_gain)    
        best_split = split_values[best_info_gain_index]
        return (best_info_gain, best_split, numeric_variable, True)

weight_info_gain, weight_split, _, _ = max_information_gain_split(data['nums/length username'], data['fake'])    
print('The best split for \"nums/length username\" is when the variable is less than', weight_split, 
'\n information gain for that split is:', weight_info_gain, '\n')
    
print(X.apply(max_information_gain_split, y=data['fake']))


def get_best_split(y, data):
    '''
    Given a data, select the best split and return the variable, the value, the variable type and the information gain.
    y: name of the target variable
    data: dataframe where to find the best split.
    '''

    masks = data.drop(y, axis=1).apply(max_information_gain_split, y=data[y])
    if sum(masks.loc[3, :]) > sum(masks.loc[2, :]):
        return(None, None, None, None)
    else:
        # Get only masks that can be splitted
        masks = masks.loc[:, masks.loc[3,: ]]

        # Get the results of split with highest Information Gain
        split_variable = masks.iloc[0].astype(np.float32).idxmax()
        # split_valid = masks[split_variable][]
        split_value = masks[split_variable][1]
        split_info_gain = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

        return (split_variable, split_value, split_info_gain, split_numeric)



def make_split(variable, value, data, is_numeric):
    '''
    Given a data and split conditions, do the split.
    variable: variable with which make the split.
    value: value of the variable to make the split.
    data: data to be splitted.
    is_numeric: boolean considering if the variable to be splitted is numeric or not.
    '''

    if is_numeric:
        data_1 = data[data[variable]<value]
        data_2 = data[(data[variable]<value) == False]
    else:
        data_1 = data[data[variable].isin(value)]
        data_2 = data[(data[variable].isin(value)) == False]
    
    return (data_1, data_2)


def make_prediction(data, target_factor):
    '''
    Given the target variable, make a prediction.
    data: pandas series for target varialbe
    target_factor: boolean considering if the variable is a factor or not
    '''

    # make predictions 
    if target_factor:
        pred = data.value_counts().idxmax()
    else:
        pred = data.mean()

    return pred
    


def train_tree(data,y, target_factor, max_depth = None,min_samples_split = None, 
                min_information_gain = 1e-20, counter=0, max_categories = 1000):
    
    '''
    Trains a Decission Tree
    data: Data to be used to train the Decission Tree
    y: target variable column name
    target_factor: boolean to consider if target variable is factor or numeric.
    max_depth: maximum depth to stop splitting.
    min_samples_split: minimum number of observations to make a split.
    min_information_gain: minimum information gain to consider a split to be valid.
    max_categories: maximum number of different values accepted for categorical values. 
                    High number of different values will slow down learning process.
    '''


    # check that max_categories is fulfilled
    if counter == 0:
        types = data.dtypes
        check_columns = types[types == "object"].index
        for column in data.columns:
            var_length = len(data[column].value_counts())
            if var_length > max_categories:
                raise ValueError('The varialbe ' + column + ' has ' + str(var_length) + 
                            ' unique values, which is more than the accepted ones: ' + str(max_categories))
        
    # check for depth conditions
    if max_depth == None:
        depth_cond = True
    else:
        if counter < max_depth:
            depth_cond = True
        else:
            depth_cond = False

    # check for sample conditions
    if min_samples_split == None:
        sample_cond = True
    else:
        if data.shape[0] < min_samples_split:
            sample_cond = True
        else: 
            sample_cond = False

    # check for information gain conditions
    if depth_cond & sample_cond:
        var, val, info_gain, var_type = get_best_split(y, data)

        # if information gain is fulfilled, make split
        if info_gain is not None and info_gain >= min_information_gain:
            counter += 1

            left, right = make_split(var, val, data, var_type)

            # instantiate sub-tree
            split_type = "<=" if var_type else "in"
            question = "{} {}  {}".format(var, split_type, val)

            subtree = {question: []}

            print(subtree)

            # find answers (recursion)
            yes_answer = train_tree(left, y, target_factor,
                max_depth, min_samples_split, min_information_gain, counter)

            no_answer = train_tree(right, y, target_factor,
                max_depth, min_samples_split, min_information_gain, counter)

            if yes_answer == no_answer:
                subtree = yes_answer
            else:
                subtree[question].append(yes_answer)
                subtree[question].append(no_answer)
        
        # if it doesn't match information gain, make prediction
        else:
            pred = make_prediction(data[y], target_factor)
            return pred

    # Drop dataset if doesn't match depth or sample conditions
    else:
        pred = make_prediction(data[y], target_factor)
        return pred
    return subtree
            

max_depth = 5
min_samples_split = 20
min_information_gain = 1e-5

decision_tree = train_tree(data, 'fake', True, max_depth, min_samples_split, min_information_gain)



def classify_datas(observation, tree):
    question = list(tree.keys())[0]

    if question.split()[1] == '<=':
        if observation[question.split()[0]] <=float(question.split()[2]):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if observation[question.split()[0]] in (question.split()[2]):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # if the answer is not a dictionary
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_datas(observation, answer)


# print(classify_datas(features, decision_tree))

# Visualization
features = ['profile pic', 'nums/length username', 'fullname words', 'nums/length fullname', 'name==username', 'description length',
            'external URL', 'private', '#posts', '#followers', '#follows']

# export_graphviz(decision_tree, out_file = 'DT_sklearn.dot', feature_names = features, class_names = ['fake', 'real'], filled=True, rounded=True)
# dot -Tpng DT_sklearn.dot -o DT_sklearn.png


