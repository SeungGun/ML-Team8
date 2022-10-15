# Import relevant libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

warnings.filterwarnings(action='ignore')


# Processing of missing data - Remove missing data
def processing_data(df):
    df = df.replace('?', np.nan)
    df = df.dropna(axis=0)
    df = df.astype(dtype='int64')

    return df


# Encoding target variable using label encoding
def label_encode(y):
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y)

    return encoded_y


# Scaling independent variables using robust scaling
def scaling_data(X_data):
    transformer = RobustScaler()
    transformer.fit(X_data)
    X_ = transformer.transform(X_data)

    return X_


# K-fold cross validation
def Kf_CV(X, y, trained_model, K):
    cv = KFold(n_splits=K, shuffle=True, random_state=42)
    model = trained_model.fit(X, y)
    accuracy = cross_val_score(model, X, y, scoring="accuracy", cv=cv)

    return accuracy


# Visualization
def visualizing_result(K3, K15):
    plt.figure(figsize=(12, 6))
    plt.title("Performance comparison", fontsize=30)
    labels = ['Tree_gini', 'Tree_entropy', 'Logistic', 'SVM']
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, K3, 0.4, color='blue', label='K = 3')
    plt.bar(X_axis + 0.2, K15, 0.4, color='red', label='K = 15')
    plt.xticks(X_axis, labels)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Mean Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


# Build a decision tree
def decision_tree(X, y, criterion, optimize=True):
    tree_model = DecisionTreeClassifier(criterion=criterion)
    if optimize:
        optimize_params = optimizing_params_decision_tree(tree_model, X, y)
        tree_model = DecisionTreeClassifier(criterion=criterion, max_depth=optimize_params['max_depth'],
                                            min_samples_leaf=optimize_params['min_samples_leaf'],
                                            max_features=optimize_params['max_features'],
                                            max_leaf_nodes=optimize_params['max_leaf_nodes'])
        tree_model.fit(X, y)
    else:
        tree_model.fit(X, y)

    return tree_model


# Build a logistic model
def logistic(X, y, optimize=True):
    logistic_model = LogisticRegression()
    encoded_y = label_encode(y)
    if optimize:
        optimize_params = optimizing_param_logistic(logistic_model, X, encoded_y)
        lr_model = LogisticRegression(penalty=optimize_params['penalty'], C=optimize_params['C'],
                                      random_state=optimize_params['random_state'], solver=optimize_params['solver'])
        lr_model.fit(X, encoded_y)
        return lr_model
    else:
        logistic_model.fit(X, encoded_y)

        return logistic_model


# Build a SVM model
def svm(X, y, optimize=True):
    svm_classifier = SVC(kernel='linear')
    if optimize:
        optimize_params = optimizing_params_svm(svm_classifier, X, y)
        svm_model = SVC(kernel=optimize_params['kernel'], C=optimize_params['C'], gamma=optimize_params['gamma'])
        svm_model.fit(X, y)
        return svm_model
    else:
        svm_classifier.fit(X, y)

        return svm_classifier


# Tuning parameters of decision tree
def optimizing_params_decision_tree(model, X, y):
    param_grid = {'max_depth': np.arange(1, 5),
                  'min_samples_leaf': np.arange(1, 5), 'max_features': np.arange(1, 5),
                  'max_leaf_nodes': np.arange(2, 5)}

    tree_gscv = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=3)
    tree_gscv.fit(X, y)

    return tree_gscv.best_params_


# Tuning parameters of logistic
def optimizing_param_logistic(model, X, y):
    params = {'penalty': ['l2', 'l1'],
              'C': [0.01, 0.1, 1, 3, 5, 10],
              'random_state': [0, 42],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
              }
    grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=3)
    grid_clf.fit(X, y)

    return grid_clf.best_params_


# Tuning parameters of svm
def optimizing_params_svm(model, X, y):
    param_grid = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'scale', 'auto']}
    svm_gscv = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=3)
    svm_gscv.fit(X, y)

    return svm_gscv.best_params_


# data: raw dataSet
# scaling: whether scale data
# param_tuning: whether tuning parameters
# algorithm: model name
# K: K for k-fold
def giant_function(X, y, scaling, param_tuning, algorithm, K):
    if scaling:  # if use scale, scaling data
        X = scaling_data(X)

    if algorithm == 'tree_gini':
        trained_model = decision_tree(X, y, 'gini', param_tuning)
        result = Kf_CV(X, y, trained_model, K)

    elif algorithm == 'tree_entropy':
        trained_model = decision_tree(X, y, 'entropy', param_tuning)
        result = Kf_CV(X, y, trained_model, K)

    elif algorithm == 'logistic_regression':
        trained_model = logistic(X, y, param_tuning)
        result = Kf_CV(X, y, trained_model, K)

    elif algorithm == 'svm':
        trained_model = svm(X, y, param_tuning)
        result = Kf_CV(X, y, trained_model, K)

    return result


## Main function
# Load the dataset
Raw_dataset = pd.read_csv('breast-cancer-wisconsin.data', header=None)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(Raw_dataset)

# Verifying missing data
# print(Raw_dataset.info())
# print()
# pd.set_option('display.max_columns', None)
# print(Raw_dataset.describe())  # In the seventh column, it can be seen that there is data that is not a number

data = processing_data(Raw_dataset)  # remove missing data
X = data.iloc[:, 1:10].values  # split X
y = data.iloc[:, 10].values  # split y

# call giant function
# Performance comparison before and after Scaling
result1 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='tree_gini', K=3)
result2 = giant_function(X, y, scaling=False, param_tuning=True, algorithm='tree_gini', K=3)
mean_result1 = np.average(result1)
mean_result2 = np.average(result2)

result3 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='tree_entropy', K=3)
result4 = giant_function(X, y, scaling=False, param_tuning=True, algorithm='tree_entropy', K=3)
mean_result3 = np.average(result3)
mean_result4 = np.average(result4)

result5 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='logistic_regression', K=3)
result6 = giant_function(X, y, scaling=False, param_tuning=True, algorithm='logistic_regression', K=3)
mean_result5 = np.average(result5)
mean_result6 = np.average(result6)

result7 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='svm', K=3)
result8 = giant_function(X, y, scaling=False, param_tuning=True, algorithm='svm', K=3)
mean_result7 = np.average(result7)
mean_result8 = np.average(result8)

print("Decision Tree_gini --> before scaling: {0} after scaling: {1}".format(round(mean_result1, 4),
                                                                             round(mean_result2, 4)))
print("Decision Tree_entropy --> before scaling: {0} after scaling: {1}".format(round(mean_result3, 4),
                                                                                round(mean_result4, 4)))
print("Logistic Regression --> before scaling: {0} after scaling: {1}".format(round(mean_result5, 4),
                                                                              round(mean_result6, 4)))
print("Support Vector Machine --> before scaling: {0} after scaling: {1}".format(round(mean_result7, 4),
                                                                                 round(mean_result8, 4)))

# K = 3
result1 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='tree_gini', K=3)
result2 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='tree_entropy', K=3)
result3 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='logistic_regression', K=3)
result4 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='svm', K=3)

avg1 = np.average(result1)
avg2 = np.average(result2)
avg3 = np.average(result3)
avg4 = np.average(result4)

k3_data = [avg1, avg2, avg3, avg4]

# K = 15
result5 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='tree_gini', K=15)
result6 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='tree_entropy', K=15)
result7 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='logistic_regression', K=15)
result8 = giant_function(X, y, scaling=True, param_tuning=True, algorithm='svm', K=15)

avg5 = np.average(result5)
avg6 = np.average(result6)
avg7 = np.average(result7)
avg8 = np.average(result8)

k15_data = [avg5, avg6, avg7, avg8]

visualizing_result(k3_data, k15_data)
