# Machine Learning - Programming Homework #1

## Used Dataset
- The Wisconsin Cancer Dataset
> Link: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

<br/>

## Functions
---
### 1. Preprocessing
- Remove Missing Data
```python
def processing_data(df):
```
**parameters**
- df: The entire original data set

**return**
- df: the preprocessed data filled mean value or dropped rows

<br/>

### 2. Encoding
- Encode categorical data using Label-Encoding
```python
def label_encode(y):
```
**parameters**
- y: the categorical data to encode

**return
- encoded_y: the encoded data

<br/>

### 3. Scaling
- Scale numerical data using Robust Scaling
```python
def scaling_data(X_data):
```
**parameters**
- X_data: the numerical data to scale

**return**
- X_: the scaled data

<br/>

### 4. Evaluation
- Evaluate model using K-fold cross validation
```python
def Kf_CV(X, y, trained_model, K):
```
**parameters**
- X: independent variables
- y: target variable
- trained_model: the trained model to evaluate
- K: the number of split

**return**
- accuracy: the accuracy of its model

### 5. Visualization
- Visualize evaluation of models using matplotlib
```python
def visualizing_result(K3, K15):
```
**parameters**
- K3: the result of evaluation when K is 3 in K-fold
- K15: the result of evaluation when K is 15 in K-fold

### 6. Model Building
- Build a models 
1. Decision Tree(gini, entropy)
```python
def decision_tree(X, y, criterion, optimize=True):
```
**parameters**
- X: independent variables
- y: target variable
- criterion: gini or entropy
- optimize: whether tune hyperparameter of decision tree

**return**
- tree_model: the trained decision tree model instance
<br/>

2. Logistic Regression
```python
def logistic(X, y, optimize=True):
```
**parameters**
- X: independent variables
- y: target variable
- optimize: whether tune hyperparameter of Logistic Regression

**return**
- logistic_model: the trained logistic model instance

<br/>

3. SVM
```python
def svm(X, y, optimize=True):
```
**parameters**
- X: independent variables
- y: target variable
- optimize: wheather tune hyperparameter of SVM

**return**
- svm_model: the trained svm model instance

<br/>

### 7. Parameter Tuning
- tuning parameters of each model using GridSearchCV
1. Decision Tree Parameter Tuning
```python
def optimizing_params_decision_tree(model, X, y):
```
**parameters**
- model: the base model instance of decision tree
- X: independent variables
- y: target variable

**return**
- tree_gscv.best_params: the dictionary of best parameters

<br/>

2. Logistic Regression Parameter Tuning
```python
def optimizing_param_logistic(model, X, y):
```
**parameters**
- model: the base model instance of logistic regression
- X: independent variables
- y: target variable

**return**
- grid_clf.best_params: the dictionary of best parameters

<br/>

3. SVM Model Parameter Tuning
```python
def optimizing_params_svm(model, X, y):
```
**parameters**
- model: the base model instance of SVM
- X: independent variables
- y: target variable

**return**
- svm_gscv.best_params: the dictionary of best parameters

<br/>

### 8. Giant Function
- Run in a all combinations for scalers, encoders, models
```python
def giant_function(X, y, scaling, param_tuning, algorithm, K):
```
**parameter**
- X: indenpendent variables of raw dataset
- y: target variable of raw dataset
- scaling: whether scale data
- param_tuning: whether tuning parameters
- algorithm: model name in a string
- K: the k value for k-fold

**return**
- result: evaluation result(accuracy) of selected model at parameter

<br/>

---
## Flow
1. Load the dataset
2. Verifying missing data
3. Call giant function

<br/>

---
## Members
| member | contribution | allocation |
| :---------------: | :--------: | :----: |
| 202035518 노형주| 33% | Preprocessing, Label encoding, Logising Regression, Parameter Tuning|
| 201935069 신승건| 33% | Decision Trees, Robust scaling, Giant Function, Parameter Tuning| 
| 201736054 한만규| 33% | SVM, Parameter Tuning, Visualization, Report|
