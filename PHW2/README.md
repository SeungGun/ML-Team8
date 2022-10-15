# Machine Learning - Programming Homework #2

## Used Dataset
- California Housing Price Dataset
> Link: https://www.kaggle.com/datasets/camnugent/california-housing-prices

<br/>

## Functions
---
### 1. processing_data
- Handle the missing data by filling mean value of dropping rows
```python
def processing_data(df):
```
**parameters**
- df: The entire original data set

**return**
- df: the preprocessed data filled mean value or dropped rows

<br/>

### 2. Encoding
- Encode categorical data using below methods
1. Label Encoding
2. One-Hot Encoding
3. Ordinal Encoding

<br/>

```python
def encoding_label(df):
def encoding_oh(df):
def encoding_ordinal(df):
```
**parameters**
- df: The entire original data set

**return**
- df: the encoded data

<br/>

### 3. Scaling
- Scale numerical data using below methods
1. Min-Max Scaling
2. Robust Scaling
3. Z-Score Scaling
```python
def min_max_scaling(df):
def robust_scaling(df):
def z_score_scaling(df):

<br/>

```
**parameters**
- df: The entire original data set

**return**
- df: the scaled data

<br/>

### 4. Quality Measure
- Measure quality of Clustering analysis using below methods
1. Purity score**
```python
def purity_score(y_true, y_pred):
```
**parameters**
- y_true: actual y values
- y_pred: predict y values

**return**
- purity: The measure score by purity  

<br/>

**2. Knee Method**
```python
def knee_method(X):
```
**parameters**
- X: independent variable

<br/>

**3. Silhouette score**
```python
def sil_score(X, labels):
```
**parameters**
- X: The dataset
- labels: The labels of clusters

**return**
- silhouette: The measure score by silhouette score

<br/>

### 5. Make Model
- Build Clustering models and plot clusters
1. CLARANS
```python
def model_clarans(data, K):
```
**parameters**
- data: The original dataset for clustering
- K: The number of clusters

<br/>

2. K-Means
```python
def model_K_Means(X, K):
```
**parameters**
- X: The original dataset for clustering
- K: The number of clusters

<br/>

3. EM (GMM)
```python
def model_GMM(X, K):
```
**parameters**
- X: The original dataset for clustering
- K: The number of clusters

<br/>

4. DBSCAN
```python
def model_DBSCAN(X, K):
```
**parameters**
- X: The original dataset for clustering
- K: The number of clusters

<br/>

### 6. AutoML(Giant func.)
- Run in a all combinations for scalers, encoders, models
```python
def AutoML(scaler_list, encoder_list, model_list, dataset):
```
**parameters**
- scaler_list: the list of name of scalers in a string
- encoder_list: the list of name of encoders in a string
- model_list: the list of name of models in a string
- dataset: the original dataset

<br/>

---
## Flow
First, list the original dataset and the elements of each function, put them in AutoML, and turn them so that all the results are output at onece.

<br/>

---
## Members
| member | contribution | allocation |
| :---------------: | :--------: | :----: |
| 202035518 노형주| 33% | Processing data, K-Means, GMM, Min-Max Scaling, Label Encoding, One-hot Encoding, Silhouette score|
| 201935069 신승건| 33% | CLARANS, Robust scaling, Purity, Scikit-learn style comment | 
| 201736054 한만규| 33% | DBSCAN, Z-score scaling, Ordinal encoding, Knee method |
