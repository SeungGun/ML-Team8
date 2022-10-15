# Import relevant libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from pyclustering.cluster.clarans import clarans
from pyclustering.utils import timedcall
from sklearn.metrics import cluster

warnings.filterwarnings(action='ignore')


# Processing of the data
def processing_data(df):
    """Handle the missing data by filling mean value or dropping rows

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    df : DataFrame
         the preprocessed data filled mean value or dropped rows
    """
    # Processing of missing data - replacing with the average of the remaining values
    value_mean = (round(df['total_bedrooms'].mean()))
    df = df.fillna(value_mean)

    # Processing for unnecessary row - drop the row
    df = df.drop(['median_house_value'], axis=1)

    return df


# Label Encoding
def encoding_label(df):
    """Encode categorical data using Label Encoding

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    df : DataFrame
         the encoded data
    """
    encoder = LabelEncoder()
    df['ocean_proximity'] = encoder.fit_transform(df['ocean_proximity'])

    return df


# One hot Encoding
def encoding_oh(df):
    """Encode categorical data using One-Hot Encoding

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    oh_labels : DataFrame
         the encoded data
    """
    # Step1: Converts all characters to numeric
    encoder = LabelEncoder()
    encoder.fit(df['ocean_proximity'])
    labels = encoder.transform(df['ocean_proximity'])

    # Step2: Convert to 2D data
    labels = labels.reshape(-1, 1)

    # Step3: Apply One-Hot Encoding
    oh_encoder = OneHotEncoder()
    oh_encoder.fit(labels)
    oh_labels = oh_encoder.transform(labels)

    return oh_labels


# Ordinal Encoding
def encoding_ordinal(df):
    """Encode categorical data using Ordinal Encoding

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    df : DataFrame
         the encoded data
    """
    encoder = OrdinalEncoder()
    df['ocean_proximity'] = encoder.fit_transform(df['ocean_proximity'])
    return df


# Min Max Scaling
def min_max_scaling(df):
    """Scale numerical data using Min-Max Scaler

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    scaled_data : DataFrame
         the scaled data by min-max scaling
    """
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    return scaled_data


# Robust Scaling
def robust_scaling(df):
    """Scale numerical data using Robust Scaler

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    scaled_data : DataFrame
         the scaled data by robust scaling
    """
    scaler = RobustScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    return scaled_data


# Z-score Scaling
def z_score_scaling(df):
    """Scale numerical data using Standard(Z-Score) Scaler

    Parameters
    ----------
    df : DataFrame
         The entire original data set

    Returns
    ----------
    scaled_data : DataFrame
         the scaled data by standard(z-score) scaling
    """
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)

    return scaled_data


# Purity for quality measure
def purity_score(y_true, y_pred):
    """Measure quality of Clustering analysis using Purity

    Parameters
    ----------
    y_true : Series
             actual y values

    y_pred : Series
             predict y values

    Returns
    ----------
    purity : float
         The measure score by purity
    """
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# knee method for quality measure
def knee_method(X):
    """Measure quality of Clustering analysis using Knee Method
    And plot result of knee method

    Parameters
    ----------
    X : Series
        Independent variable
    """
    # Elbow/Knee Method
    plt.figure(figsize=(10, 5))
    nn = NearestNeighbors(n_neighbors=5).fit(X)
    distances, idx = nn.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()


# silhouette score for for quality measure
def sil_score(X, labels):
    """Measure quality of Clustering analysis using Silhouette score

    Parameters
    ----------
    X : Series
             The dataset

    labels : Series
             The labels of clusters

    Returns
    ----------
    silhouette : float
         The measure score by silhouette score
    """
    return silhouette_score(X, labels, metric='euclidean')


def model_clarans(data, K):
    """Build a CLARANS Clustering model and Plot Clusters

    Parameters
    ----------
    data : DataFrame
             The original dataset for clustering

    K : Int
        The number of clusters

    """
    clarans_instance = clarans(data.to_numpy(), K, 4, 3)
    (ticks, result) = timedcall(clarans_instance.process)

    clusters = clarans_instance.get_clusters()  # result clusters
    medoids = clarans_instance.get_medoids()  # result medoids

    colors = ['red', 'blue', 'green', 'black', 'gray', 'purple', 'brown', 'orange', 'cyan', 'magenta']

    for i in range(len(clusters)):
        cs = []
        for j in range(len(clusters[i])):
            cs.append(data.iloc[clusters[i][j]])
        newDf = pd.DataFrame(cs)
        plt.scatter(newDf['longitude'], newDf['latitude'], color=colors[i])

    d = []
    for i in range(len(medoids)):
        d.append(data.iloc[medoids[i]])

    newDf = pd.DataFrame(d)
    plt.scatter(newDf['longitude'], newDf['latitude'], color='yellow')
    plt.show()


def model_DBSCAN(X, K):
    """Build a DBSCAN Clustering model and Plot Clusters

    Parameters
    ----------
    X : DataFrame
        The original dataset for clustering

    K : Int
        The number of clusters

    """
    db = DBSCAN(eps=0.12, min_samples=K).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor=tuple(col),
            markersize=4,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "x",
            markerfacecolor=tuple(col),
            markeredgecolor=tuple(col),
            markersize=3,
        )

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()

    knee_method(X)


def model_K_Means(X, K):
    """Build a K-Means Clustering model and Plot Clusters

    Parameters
    ----------
    X : DataFrame
        The original dataset for clustering

    K : Int
        The number of clusters

    """
    kmeans = KMeans(n_clusters=K)
    model = kmeans.fit(X)
    labels = kmeans.labels_

    centers = model.cluster_centers_

    # Create cluster label
    X['K-means'] = kmeans.fit_predict(X)
    X['K-means'] = X['K-means'].astype("category")

    sns.set_style('whitegrid')
    sns.relplot(x='longitude', y='latitude', hue='K-means', data=X, kind='scatter')
    plt.show()
    print(sil_score(X, labels))


def model_GMM(X, K):
    """Build a EM(GMM) Clustering model and Plot Clusters

    Parameters
    ----------
    X : DataFrame
        The original dataset for clustering

    K : Int
        The number of clusters

    """
    gmm = GaussianMixture(n_components=K)
    gmm_label = gmm.fit_predict(X)
    X['gmm_label'] = gmm_label

    sns.set_style('whitegrid')
    sns.relplot(x='longitude', y='latitude', hue='gmm_label', data=X, kind='scatter')
    plt.show()


def AutoML(scaler_list, encoder_list, model_list, dataset):
    """Run in a all combinations for scalers, encoders, models

    Parameters
    ----------
    scaler_list : list
                  the list of name of scalers in a string

    encoder_list : list
                   the list of name of encoders in a string

    model_list : list
                 the list of name of models in a string

    dataset : DataFrame
              The original dataset
    """
    # handling missing data
    dataset = processing_data(dataset)

    # extract features
    features = ['longitude', 'latitude', 'median_income']

    # iterate combinations
    for K in range(2, 12, 2):
        # extract features
        select_df = dataset[features]
        for i in range(len(scaler_list)):
            if scaler_list[i] == 'Z-score':
                select_df = pd.DataFrame(z_score_scaling(select_df), columns=features)
            elif scaler_list[i] == 'Robust':
                select_df = pd.DataFrame(robust_scaling(select_df), columns=features)
            elif scaler_list[i] == 'Min_Max':
                select_df = pd.DataFrame(min_max_scaling(select_df), columns=features)
            for j in range(len(encoder_list)):
                if 'ocean_proximity' in select_df.columns:
                    if encoder_list[j] == 'One hot':
                        select_df = encoding_oh(select_df)
                    elif encoder_list[j] == 'Label':
                        select_df = encoding_label(select_df)
                    elif encoder_list[j] == 'Ordinal':
                        select_df = encoding_ordinal(select_df)
                for l in range(len(model_list)):
                    if model_list[l] == 'K-means':
                        model_K_Means(select_df, K)
                    elif model_list[l] == 'GMM':
                        model_GMM(select_df, K)
                    elif model_list[l] == 'CLARANS':
                        model_clarans(select_df, K)
                    elif model_list[l] == 'DBSCAN':
                        model_DBSCAN(select_df, K)


## Main function
# Load the dataset with name of columns
Raw_dataset = pd.read_csv('housing.csv')

scaler_list = ['Z-score', 'Robust', 'Min Max']  # scaler list
encoder_list = ['One hot', 'Label', 'Ordinal']  # encoder list
model_list = ['K-means', 'GMM', 'CLARANS', 'DBSCAN']  # model list

# Call auto machine learning function
AutoML(scaler_list, encoder_list, model_list, Raw_dataset)
