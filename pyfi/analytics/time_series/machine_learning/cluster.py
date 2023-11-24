import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

import joblib


def run_cluster(df):

    print('***********RUNNING CLUSTER***********')


    cluster_df = df.pct_change().dropna(axis=0, how = 'all').T


    joblib.parallel_backend('threading', n_jobs=3)

    X_train = np.array(cluster_df) 
    X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train)

    n_clusters = 10
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", verbose=False)

    # Fit the clustering model
    kmeans.fit(X_train)

    # Access cluster labels
    cluster_labels = kmeans.labels_

    cluster_df = pd.DataFrame({'Symbol': cluster_df.index, 'Cluster': cluster_labels})

    cluster_df = cluster_df.sort_values(by='Cluster')
    
    # display(cluster_df.head())
    
    return cluster_df