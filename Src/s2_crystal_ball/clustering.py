
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# for test
import sys, os 
sys.path.append(os.getcwd())
from s1_data_preparation.data_preparation import retrieve_data

def clusterCol(X, cluster_dict): # to cluster datapoints 

    min_clusters = cluster_dict['min_clusters']
    max_clusters = cluster_dict['max_clusters']
    cluster_type = cluster_dict['cluster_type']
    factor = cluster_dict['factor']

    
    # to store outcomes for evaluation 
    models = [] 
    scores = [] 
    
    for index in range(min_clusters, max_clusters+1, 1):
        
        while True: 
            try:
                # KMeans Model 
                if cluster_type == 'kmeans': 
                    x_values = X[X.columns[0]]
                    initial_centroids = []

                    # identify appropriate statistics 
                    if 'x_' in X.columns[0]:
                        minimum = x_values.min() 
                        maximum = x_values.max() 
                    else: 
                        minimum = x_values.mean() - factor*x_values.std()
                        maximum = x_values.mean() + factor*x_values.std()
                    unit = (maximum - minimum) / (index - 1)
                    centre = minimum + unit 

                    # append initial clusters 
                    initial_centroids.append([minimum])
                    for index1 in range(index - 2):
                        initial_centroids.append([centre])
                        centre += unit 
                    initial_centroids.append([maximum])

                    model = KMeans(n_clusters = index, init = initial_centroids)
                    model.fit(X)
                    pred_cluster = model.predict(X)
                elif cluster_type == 'gmm': 
                    model = GaussianMixture(n_components = index)
                    model.fit(X)
                    pred_cluster = model.predict(X)

                # store the model 
                models.append(model)
                score = silhouette_score(X, pred_cluster)
                scores.append(score)

                break
            except: 
                continue
        
    # identify the best model
    maxscore = max(scores)
    index = scores.index(maxscore) 
    model = models[index]
    clusters = index + min_clusters

    # obtain predictions 
    pred_cluster = model.predict(X)
    
    # preparing results 
    X['cluster'] = pred_cluster     

    return model, clusters, X


if __name__ == '__main__': 
    traindf, testdf = retrieve_data()
    
    X = traindf[['MACDh_8_21_9']]
    print('here')
    model, cluster, X, pred_proba =  clusterCol(X)
    print('here2')
    print(X)


# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd

# import sys, os 
# sys.path.append(os.getcwd())
# from s1_data_preparation.data_preparation import retrieve_data

# def clusterCol(X, cluster_dict):  
#     """Clusters data using KMeans, GMM, or DBSCAN based on cluster_dict parameters."""

#     min_clusters = cluster_dict.get('min_clusters', 2)
#     max_clusters = cluster_dict.get('max_clusters', 10)
#     cluster_type = cluster_dict.get('cluster_type', 'kmeans')
#     factor = cluster_dict.get('factor', 1.0)

#     # Normalize data for DBSCAN compatibility
#     scaler = MinMaxScaler()
#     X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#     models = [] 
#     scores = [] 

#     # Handling DBSCAN separately
#     if cluster_type == 'dbscan':
#         eps = cluster_dict.get('eps', 0.3)  # Adjust eps based on normalized data
#         min_samples = cluster_dict.get('min_samples', 5)

#         # Apply DBSCAN
#         model = DBSCAN(eps=eps, min_samples=min_samples)
#         pred_cluster = model.fit_predict(X_scaled)  # Use scaled data for DBSCAN

#         # Compute number of clusters (excluding noise)
#         clusters = len(set(pred_cluster)) - (1 if -1 in pred_cluster else 0)

#         # Compute silhouette score only if there are at least 2 clusters
#         score = silhouette_score(X_scaled, pred_cluster) if clusters > 1 else None

#         X['cluster'] = pred_cluster
#         return model, clusters, X, score

#     # KMeans & GMM clustering
#     for index in range(min_clusters, max_clusters + 1):
#         try:
#             if cluster_type == 'kmeans': 
#                 x_values = X[X.columns[0]]
#                 initial_centroids = []

#                 # Identify range for centroids
#                 if 'x_' in X.columns[0]:
#                     minimum = x_values.min() 
#                     maximum = x_values.max() 
#                 else: 
#                     minimum = x_values.mean() - factor * x_values.std()
#                     maximum = x_values.mean() + factor * x_values.std()
                
#                 unit = (maximum - minimum) / (index - 1)
#                 centre = minimum + unit 

#                 # Append initial centroids
#                 initial_centroids.append([minimum])
#                 for _ in range(index - 2):
#                     initial_centroids.append([centre])
#                     centre += unit 
#                 initial_centroids.append([maximum])

#                 model = KMeans(n_clusters=index, init=initial_centroids, n_init=10)
#                 model.fit(X)
#                 pred_cluster = model.predict(X)
            
#             elif cluster_type == 'gmm': 
#                 model = GaussianMixture(n_components=index)
#                 model.fit(X)
#                 pred_cluster = model.predict(X)

#             # Store results
#             models.append(model)
#             score = silhouette_score(X, pred_cluster)
#             scores.append(score)

#         except: 
#             continue
    
#     # Identify best model based on silhouette score
#     maxscore = max(scores)
#     index = scores.index(maxscore) 
#     model = models[index]
#     clusters = index + min_clusters

#     # Obtain final cluster predictions
#     pred_cluster = model.predict(X)
#     X['cluster'] = pred_cluster     

#     return model, clusters, X, maxscore

