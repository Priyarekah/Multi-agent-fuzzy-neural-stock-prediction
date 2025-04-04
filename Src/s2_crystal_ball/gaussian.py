# for test
import sys, os 
sys.path.append(os.getcwd())
from s1_data_preparation.data_preparation import retrieve_data
from s2_crystal_ball.clustering import clusterCol
# from s2_crystal_ball.config import *

import math 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def Gauss(x, mean, std, cmax, cmin, height = 1): # to calculate Gaussian value according to cluster statistics 

    if (cmax - cmin) == 0: return 0
    
    return height*math.exp(-((x-mean)**2)/((2*std)**2))

def aggregateClusters(df, target): # obtain cluster statistics 
    
    clusters = df['cluster'].unique() 
    clusters.sort()
    
    cluster_details = {}
    count = 0

    for cluster in clusters:
        subdf = df[df['cluster']==cluster]

        cmean = subdf[target].mean() 
        cstd = subdf[target].std(ddof = 0)
        cmin = subdf[target].min() 
        cmax = subdf[target].max() 

        cdictionary = {
            'cluster': count, 
            'mean': cmean, 
            'std': cstd, 
            'min': cmin, 
            'max': cmax, 
            'minP': cmean - 3*cstd, 
            'maxP': cmean + 3*cstd
        }
        
        cluster_details[f'{target}_c{count}'] = cdictionary
        
        # update new cluster naming 
        df['cluster'] = df['cluster'].replace(cluster, count)
        count += 1
    
    return df, cluster_details

def gaussianMembership(x_val, cluster_details): # to calculate membership of each data point to individual clusters (crisp -> fuzzy)
    
    output = [0 for cluster in range(len(cluster_details))] 
    
    for cluster in cluster_details:
        # ref: the actual dictionary for the cluster 
        ref = cluster_details[cluster]
        # calculate list of outputs foreach cluster 
        output[ref['cluster']] = Gauss(x_val, ref['mean'], ref['std'], ref['max'], ref['min'])
    
    # summation = sum(output)
    # output = [item/summation for item in output]

    return output

def fuzzify(df, target): # section driver code 

    # obtain cluster statistics for Gaussian Distribution 
    df, cluster_details = aggregateClusters(df, target)
    
    # calculate Gaussian value of each data point according to cluster statistics 
    df['membership'] = df.apply(lambda x: gaussianMembership(x[target], cluster_details), axis = 1)
    
    # format membership of each data point to each cluster 
    df[[f'{target}_c{cluster}' for cluster in set(df['cluster'])]] = pd.DataFrame(df['membership'].tolist(), index = df.index)

    return df, cluster_details

def clusterMerge(df, target, cluster_details, cluster_config):  

    # if 'y_' in target: return df, cluster_details

    min_clusters = cluster_config['merged_min_clusters']
    max_clusters = cluster_config['merged_max_clusters']
    min_std = cluster_config['merged_min_std']
    maxy_clusters = cluster_config['merged_maxy_clusters']
    miny_std = cluster_config['merged_miny_std']

    outputdf = df.copy()

    while True:     
        cluster_set = set(outputdf['cluster'])
        
        # break condition 
        if len(cluster_set) <= min_clusters: break
            
        cluster_details = dict(sorted(cluster_details.items(), key=lambda x: x[1]["mean"])) # sort dictionary using mean of cluster

        # distance matrix 
        dist_left = [math.inf for index in range(len(cluster_details))] 
        dist_left_std = [math.inf for index in range(len(cluster_details))] 

        dist_right = [math.inf for index in range(len(cluster_details))] 
        dist_right_std = [math.inf for index in range(len(cluster_details))] 

        keys = list(cluster_details.keys())

        for index in range(len(cluster_details)):

            ref = cluster_details[keys[index]] 

            if index == 0: 
                dist_right_std[index] = (cluster_details[keys[index + 1]]['mean'] - ref['mean'])/ref['std']
                dist_right[index] = (cluster_details[keys[index + 1]]['mean'] - ref['mean'])

            elif index == len(cluster_details) - 1: 
                dist_left_std[index] = (ref['mean'] - cluster_details[keys[index-1]]['mean'])/ref['std']
                dist_left[index] = (ref['mean'] - cluster_details[keys[index-1]]['mean'])
            else: 
                dist_right_std[index] = (cluster_details[keys[index + 1]]['mean'] - ref['mean'])/ref['std']
                dist_right[index] = (cluster_details[keys[index + 1]]['mean'] - ref['mean'])

                dist_left_std[index] = (ref['mean'] - cluster_details[keys[index - 1]]['mean'])/ref['std']
                dist_left[index] = (ref['mean'] - cluster_details[keys[index - 1]]['mean'])

        # find minimum 
        refL = dist_left
        refR = dist_right
        minL = min(refL)
        minR = min(refR)

        # breakout condition if clusters distinct enough 
        if 'x_' in target: 
            if len(set(outputdf['cluster'])) < max_clusters and min(dist_left_std) > min_std and min(dist_right_std) > min_std: break 
        else: 
            if len(set(outputdf['cluster'])) < maxy_clusters and min(dist_left_std) > miny_std and min(dist_right_std) > miny_std: break 

        # identify replacement clusters 
        if minL <= minR: 
            index = refL.index(minL)
            cluster = cluster_details[keys[index]]['cluster']
            replacement_cluster = cluster_details[keys[index - 1]]['cluster'] # replacement cluster is the one to the left
        else: 
            index = refR.index(minR)
            cluster = cluster_details[keys[index]]['cluster']
            replacement_cluster = cluster_details[keys[index + 1]]['cluster'] # replacement cluster is the one to the right 

        # fuzzify again 
        outputdf = outputdf[[target, 'cluster']]
        outputdf['cluster'] = outputdf['cluster'].replace(cluster, replacement_cluster)
        outputdf, cluster_details = fuzzify(outputdf, target)   

    return outputdf, cluster_details

def fuzzifyTestset(df, target, cluster_details): 

    # calculate Gaussian value of each data point according to cluster statistics 
    df['membership'] = df.apply(lambda x: testsetGaussianMembership(x[target], target, cluster_details), axis = 1)        

    # format membership of each data point to each cluster 
    df[[f'{target}_c{cluster}' for cluster in range(len(cluster_details[target]))]] = pd.DataFrame(df['membership'].tolist(), index = df.index)

    return df 

def testsetGaussianMembership(x_val, target, cluster_details): # to calculate membership of each data point to individual clusters

    output = [0 for cluster in range(len(cluster_details[target]))] 
        
    for cluster in cluster_details[target]:
                
        ref = cluster_details[target][cluster]
        
        output[ref['cluster']] = Gauss(x_val, ref['mean'], ref['std'], ref['max'], ref['min'])

    return output

def initialClusterProcessing(outputdf, target, cluster_config): 

    initialization_min = cluster_config['initial_min_datapoints']

    cluster_agg = outputdf.groupby('cluster', as_index = True).agg(clusterCount = ('cluster','count'), clusterMean = (f'{target}', 'mean')).reset_index(drop = False)
    cluster_agg = cluster_agg[cluster_agg['clusterCount'] >= initialization_min]

    outputdf1 = outputdf.reset_index().merge(cluster_agg, on = 'cluster', how = 'inner').set_index('Date')
    outputdf1 = outputdf1.drop(['clusterCount', 'clusterMean'], axis = 1)
    
    outputdf1.sort_index(inplace=True, ascending=True)
    
    return outputdf1

def plotGaussian(cluster_details, plot_x = 1, plotrange = (-1, 5), size = (30, 7)):
    # plotting y variables 



    for target in cluster_details:

        extremes_min = [] 
        extremes_max = [] 

        for cluster in cluster_details[target]: 

            extremes_min.append(cluster_details[target][cluster]['minP'])
            extremes_max.append(cluster_details[target][cluster]['maxP'])
 
        # skip variables
        if plot_x == 1: 
            if 'y_' in target: continue
        else: 
            if 'x_' in target: continue
        
        df = pd.DataFrame()

        df[target] = np.arange(min(extremes_min)-0.2, max(extremes_max)+0.2, 0.001)
        
        # for cluster in cluster_details: 
        df = fuzzifyTestset(df, target, cluster_details).drop(['membership'], axis = 1)
            
        plt.figure(figsize=size)
        plt.title(f"{target}_Clustering")    
        
        for cluster in df.columns: 
            if cluster == target: continue 
            plt.plot(df[target], df[cluster], label = cluster)
        plt.legend(loc="upper left")

        plt.xlabel(f"{target} - % Change")

        plt.ylabel("Membership")

        plt.show()  
    return  

if __name__ == '__main__':

    trainset, valset, testset = retrieve_data("D05.SI")

    trainset = trainset[['y_Tp0_Change', 'y_Tp1_Change']]

    variable_cluster = {} 
    ftraindf = pd.DataFrame()

    # for trainset 
    for target in trainset.columns: 
        subdf = trainset[[target]]

        # GMM Clustering 
        model, clusters, outputdf, pred_proba = clusterCol(subdf)

        # Analyze for Gaussian Membership
        outputdf1, initial_cluster_details = fuzzify(outputdf, target) 

        # Merge Clusters
        outputdf2, variable_cluster[target] = clusterMerge(outputdf, pred_proba, target, initial_cluster_details)
            
        # format output df 
        if len(ftraindf) == 0: ftraindf = outputdf2.copy().drop(['cluster', 'membership', target], axis = 1)
        else: ftraindf = pd.concat([ftraindf, outputdf2], axis = 1).drop(['cluster', 'membership', target], axis = 1)
    
    # for testset 
        
    testset = testset[['y_Tp0_Change', 'y_Tp1_Change']]
    ftestdf = pd.DataFrame()

    for target in testset.columns: 
        subdf = testset[[target]]
        
        outputdf1 = fuzzifyTestset(subdf, target, variable_cluster) 
        
        if len(ftestdf) == 0: ftestdf = outputdf1.copy().drop(['membership', target], axis = 1)
        else: ftestdf = pd.concat([ftestdf, outputdf1], axis = 1).drop(['membership', target], axis = 1)

    print(ftestdf)
