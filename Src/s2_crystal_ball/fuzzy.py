
import sys, os 
sys.path.append(os.getcwd())
from s2_crystal_ball.clustering import clusterCol
from s2_crystal_ball.gaussian import *
from tqdm import tqdm

def fuzzifyTrainTest(trainset, valset, testset, cluster_dict, tsk = 0):
    variable_cluster = {} 
    ftraindf = pd.DataFrame()

    # transform_col = [col for col in trainset.columns if 'refPrice_' not in col or 'yref_' not in col or '_Date' not in col]
    skip_col = [col for col in trainset.columns if 'refPrice_' in col or 'yref_' in col or 'Date' in col]

    # for trainset 
    print('fuzzifying trainset')
    for target in tqdm(trainset.columns): 
        
        subdf = trainset[[target]]
        oridf = subdf.copy()        
        
        if target in skip_col: 
            ftraindf = pd.concat([ftraindf, subdf], axis = 1)
            continue  
        # # Call clustering function
        # if cluster_dict['cluster_type'] == 'dbscan':
        #     model, clusters, outputdf, score = clusterCol(subdf, cluster_dict)  # DBSCAN returns 4 values
        # else:
        model, clusters, outputdf = clusterCol(subdf, cluster_dict)  # KMeans/GMM return 3 values

        # process dataframe for outliers 
        outputdf = initialClusterProcessing(outputdf, target, cluster_dict)

        # Analyze for Gaussian Membership
        outputdf1, initial_cluster_details = fuzzify(outputdf, target) 

        # Merge Clusters
        outputdf2, variable_cluster[target] = clusterMerge(outputdf, target, initial_cluster_details, cluster_dict['merger_details'])
                
        # format output df (for predictor columns) 
        if tsk == 1: 
            if 'x_' in target: 
                if len(ftraindf) == 0: ftraindf = outputdf2.copy().drop(['cluster', 'membership', target], axis = 1)
                else: ftraindf = pd.concat([ftraindf, outputdf2], axis = 1).drop(['cluster', 'membership', target], axis = 1)
            else: 
                if len(ftraindf) == 0: ftraindf = oridf.copy()
                else: ftraindf = pd.concat([ftraindf, oridf], axis = 1)   

        else: 
            if len(ftraindf) == 0: ftraindf = outputdf2.copy().drop(['cluster', 'membership'], axis = 1)
            else: ftraindf = pd.concat([ftraindf, outputdf2], axis = 1).drop(['cluster', 'membership'], axis = 1)            
            if 'y_' in target:
                ftraindf = ftraindf.rename(columns={target: target.replace('y_', 'ypcref_')})
            else:
                if target in ftraindf.columns:  # Avoid KeyError
                    ftraindf = ftraindf.drop([target], axis=1)

    # provide 0 membership to outliers 
    ftraindf.fillna(0, inplace = True)

    # for testset 
    ftestdf = pd.DataFrame()
    
    print('fuzzifying testset')
    for target in tqdm(testset.columns): 
        
        subdf = testset[[target]]        
        
        if target in skip_col: 
            ftestdf = pd.concat([ftestdf, subdf], axis = 1)
            continue         

        if tsk == 1: 
            if 'x_' in target: 
                outputdf1 = fuzzifyTestset(subdf, target, variable_cluster) 
                if len(ftestdf) == 0: ftestdf = outputdf1.copy().drop(['membership'], axis = 1)
                else: ftestdf = pd.concat([ftestdf, outputdf1], axis = 1).drop(['membership'], axis = 1)
            else: 
                if len(ftestdf) == 0: ftestdf = subdf.copy()
                else: ftestdf = pd.concat([ftestdf, subdf], axis = 1) 
        else: 
            outputdf1 = fuzzifyTestset(subdf, target, variable_cluster) 
            if len(ftestdf) == 0: ftestdf = outputdf1.copy().drop(['membership'], axis = 1)
            else: ftestdf = pd.concat([ftestdf, outputdf1], axis = 1).drop(['membership'], axis = 1)  
            if 'y_' in target: ftestdf = ftestdf.rename(columns = {target: target.replace('y_', 'ypcref_')})
            else: ftestdf = ftestdf.drop([target], axis = 1)
    try:
        if len(valset) == 0: fvaldf = None
        else: 
            # for valset 
            fvaldf = pd.DataFrame()
            
            print('fuzzifying valset')
            for target in tqdm(valset.columns): 
                
                subdf = valset[[target]]        
                
                if target in skip_col: 
                    fvaldf = pd.concat([fvaldf, subdf], axis = 1)
                    continue         

                if tsk == 1: 
                    if 'x_' in target: 
                        outputdf1 = fuzzifyTestset(subdf, target, variable_cluster) 
                        if len(fvaldf) == 0: fvaldf = outputdf1.copy().drop(['membership'], axis = 1)
                        else: fvaldf = pd.concat([fvaldf, outputdf1], axis = 1).drop(['membership'], axis = 1)
                    else: 
                        if len(fvaldf) == 0: fvaldf = subdf.copy()
                        else: fvaldf = pd.concat([fvaldf, subdf], axis = 1) 
                else: 
                    outputdf1 = fuzzifyTestset(subdf, target, variable_cluster) 
                    if len(fvaldf) == 0: fvaldf = outputdf1.copy().drop(['membership'], axis = 1)
                    else: fvaldf = pd.concat([fvaldf, outputdf1], axis = 1).drop(['membership'], axis = 1)                    
                    if 'y_' in target: fvaldf = fvaldf.rename(columns = {target: target.replace('y_', 'ypcref_')})
                    else: fvaldf = fvaldf.drop([target], axis = 1)                    
    except: 
        fvaldf = None

    return ftraindf, fvaldf, ftestdf, variable_cluster
