

import sys, os, copy, random
import pandas as pd 
import torch, pickle
import gc

# from tqdm import tqdm 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from tqdm.auto import tqdm

sys.path.append(os.getcwd())
from s1_data_preparation.config import *
from s2_crystal_ball.config import *
from s2_crystal_ball.keras_neural import mlpConstructor, kerasTrain
from s2_crystal_ball.pytorch_neural import Transformer, pytorchTrain, MLP
from s2_crystal_ball.evaluation import evaluateModel

def neuralDataPreparation(ftraindf, fvaldf, ftestdf, period = PERIOD, plus_target = 1): 

    x_cols = [col for col in ftraindf.columns if 'x_' in col]
    y_cols = [col for col in ftraindf.columns if f'y_Tp{plus_target}_PriceChg_' in col]
    ref_cols = [col for col in ftraindf.columns if f'yref_Tp{plus_target}_Price' in col or f'refPrice_Tm{period}' in col or f'ypcref_Tp{plus_target}_PriceChg' in col or f'yref_Tp{plus_target}_Date' in col or 'yref_Tm0_close' in col]


    trainset = {'X': ftraindf[x_cols], 'y': ftraindf[y_cols], 'ref': ftraindf[ref_cols]}
    valset = {'X': fvaldf[x_cols], 'y': fvaldf[y_cols], 'ref': fvaldf[ref_cols]}
    testset = {'X': ftestdf[x_cols], 'y': ftestdf[y_cols], 'ref': ftestdf[ref_cols]}
    
    return trainset, valset, testset    


def neuralConstructor(model_dict): 

    model = None 

    if model_dict['model_type'] == 'mlp': model = mlpConstructor(model_dict)
    # elif model_dict['model_type'] == 'rnn': model = rnnConstructor(model_dict)
    elif model_dict['model_type'] == 'transformer': model = Transformer(model_dict)

    return model 
def monoNeuralPipeline(ftraindf, fvaldf, ftestdf, model_dict, cluster_details, plus_target=1, chosen_features = None, mode = 0):

    header = f'y_Tp{plus_target}_PriceChg'

    if mode == 1:
        rftraindf, rfvaldf, rftestdf, featureSelection = mces(ftraindf, fvaldf, ftestdf, cluster_details, plus_target, chosen_features)
        trainset, valset, testset = neuralDataPreparation(rftraindf, rfvaldf, rftestdf, plus_target=plus_target)

    else:
        trainset, valset, testset = neuralDataPreparation(ftraindf, fvaldf, ftestdf, plus_target=plus_target)

    model_dict['input_size'], model_dict['output_size'], model_dict['day_target'] = len(trainset['X'].columns), len(trainset['y'].columns), plus_target

    if model_dict['model_type'] == 'transformer':
        # select nheads & encoder layers 
        nheads = [i for i in range(1, 11) if model_dict['input_size'] % i == 0]
        nhead = max(nheads)
        model_dict['transformer']['nhead'] = nhead
        model_dict['transformer']['num_encoder_layers'] = nhead       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = neuralConstructor(model_dict).to(device)
    else:
        model = neuralConstructor(model_dict)

    outcome = {}
    ticker = 'S63.SI'
    if model_dict['model_type'] != 'transformer': 
        model, outcome['train_pred'], outcome['val_pred'], outcome['test_pred'] = kerasTrain(model, model_dict, cluster_details, trainset, valset, testset, header)
    else: 
        model, outcome['val_pred'], outcome['test_pred'] = pytorchTrain(model, model_dict, cluster_details, trainset, valset, testset, header)
    with open(f'/home/priya/Desktop/fyp/Src alwin/Src/data/{ticker}/cluster_details.pkl', 'rb') as handle:
        cluster_details = pickle.load(handle)


    with open(f'/home/priya/Desktop/fyp/Src alwin/Src/data/{ticker}/cluster_details.pkl', 'rb') as handle:
        cluster_details = pickle.load(handle)
    print("✅ Loaded cluster_details")

    print(f"Calling evaluateModel with arguments:")
    print(f"outcome keys: {list(outcome.keys())}")
    print(f"trainset keys: {list(trainset.keys())}")
    print(f"valset keys: {list(valset.keys())}")
    print(f"testset keys: {list(testset.keys())}")
    print(f"cluster_details keys: {list(cluster_details.keys())}")
    print(f"header: {header}")
    print(f"plus_target: {plus_target}")
    train = trainset
    test = testset
    val = valset
    clust = cluster_details
    head = header
    target = plus_target
    PERIOD = 14

    eval_res = evaluateModel(
        outcome=outcome,
        trainset=trainset,
        valset=valset,
        testset=testset,
        cluster_details=cluster_details,
        header=header,
        plus_target=plus_target
    )


    eval_res['header'] = header

    gc.collect()

    if mode == 1: 
        return model, eval_res, featureSelection
    return model, eval_res


def optimizeMonoNetwork(ftraindf, fvaldf, ftestdf, cluster_details, plus_target): 

    model_type = 'transformer'

    configurations = generateConfigurations(model_type, ftraindf)
    
    models = [] 
    val_performance_r2 = []
    val_performance_rmse = []
    test_performance_r2 = []     
    test_performance_rmse = [] 
    
    for configuration in tqdm(configurations): 
        print(configuration)
        model, eval_res = monoNeuralPipeline(ftraindf, fvaldf, ftestdf, configuration, cluster_details, plus_target)
        models.append(model)
        val_performance_r2.append(eval_res['val_pred']['r2'])
        val_performance_rmse.append(eval_res['val_pred']['rmse'])
        test_performance_r2.append(eval_res['test_pred']['r2'])
        test_performance_rmse.append(eval_res['test_pred']['rmse'])
        
        # print(f'Config {configurations.index(configuration)} - val_r2: {eval_res['val_pred']['r2']}, val_rmse: {eval_res['val_pred']['rmse']} | test_r2: {eval_res['test_pred']['r2']}, test_rmse: {eval_res['test_pred']['rmse']}')

    
    index = val_performance_rmse.index(max(val_performance_rmse))
    optimal_config = configurations[index]
    optimal_model = models[index]
    print(f'Optimal Config {index} - val_r2: {val_performance_r2[index]}, val_rmse: {val_performance_rmse[index]} | test_r2: {test_performance_r2[index]}, test_rmse: {test_performance_rmse[index]}')
    
    return optimal_model, optimal_config

def generateConfigurations(model_type, ftraindf):

    configurations = []

    x_cols = [col for col in ftraindf if 'x_' in col]


    if model_type == 'transformer': 
        template = {
            'model_type': 'transformer', 
            'transformer':{
        #         'd_model': 
                # 'nheads': 1, 
        #         'dim_feedforward': {64, 128, 256, 512}, 
        #         'num_encoder_layers': {2, 4, 8, 16}, 
                'shuffle': False, 
                'optimizer': {
                    'optim_type': 'adam', 
                    'learning_rate': 0.001, 
                },         
            }, 
            'early_stopper': {
                'patience': 10, 
                'min_delta': 0,
            },     
            'epochs': 100,
        #     'batch_size': {32, 64, 128, 256},
        }

        dim_feedforward = [128, 256] # [64, 128, 256, 512]
        num_encoder_layers = [2, 4, 6] # [2, 4, 6, 8]
        batch_sizes = [32, 64, 128] # [32, 64, 128, 256]
        nheads = [i for i in range(1, 9) if len(x_cols)%i == 0]

        for dim in dim_feedforward: 
            for encoder_layer in num_encoder_layers: 
                for batch_size in batch_sizes: 
                    for nhead in nheads: 
                        instance = copy.deepcopy(template)
                        
                        instance[instance['model_type']]['dim_feedforward'] = dim
                        instance[instance['model_type']]['num_encoder_layers'] = encoder_layer
                        instance[instance['model_type']]['nheads'] = nhead
                        instance['batch_size'] = batch_size
                        
                        configurations.append(instance)

    return configurations



def mcesDefuzzy(m1_pred, header, y_cols, cluster_details): 

    m1_pred['minimum'] = m1_pred.apply(lambda x: min(x), axis = 1)
    for col in y_cols: m1_pred[col] = m1_pred.apply(lambda x: round((x[col] - x['minimum']), 6), axis = 1)

    m1_pred['summation'] = m1_pred.apply(lambda x: sum(x), axis = 1)
    for col in y_cols: m1_pred[col] = m1_pred.apply(lambda x: round(x[col]/x['summation'], 6), axis = 1)

    m1_pred = m1_pred.drop(['minimum', 'summation'], axis = 1) 

    m1_pred['pc_pred'] = 0 
    for col in y_cols: m1_pred['pc_pred'] = m1_pred.apply(lambda x: x['pc_pred'] + x[col]*cluster_details[header][col]['mean'], axis = 1)

    return m1_pred['pc_pred']

def mces(ftraindf, fvaldf, ftestdf, cluster_details, plus_target, chosen_features = None, threshold = 10, iterations = 0): 

    print('[mces initiated] training mlp model ...')

    price_features = [col for col in list(cluster_details.keys()) if 'Price' in col and 'x_' in col]
    x_cols = [col for col in ftraindf.columns if 'x_' in col]
    remainder_cols = [col for col in ftraindf.columns if col not in x_cols]

    if chosen_features == None: 

        # define MLP Model 
        model_dict = {
            'model_type': 'mlp', 
            'mlp': {
                'layers': {
                    0: {'nodes': 256}, 
                    1: {'nodes': 64}, 
                    2: {'nodes': 128}, 
                }, 
                'hl_activation': 'relu', 
                'ol_activation': 'sigmoid',
                'optimizer': {
                    'optim_type': 'adam', 
                    'learning_rate': 0.001,
                }, 
                'shuffle': False, 
                'verbose': 0,
            },
            'early_stopper': {
                'patience': 5, 
                'min_delta': 0        
            },
            'epochs': 20,
            'batch_size': 16,
        }

        # train a model with full features 
        model, eval_res, nothing = monoNeuralPipeline(ftraindf, fvaldf, ftestdf, model_dict, cluster_details, plus_target, mode = 0)

        x_features = [col for col in list(cluster_details.keys()) if 'x_' in col and col not in price_features]
        y_cols = [col for col in ftraindf.columns if f'y_Tp{plus_target}_' in col]
        header = f'y_Tp{plus_target}_PriceChg'
        target = ftraindf[f'ypcref_Tp{plus_target}_PriceChg']

        X_mces = ftraindf[x_cols].reset_index(drop = True)

        mces_df = pd.DataFrame({'cols': x_features})
        scores = [0 for item in range(len(x_features))] 
        frequency = [0 for item in range(len(x_features))] 
        in_mask = [0 for item in range(len(x_features))] 
        
        if iterations == 0: iterations = len(X_mces.index)
        # print(f'xcols len: {len(x_cols)}')
        for row_index in tqdm(range(iterations)): 

            temp = X_mces.copy()[X_mces.index == row_index]
            temp_target = pd.Series(target[row_index])

            mask1 = [random.randint(0, 1) for col in range(len(x_features))]

            # in case no more features left 
            while sum(mask1) < 7: mask1 = [random.randint(0, 1) for col in range(len(x_features))]
            mces_df['mask1'] = mask1

            disabled = list(mces_df[mces_df['mask1'] == 0]['cols'])
            enabled = list(mces_df[mces_df['mask1'] == 1]['cols'])

            inverted = random.choice(enabled)
            inverted_index = x_features.index(inverted)

            # retrieve respective cols 

            for col in disabled: 
                for cluster in list(cluster_details[col].keys()): 
                    temp[cluster] = 0
            for col in enabled: in_mask[x_features.index(col)] += 1
            frequency[inverted_index] += 1

            # mask 1 
            

            pred = model.predict(temp, verbose = model_dict['mlp']['verbose'])
            # print(f'Pred shape: {pred.shape}')
            # print(y_cols)

            m1_pred = pd.DataFrame(pred, columns=y_cols)
            m1_predp = mcesDefuzzy(m1_pred, header, y_cols, cluster_details)
            rmse1 = mean_squared_error(temp_target, m1_predp)

            for cluster in list(cluster_details[inverted].keys()): 
                temp[cluster] = X_mces[cluster].mean()

            m2_pred = pd.DataFrame(model.predict(temp, verbose = model_dict['mlp']['verbose']), columns=y_cols)
            m2_predp = mcesDefuzzy(m2_pred, header, y_cols, cluster_details)
            rmse2 = mean_squared_error(temp_target, m2_predp)

            rmseDiff = rmse1 - rmse2

            scores[inverted_index] += rmseDiff
            
            # if row_index > 3: break

        mces_df['scores'] = scores
        mces_df['in_mask_freq'] = in_mask
        mces_df['frequency'] = frequency
        mces_df = mces_df.drop(['mask1'], axis = 1)
        
        try: 
            mces_df['weighted_scores'] = mces_df.apply(lambda x: x['scores']/x['frequency'], axis = 1)
        except: 
            mces_df['weighted_scores'] = mces_df.apply(lambda x: x['scores'], axis = 1)
        mces_df = mces_df.sort_values(by = ['weighted_scores'], ascending = False)    
        
        top_features = list(mces_df[mces_df['weighted_scores'] > 0]['cols'])
#         if len(top_features) < threshold: top_features = list(mces_df.head(threshold)['cols'])
        
        top_features += price_features
    
    else: top_features = chosen_features['top_features']

    feature_cols = [] 
    for feature in top_features: feature_cols += list(cluster_details[feature].keys())

    feature_cols += remainder_cols
    
    rftraindf, rfvaldf, rftestdf = ftraindf.copy()[feature_cols], fvaldf.copy()[feature_cols], ftestdf.copy()[feature_cols]

    featureSelection = {}
    featureSelection['top_features'] = top_features 
    if chosen_features == None: featureSelection['mces_df'] = mces_df 
    else: featureSelection['top_features'] = chosen_features['mces_df']

    print(f'[MCES] Features Selected: {top_features}')

    return rftraindf, rfvaldf, rftestdf, featureSelection


def ensemble(val_df, test_df, ref, pred_period = PRED_PERIOD):
    output_results = test_df[[col for col in test_df.columns if 'mdate_ref' in col or 'close' in col]]
    # print(output_results.columns)
    traincols = [col for col in val_df.columns if 'mdate_ref' not in col]
    traindata = val_df[traincols]

    testcols = [col for col in test_df.columns if 'mdate_ref' not in col]
    testdata = test_df[testcols]

    # print(f'traindata: {traincols}')
    # print(f'testdata: {testcols}')

    headers = []
    r2_scores = []
    rmse_scores = []
    mape_scores = []

    for index in range(1, pred_period + 1):

        # train test preparation
        X_train = traindata.iloc[:, index :]

        y_train = val_df.iloc[:, [0]]

        X_test = testdata.iloc[:, index :]
        y_test = testdata.iloc[:, [0]]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test['close'], pred)
        rmse = mean_squared_error(y_test['close'], pred)**0.5
        mape = mean_absolute_percentage_error(y_test['close'], pred)

        print(f'[TEST OUTCOME_Tp{index}] - R2: {r2}, RMSE: {rmse}, MAPE: {mape}')

        headers.append(f'Tp{index}_pred')
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mape_scores.append(mape)

        output_results[f'Tp{index}_pred'] = pred

    stats_data = {
        'columns': headers,
        'r2': r2_scores,
        'rmse': rmse_scores,
        'mape': mape_scores
    }
    stats_df = pd.DataFrame(stats_data)


    output = ref

    for pred_day in range(1, pred_period + 1):

        header_ref = f'Tp{pred_day}_'

        instance = output_results[[col for col in output_results.columns if header_ref in col]]
        # print(instance.columns)
        instance = instance.reset_index(drop = False) # pred_date out
        instance = instance.rename(columns = {'pred_date': f'Tp{pred_day}_date_ref', f'Tp{pred_day}_mdate_ref':'Date'})
        instance = instance.set_index('Date')

        output = pd.concat([output, instance], axis = 1)

    output = output.dropna()
    output1_dates = output[[col for col in output.columns if 'mdate' in col]]
    output1_values = output[[col for col in output.columns if 'mdate' not in col]]

    return output_results, output1_values, output1_dates, stats_df


print("✅ Ensemble predictions and stats saved to CSV files.")

def rangeNeuralPipeline(ticker, ftraindf, fvaldf, ftestdf, cluster_details, model_dict, chosen_features = None, google = 1, pred_period = PRED_PERIOD, n_elements = 6):
    if chosen_features == None:
        if google == 1:
            wkdir = f'/home/priya/Desktop/fyp/Src alwin/Src/s3_crystalball outcome/{ticker}/data'
            with open(f'{wkdir}/mces/features_selected.pkl', 'rb') as handle:
                chosen_features = pickle.load(handle)
        else:
            wkdir = f'data/{ticker}/transformer'
            with open(f'data/{ticker}/mces/features_selected.pkl', 'rb') as handle:
                chosen_features = pickle.load(handle)

    headers_compile = []
    r2_compile = []
    rmse_compile = []
    mape_compile = []

    date_cols = [col for col in ftestdf.columns if '_Date' in col]

    for plus_target in tqdm(range(1, pred_period + 1)):

        # train model, predict & evaluate model performance
        model, eval_res = monoNeuralPipeline(ftraindf=ftraindf, fvaldf=fvaldf, ftestdf=ftestdf, model_dict=model_dict[plus_target], cluster_details=cluster_details, plus_target=plus_target, chosen_features=chosen_features[plus_target])
        # model, eval_res = monoNeuralPipeline(ftraindf=ftraindf, fvaldf=fvaldf, ftestdf=ftestdf, model_dict=model_dict, cluster_details=cluster_details, plus_target=plus_target, chosen_features=chosen_features)


        # evaluation metrics (for summary)
        iheader = eval_res['header']
        ir2 = eval_res['test_pred']['r2']
        irmse = eval_res['test_pred']['rmse']
        imape = eval_res['test_pred']['mape']

        headers_compile.append(iheader)
        r2_compile.append(ir2)
        rmse_compile.append(irmse)
        mape_compile.append(imape)

        # save model
        torch.save(model.state_dict(), f'{wkdir}/Tp{plus_target}.pt')

        # save prediction dataset
        # obtain the columns for reference
        instance_references = [col for col in date_cols if f'Tp{plus_target}_' in col] + ['yref_Tm0_close']

        # to include predicted dates by models
        date_testref = ftestdf[instance_references]
        date_valref = fvaldf[instance_references]

        # concat prediction results (to obtain date references )
        test_results = pd.concat([eval_res['test_pred']['ref'], date_testref], axis = 1)
        test_results = test_results.rename(columns = {'price_pred' : f'Tp{plus_target}_pred', 'yref_Tm0_close' : 'close', f'yref_Tp{plus_target}_Date': f'Tp{plus_target}_date_ref'})

        val_results = pd.concat([eval_res['val_pred']['ref'], date_valref], axis = 1)
        val_results = val_results.rename(columns = {'price_pred' : f'Tp{plus_target}_pred', 'yref_Tm0_close' : 'close', f'yref_Tp{plus_target}_Date': f'Tp{plus_target}_date_ref'})

        # save the results
        val_results.to_csv(f'{wkdir}/val/Tp{plus_target}_valresults.csv')
        test_results.to_csv(f'{wkdir}/test/Tp{plus_target}_testresults.csv')

        # save train predictions for fuzzy logic
        eval_res['train_pred']['cluster_ref'].to_csv(f'{wkdir}/train/Tp{plus_target}_train_clustermembership.csv')
        eval_res['val_pred']['cluster_ref'].to_csv(f'{wkdir}/val/Tp{plus_target}_val_clustermembership.csv')
        eval_res['test_pred']['cluster_ref'].to_csv(f'{wkdir}/test/Tp{plus_target}_test_clustermembership.csv')


        if plus_target == 1:

            for index in range(2):

                if index == 1:
                    item = test_results
                    header = 'test'
                elif index == 0:
                    item = val_results
                    header = 'val'


                # predictions by model dates
                overall_results_predmodel = item[['close', f'Tp{plus_target}_pred', f'Tp{plus_target}_date_ref']] # by prediction model
                overall_results_predmodel.to_csv(f'{wkdir}/{header}/OVERALL_prediction_by_model.csv')

                # predictions by prediction dates
                if index == 1: overall_results_predday = pd.concat([val_results, test_results], axis = 0)
                else: overall_results_predday = val_results
                close_ref = overall_results_predday[['close']]
                overall_results_predday = overall_results_predday[[f'Tp{plus_target}_date_ref', f'Tp{plus_target}_pred']]

                # reindex to reference predicted date
                overall_results_predday = overall_results_predday.reset_index(drop = False)
                overall_results_predday = overall_results_predday.rename(columns = {'Date':f'Tp{plus_target}_mdate_ref', f'Tp{plus_target}_date_ref':'pred_date'})
                overall_results_predday = overall_results_predday.set_index('pred_date')
                overall_results_predday = pd.concat([overall_results_predday, close_ref], axis = 1)
                overall_results_predday = overall_results_predday[['close', f'Tp{plus_target}_pred', f'Tp{plus_target}_mdate_ref']]
                overall_results_predday = overall_results_predday.dropna()
                overall_results_predday['pred_date'] = overall_results_predday.index
                overall_results_predday = overall_results_predday.set_index('pred_date')
                if index == 1: overall_results_predday = overall_results_predday.tail(len(test_results))

                overall_results_predday.to_csv(f'{wkdir}/{header}/OVERALL_prediction_by_date.csv')

        else:
            for index in range(2):

                if index == 1:
                    item = test_results
                    header = 'test'
                elif index == 0:
                    item = val_results
                    header = 'val'


                # read from csv
                overall_results_predmodel = pd.read_csv(f'{wkdir}/{header}/OVERALL_prediction_by_model.csv', index_col = 'Date')
                overall_results_predday = pd.read_csv(f'{wkdir}/{header}/OVERALL_prediction_by_date.csv', index_col = 'pred_date')

                # predmodel
                predmodel = item[[f'Tp{plus_target}_pred', f'Tp{plus_target}_date_ref']] # by prediction model
                overall_results_predmodel = pd.concat([overall_results_predmodel, predmodel], axis = 1)

                # predday
                # predictions by prediction dates
                if index == 1: pred_day = pd.concat([val_results, test_results], axis = 0)
                else: pred_day = val_results
                pred_day = pred_day[[f'Tp{plus_target}_date_ref', f'Tp{plus_target}_pred']]

                # reindex to reference predicted date
                pred_day = pred_day.reset_index(drop = False)
                pred_day = pred_day.rename(columns = {'Date':f'Tp{plus_target}_mdate_ref', f'Tp{plus_target}_date_ref':'pred_date'})
                pred_day = pred_day.set_index('pred_date')
                pred_day = pred_day[[f'Tp{plus_target}_pred', f'Tp{plus_target}_mdate_ref']]
                overall_results_predday = pd.concat([overall_results_predday, pred_day], axis = 1)
                overall_results_predday = overall_results_predday.dropna()

                # write to csv
                overall_results_predmodel.to_csv(f'{wkdir}/{header}/OVERALL_prediction_by_model.csv')
                overall_results_predday.to_csv(f'{wkdir}/{header}/OVERALL_prediction_by_date.csv')

        gc.collect()

    ref = overall_results_predmodel[['close']]

    stats_data = {
        'columns': headers_compile,
        'r2_price': r2_compile,
        'rmse': rmse_compile,
        'mape': mape_compile
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f'{wkdir}/statistics_before_ensemble.csv')

    return overall_results_predmodel, overall_results_predday

