import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error
import sys, os 
sys.path.append(os.getcwd())
from s1_data_preparation.config import *
# def evaluateModel(outcome, trainset, valset, testset, cluster_details, header, plus_target, period=14):

#     eval_res = {}
#     print("Starting model evaluation...")
    
#     for result in outcome: 
#         print(f"Processing result: {result}")

#         # if result == 'train_pred': continue

#         mode = 0

#         ## normalization 
#         pred_res = outcome[result].copy()
#         print(f"Initial prediction results for {result}:\n", pred_res.head())
        
#         if result == 'val_pred': 
#             reference = valset['ref']
#         elif result == 'test_pred': 
#             reference = testset['ref']
#         else: 
#             reference = trainset['ref']
#         print(f"Reference data for {result}:\n", reference.head())
        
#         cols = list(pred_res.columns)
#         print(f"Prediction columns: {cols}")

#         if mode == 0: 
#             print("Applying normalization mode 0...")
            
#             pred_res['summation'] = pred_res.apply(lambda x: sum(x), axis=1)
#             for col in cols: 
#                 pred_res[col] = pred_res.apply(lambda x: round(x[col]/x['summation'], 6), axis=1)
            
#             pred_res = pred_res.drop(['summation'], axis=1) 
#             print(f"Normalized predictions: \n", pred_res.head())

#         if mode == 1: 
#             print("Applying normalization mode 1...")
#             pred_res['maximum'] = pred_res.apply(lambda x: max(x), axis=1)
#             for col in cols: 
#                 pred_res[col] = pred_res.apply(lambda x: 1 if x[col] == x['maximum'] else 0, axis=1)
            
#         ## defuzzify 
#         print("Performing defuzzification...")
#         pred_res['pc_pred'] = 0 
#         for col in cols: 
#             pred_res['pc_pred'] = pred_res.apply(lambda x: x['pc_pred'] + x[col]*cluster_details[header][col]['mean'], axis=1)
#         print(f"Defuzzified predictions: \n", pred_res.head())
    
#         # price change 
#         print("Calculating price predictions...")
#         price_pred = pd.concat([pred_res, reference[[f'refPrice_Tm{period}', f'yref_Tp{plus_target}_Price', f'ypcref_Tp{plus_target}_PriceChg']]], axis=1)
#         price_pred['price_pred'] = price_pred.apply(lambda x: x[f'refPrice_Tm{period}']*(1+x['pc_pred']), axis=1) 
#         price_pred = price_pred[[f'yref_Tp{plus_target}_Price', 'price_pred', f'ypcref_Tp{plus_target}_PriceChg', 'pc_pred']]
#         price_pred['error'] = price_pred.apply(lambda x: abs(x['price_pred'] - x[f'yref_Tp{plus_target}_Price'])/x[f'yref_Tp{plus_target}_Price'], axis=1)
#         print(f"Price prediction results: \n", price_pred.head())

#         pred_r2 = r2_score(price_pred[f'yref_Tp{plus_target}_Price'], price_pred['price_pred'])
#         pred_rmse = mean_squared_error(price_pred[f'yref_Tp{plus_target}_Price'], price_pred['price_pred'])**0.5
#         print(f"Metrics - RMSE: {pred_rmse}, R2: {pred_r2}, MAPE: {price_pred['error'].mean()}")

#         eval_res[result] = {
#             'rmse': pred_rmse, 
#             'r2': pred_r2, 
#             'mape': price_pred['error'].mean(), 
#             'predicted': reference, 
#             'ref': price_pred, 
#             'cluster_ref': pred_res
#         }
    
#     print("Model evaluation completed.")
#     return eval_res


def evaluateModel(outcome, trainset, valset, testset, cluster_details, header, plus_target):
    eval_res = {}

    period = 14  # ✅ Add this line!

    for result in outcome:
        pred_res = outcome[result].copy()
        
        if result == 'val_pred':
            reference = valset['ref']
        elif result == 'test_pred':
            reference = testset['ref']
        else:
            reference = trainset['ref']

        cols = list(pred_res.columns)

        # Normalize (mode 0)
        pred_res['summation'] = pred_res.apply(lambda x: sum(x), axis=1)
        for col in cols:
            pred_res[col] = pred_res.apply(lambda x: round(x[col] / x['summation'], 6), axis=1)
        pred_res = pred_res.drop(['summation'], axis=1)

        # Defuzzify
        # pred_res['pc_pred'] = 0
        # for col in cols:
        #     pred_res['pc_pred'] = pred_res.apply(lambda x: x['pc_pred'] + x[col] * cluster_details[header][col]['mean'], axis=1)
        pred_res['pc_pred'] = 0
        for col in cols:
            if col in cluster_details[header]:
                pred_res['pc_pred'] += pred_res[col] * cluster_details[header][col]['mean']
            else:
                print(f"⚠️ Warning: Cluster '{col}' missing in cluster_details[{header}] — skipped.")

        # Predict price
        price_pred = pd.concat([
            pred_res,
            reference[[f'refPrice_Tm{period}', f'yref_Tp{plus_target}_Price', f'ypcref_Tp{plus_target}_PriceChg']]
        ], axis=1)

        price_pred['price_pred'] = price_pred.apply(lambda x: x[f'refPrice_Tm{period}'] * (1 + x['pc_pred']), axis=1)
        price_pred = price_pred[[f'yref_Tp{plus_target}_Price', 'price_pred', f'ypcref_Tp{plus_target}_PriceChg', 'pc_pred']]
        price_pred['error'] = price_pred.apply(
            lambda x: abs(x['price_pred'] - x[f'yref_Tp{plus_target}_Price']) / x[f'yref_Tp{plus_target}_Price'], axis=1
        )

        pred_r2 = r2_score(price_pred[f'yref_Tp{plus_target}_Price'], price_pred['price_pred'])
        pred_rmse = mean_squared_error(price_pred[f'yref_Tp{plus_target}_Price'], price_pred['price_pred']) ** 0.5
        pred_mape = price_pred['error'].mean()

        eval_res[result] = {
            'rmse': pred_rmse,
            'r2': pred_r2,
            'mape': pred_mape,
            'predicted': reference,
            'ref': price_pred,
            'cluster_ref': pred_res
        }
        print(f"{result}: RMSE={pred_rmse:.3f}, R2={pred_r2:.3f}, MAPE={pred_mape:.3%}")

    return eval_res
