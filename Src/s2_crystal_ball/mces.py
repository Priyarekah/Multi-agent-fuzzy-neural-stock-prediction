import random, pickle
import pandas as pd 
from tqdm.auto import tqdm
from datetime import datetime

import os, sys 

from sklearn.metrics import mean_squared_error
from s1_data_preparation.config import *
from s2_crystal_ball.config import *
from s2_crystal_ball.neural_network import monoNeuralPipeline

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

    print(f'[mces initiated - {plus_target}] training mlp model ...')

    # price_features = [col for col in list(cluster_details.keys()) if 'Price' in col and 'x_' in col]
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
                    'learning_rate': 0.0001,
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
        model, eval_res = monoNeuralPipeline(ftraindf=ftraindf, fvaldf=fvaldf, ftestdf=ftestdf, model_dict=model_dict, cluster_details=cluster_details, plus_target=plus_target, mode = 0)

        # x_features = [col for col in list(cluster_details.keys()) if 'x_' in col and col not in price_features]
        x_features = [col for col in list(cluster_details.keys()) if 'x_' in col]        
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
            pred = model.predict(temp, verbose = 0)

            m1_pred = pd.DataFrame(pred, columns=y_cols)
            m1_predp = mcesDefuzzy(m1_pred, header, y_cols, cluster_details)
            rmse1 = mean_squared_error(temp_target, m1_predp)

            for cluster in list(cluster_details[inverted].keys()): 
                temp[cluster] = X_mces[cluster].mean()

            m2_pred = pd.DataFrame(model.predict(temp, verbose = model_dict['mlp']['verbose']), columns=y_cols)
            m2_predp = mcesDefuzzy(m2_pred, header, y_cols, cluster_details)
            rmse2 = mean_squared_error(temp_target, m2_predp)

            rmseDiff = rmse2 - rmse1

            scores[inverted_index] += rmseDiff
            
            # print(row_index)
            # if row_index > 10: break


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
        
        # top_features += price_features
    
    else: top_features = chosen_features['top_features']

    feature_cols = []
    for feature in top_features:
        feature_cols += list(cluster_details[feature].keys())
    feature_cols += remainder_cols

    # ✅ Add this block
    rftraindf = ftraindf[feature_cols]
    rfvaldf = fvaldf[feature_cols]
    rftestdf = ftestdf[feature_cols]

    featureSelection = {
        'top_features': top_features,
        'mces_df': mces_df
    }

    print(f'[MCES] Features Selected: {top_features}')
    return rftraindf, rfvaldf, rftestdf, featureSelection


def mcesPipeline(ticker, ftraindf, fvaldf, ftestdf, cluster_details, start_index=1, end_index=13, pred_period=13): 
    print(f"📌 Starting MCES Pipeline for {ticker} from Tp{start_index} to Tp{end_index}")

    # Load or initialize features_selected dictionary
    try:
        with open(f'/home/priya/Desktop/fyp/Src alwin/Src/s3_crystalball outcome/{ticker}/data/mces/features_selected.pkl', 'rb') as handle:
            features_selected = pickle.load(handle)   
        print("✅ Loaded existing features_selected.pkl")
    except FileNotFoundError: 
        features_selected = {}
        with open(f'/home/priya/Desktop/fyp/Src alwin/Src/s3_crystalball outcome/{ticker}/data/mces/features_selected.pkl', 'wb') as fp: 
            pickle.dump(features_selected, fp)
        print("⚠️ features_selected.pkl not found. Initialized empty dictionary.")


        # Ensure 'Date' is in datetime format
        print("🕒 Checking and converting 'Date' column to datetime format if needed...")

        if isinstance(ftraindf['Date'].iloc[0], str):  # Convert if it's a string
            ftraindf['Date'] = pd.to_datetime(ftraindf['Date']).dt.date
            print("✅ Converted 'Date' from string to datetime.date format.")
        elif isinstance(ftraindf['Date'].iloc[0], pd.Timestamp):  # Convert if it's datetime
            ftraindf['Date'] = ftraindf['Date'].dt.date
            print("✅ Converted 'Date' from datetime to datetime.date format.")
        else:
            print("⚠️ 'Date' is already in correct datetime.date format. No conversion needed.")

        # Now apply the MCES date filter
        print(f"🔍 Filtering dataset between {MCES_START} and {MCES_END}...")
        ftraindf = ftraindf[(ftraindf['Date'] >= MCES_START) & (ftraindf['Date'] < MCES_END)]
        print(f"✅ Filtered dataset shape: {ftraindf.shape}")

    # Loop through prediction targets
    for plus_target in tqdm(range(start_index, end_index + 1)): 
        print(f"\n🚀 Running MCES for Tp{plus_target}...")

        # Run MCES feature selection
        try:
            feature_cols, mces_df = mces(ftraindf, fvaldf, ftestdf, cluster_details, plus_target)
            print(f"✅ MCES completed for Tp{plus_target}. Selected {len(feature_cols)} features.")
        except Exception as e:
            print(f"❌ Error in MCES for Tp{plus_target}: {e}")
            continue

        # Load existing feature selection results
        try:
            with open(f'/home/priya/Desktop/fyp/Src alwin/Src/s3_crystalball outcome/{ticker}/data/mces/features_selected.pkl', 'rb') as handle:
                features_selected = pickle.load(handle)    
        except FileNotFoundError:
            print(f"⚠️ features_selected.pkl missing while loading for Tp{plus_target}. Initializing new dict.")
            features_selected = {}

        # Store selected features
        features_selected[plus_target] = feature_cols
        print(f"💾 Saving selected features for Tp{plus_target}...")

        # Save MCES DataFrame
        mces_file_path = f'/home/priya/Desktop/fyp/Src alwin/Src/s3_crystalball outcome/{ticker}/data/mces/Tp{plus_target}_mcesdf.csv'
        try:
            mces_df.to_csv(mces_file_path, index=False)
            print(f"✅ MCES DataFrame saved at: {mces_file_path}")
        except Exception as e:
            print(f"❌ Error saving MCES DataFrame for Tp{plus_target}: {e}")

        # Save updated features_selected
        try:
            with open(f'/home/priya/Desktop/fyp/Src alwin/Src/s3_crystalball outcome/{ticker}/data/mces/features_selected.pkl', 'wb') as fp: 
                pickle.dump(features_selected, fp)  
            print(f"✅ Updated features_selected.pkl with Tp{plus_target} features.")
        except Exception as e:
            print(f"❌ Error saving features_selected.pkl for Tp{plus_target}: {e}")

    print("\n🎯 MCES Pipeline completed successfully!\n")
    return features_selected  


# ✅ MCES Main Execution Script (D05.SI) from Pre-Fuzzified Data
if __name__ == "__main__":
    import pandas as pd
    import pickle
    import os
    from tqdm.auto import tqdm
    from s2_crystal_ball.config import MCES_START, MCES_END
    from s2_crystal_ball.mces import mces
    from s2_crystal_ball.mces import mcesDefuzzy  # if needed

    ticker = "AJBU.SI"
    print(f"\n🚀 Running MCES Execution Pipeline for {ticker.upper()} (Pre-Fuzzified Mode)\n")

    # Load pre-fuzzified data
    data_dir = f"data/{ticker}"
    ftraindf = pd.read_csv(f"{data_dir}/ftraindf.csv")
    fvaldf = pd.read_csv(f"{data_dir}/fvaldf.csv")
    ftestdf = pd.read_csv(f"{data_dir}/ftestdf.csv")
    with open(f"{data_dir}/cluster_details.pkl", "rb") as f:
        cluster_details = pickle.load(f)

    print("✅ Pre-fuzzified data loaded successfully.")

    # --- Filter training date range for MCES ---
    if isinstance(ftraindf['Date'].iloc[0], str):
        ftraindf['Date'] = pd.to_datetime(ftraindf['Date']).dt.date
    elif isinstance(ftraindf['Date'].iloc[0], pd.Timestamp):
        ftraindf['Date'] = ftraindf['Date'].dt.date

    ftraindf = ftraindf[(ftraindf['Date'] >= MCES_START) & (ftraindf['Date'] < MCES_END)]

    # --- Generate feature selection ---
    features_selected = {}
    plus_target_start = 12
    plus_target_end = 13

    for plus_target in range(plus_target_start, plus_target_end + 1):
        print(f"\n🚀 Running MCES for Tp{plus_target}...")
        rftraindf, rfvaldf, rftestdf, featureSelection = mces(
            ftraindf,
            fvaldf,
            ftestdf,
            cluster_details,
            plus_target=plus_target,
            threshold=10,
            iterations=100  # You can increase for better results
        )
        features_selected[plus_target] = featureSelection['top_features']

    # # --- Save the new feature selection ---
    # output_path = f"{data_dir}/features_selected.pkl"
    # os.makedirs(data_dir, exist_ok=True)
    # with open(output_path, "wb") as f:
    #     pickle.dump(features_selected, f)

    # print(f"\n✅ Saved new features_selected.pkl to {output_path}")
        tp_dir = f"{data_dir}/Tp{plus_target}"
        os.makedirs(tp_dir, exist_ok=True)

        # rftraindf.to_csv(f"{tp_dir}/rftraindf.csv", index=False)
        # rfvaldf.to_csv(f"{tp_dir}/rfvaldf.csv", index=False)
        # rftestdf.to_csv(f"{tp_dir}/rftestdf.csv", index=False)

        pd.DataFrame({'key': list(featureSelection.keys()), 'value': list(featureSelection.values())}).to_csv(f"{tp_dir}/featureSelection.csv", index=False)
        pd.DataFrame({"selected_features": featureSelection['top_features']}).to_csv(f"{tp_dir}/top_features.csv", index=False)

        # ✅ Save full mces_df as Tp{plus_target}_mcesdf.csv
        mcesdf_path = f"{tp_dir}/Tp{plus_target}_mcesdf.csv"
        if 'mces_df' in featureSelection:
            featureSelection['mces_df'].to_csv(mcesdf_path, index=False)
            print(f"✅ Saved full MCES DataFrame at {mcesdf_path}")

    # --- Save the new feature selection for all targets ---
    output_path = f"{data_dir}/features_selected.pkl"
    os.makedirs(data_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(features_selected, f)

    print(f"\n✅ Saved new features_selected.pkl to {output_path}")
