import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import numpy as np

PRED_PERIOD = 13  # Set your prediction horizon

def ensemble(val_df, test_df, ref, pred_period=PRED_PERIOD):
    output_results = test_df[[col for col in test_df.columns if 'mdate_ref' in col or 'yref_Tm0_close' in col]]
    print("Output results columns:", output_results.columns)

    traincols = [col for col in val_df.columns if 'mdate_ref' not in col and 'Date' not in col]
    traindata = val_df[traincols]

    testcols = [col for col in test_df.columns if 'mdate_ref' not in col and 'Date' not in col]
    testdata = test_df[testcols]

    print(f'traindata: {traincols}')
    print(f'testdata: {testcols}')

    headers = []
    r2_scores = []
    rmse_scores = []
    mape_scores = []

    for index in range(1, pred_period + 1):
        # train test preparation
        X_train = traindata.iloc[:, index:].select_dtypes(include=[np.number])
        y_train = val_df[['yref_Tm0_close']]

        X_test = testdata.iloc[:, index:].select_dtypes(include=[np.number])
        y_test = test_df[['yref_Tm0_close']]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, pred)

        print(f'[TEST OUTCOME_Tp{index}] - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAPE: {mape*100:.2f}%')

        headers.append(f'Tp{index}_pred')
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mape_scores.append(mape)

        output_results[f'Tp{index}_pred'] = pred.flatten()

    # Stats dataframe
    stats_df = pd.DataFrame({
        'columns': headers,
        'r2': r2_scores,
        'rmse': rmse_scores,
        'mape': mape_scores
    })

    # Format outputs
    output = ref.copy()

    for pred_day in range(1, pred_period + 1):
        header_ref = f'Tp{pred_day}_'
        instance = output_results[[col for col in output_results.columns if header_ref in col or 'mdate_ref' in col]]
        
        # Check for date column before renaming
        date_col = f'{header_ref}mdate_ref'
        if date_col in instance.columns:
            instance = instance.rename(columns={date_col: 'Date'})
            instance = instance.set_index('Date')
        else:
            instance.index.name = 'Index'  # fallback if no date

        output = pd.concat([output, instance[[f'{header_ref}pred']]], axis=1)

    output = output.dropna()
    output1_dates = output_results[[col for col in output_results.columns if 'mdate_ref' in col]]
    output1_values = output[[col for col in output.columns if 'mdate_ref' not in col]]

    return output_results, output1_values, output1_dates, stats_df


# # Load data
# val_df = pd.read_csv("benchmark_C38U>SI_valdf.csv", index_col=0)
# test_df = pd.read_csv("benchmark_C38U.SI_testdf.csv", index_col=0)
# ref = test_df[['yref_Tm0_close']]  # or include mdate if you want dates aligned

# # Run ensemble
# output_results, output1_values, output1_dates, stats_df = ensemble(val_df, test_df, ref)

# # Save outputs
# output_results.to_csv("ensemble_output_results.csv")
# output1_values.to_csv("ensemble_output1_values.csv")
# output1_dates.to_csv("ensemble_output1_dates.csv")
# stats_df.to_csv("ensemble_stats.csv")

# print("✅ All outputs saved.")
import os

tickers = [ 'Q0F.SI', 'S63.SI', 'S68.SI', 'U11.SI']  # Add more tickers as needed
base_path = '/home/priya/Desktop/fyp/Src alwin/Src/data'

for ticker in tickers:
    print(f"\n🔁 Running ensemble for {ticker}")

    # Define file paths
    train_path = os.path.join(base_path, ticker, 'ftraindf.csv')
    test_path = os.path.join(base_path, ticker, 'ftestdf.csv')

    # Load data
    val_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    ref = test_df[['yref_Tm0_close']]

    # Run ensemble model
    output_results, output1_values, output1_dates, stats_df = ensemble(val_df, test_df, ref)

    # Save outputs to a dedicated folder
    output_dir = os.path.join(base_path, ticker, 'ensemble_outputs')
    os.makedirs(output_dir, exist_ok=True)

    output_results.to_csv(os.path.join(output_dir, f'{ticker}_ensemble_output_results.csv'))
    output1_values.to_csv(os.path.join(output_dir, f'{ticker}_ensemble_output1_values.csv'))
    output1_dates.to_csv(os.path.join(output_dir, f'{ticker}_ensemble_output1_dates.csv'))
    stats_df.to_csv(os.path.join(output_dir, f'{ticker}_ensemble_stats.csv'))

    print(f"✅ Outputs saved for {ticker} in {output_dir}")
