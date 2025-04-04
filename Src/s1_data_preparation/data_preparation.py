import pandas as pd
import yfinance as yf

import sys, os 
sys.path.append(os.getcwd())
from s1_data_preparation.config import * 

def retrieve_data(ticker): 
    # process data 
    df = data_processing(ticker)

    # dropna 
    df = df.dropna() 

    # train test split
    traindf = df[df['Date'] < VAL_START] #[df['Date'] >= TRAIN_START]
    valdf = df[df['Date'] >= VAL_START][df['Date'] < TEST_START]
    testdf = df[df['Date'] >= TEST_START][df['Date'] < TEST_END]

    
    return traindf, valdf, testdf

def data_processing(ticker, mode=0):
    df = yf.download(ticker)

    # Fix multi-index column headers
    df.columns = df.columns.droplevel(1)  # Drop the first level ("Price")

    # Reset index to make Date a column
    df.reset_index(inplace=True)

    print("Columns after reset_index:", df.columns)  # Check the columns after reset

    # Ensure that Date is in the columns, not the index
    if 'Date' not in df.columns:
        print("Error: 'Date' column is missing!")
        return None

    # Convert Date to datetime format first
    df['Date'] = pd.to_datetime(df['Date'])

    # Then convert to date (if you specifically want to strip time)
    df['Date'] = df['Date'].dt.date  # Convert to date object

    # Set Date as the index
    df.set_index('Date', inplace=True)

    # Drop any unwanted columns (adjust as needed)
    df = df.drop(columns=['Adj Close', "Price"], errors='ignore')  # Example of dropping a column
    df.to_csv(f'data/{ticker}.csv')  # Saving to CSV if necessary (you can omit this if not needed)

    # Retrieve and process indicators
    df1 = basic_indicators(df)
    
    # Clean up columns and handle exceptions
    try:
        df1 = df1.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1)
    except KeyError:
        pass
    
    return df1

def basic_indicators(df, period=PERIOD, pred_period=PRED_PERIOD):
    # Reset the index and ensure Date is a datetime object
    df = df.reset_index(drop=False)
    
    # Convert 'Date' column to datetime if it's not already in datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Check if there are any missing or invalid dates after conversion
    if df['Date'].isnull().any():
        print("Warning: Some dates could not be converted to datetime.")
    
    # Convert Date to date type (this works now because 'Date' is datetime)
    df['Date'] = df['Date'].dt.date  # Convert to date object
    
    df = df.set_index('Date')
    df['Date'] = df.index

    # Proceed with the rest of the function as before
    df['Volume'] = df['Volume'].replace(0, ZERO_REPLACEMENT)  # replace to prevent division by 0
    df['ROC1'] = df['Close'].pct_change()

    # Creating shifted columns for the past 'period' days
    for index in range(1, period):
        y_vals_price = f'yref_Tp{index}_Price'
        y_vals_date = f'yref_Tp{index}_Date'
        x_vals_price = f'x_Tm{index}_PriceChg'
        x_vals_vol = f'x_Tm{index}_VolChg'
        x_vals_roc1 = f'x_Tm{index}_PRoc1'

        df[f'{x_vals_price}'] = df['Close'].shift(index)
        df[f'{x_vals_vol}'] = df['Volume'].shift(index)
        df[f'{x_vals_roc1}'] = df['ROC1'].shift(index)

        if index <= pred_period:  # to filter for dates within predictive range
            df[f'{y_vals_price}'] = df['Close'].shift(-(index))    
            df[f'{y_vals_date}'] = df['Date'].shift(-(index))    

    # baseline
    df[f'refPrice_Tm{PERIOD}'] = df['Close'].shift(PERIOD)
    df[f'refPrice_Tm{PERIOD}_Date'] = df['Date'].shift(PERIOD)
    df[f'x_Tm{PERIOD}_VolChg'] = df['Volume'].shift(PERIOD)

    # drop irrelevant entries & columns 
    df = df.rename(columns={'Close': 'yref_Tm0_close'})
    df = df.drop(['Volume', 'ROC1'], axis=1)
    df = df.dropna()   

    # NORMALIZATION PROCESS 
    df1 = df.copy()

    dates_col = [col for col in df.columns if 'Date' in col]
    vol_col = [col for col in df.columns if 'Vol' in col]
    vol_ref = df[vol_col]

    dates_ref = df[dates_col]

    df = df[[col for col in df.columns if 'Date' not in col]]

    df1 = df.copy()

    for col in df1.columns: 
        if 'Price' in col and 'Roc' not in col: 
            ref = f'refPrice_Tm{PERIOD}'
            if 'yref_' in col: 
                header = col.replace('yref_', 'y_').replace('_Price', '_PriceChg')
            else: 
                header = col

            if ref == col: 
                continue 
            else: 
                df1[header] = df.apply(lambda x: (x[col]-x[ref])/x[ref], axis=1)              

        elif 'Vol' in col: 
            header = col

            if ref == col: 
                continue 
            else: 
                df1[header] = vol_ref.apply(lambda x: (x[col] - min(x))/(max(x) - min(x)), axis=1)

        elif 'Date' in col or 'PriceRoc' in col: 
            continue

    df1 = df1.drop([f'x_Tm{PERIOD}_VolChg'], axis=1)

    df1 = pd.concat([df1, dates_ref], axis=1)

    return df1


def basic_import(df, ticker, period = PERIOD, pred_period = PRED_PERIOD):

    df = yf.download(ticker)
    df.columns = df.columns.droplevel(1)

    df = df.reset_index(drop = False)
    df['Date'] = df['Date'].dt.date

    df = df.set_index('Date')
    df['Date'] = df.index

    return df



if __name__ == "__main__":
    test_ticker = 'C38U.SI'  # You can change this to any ticker
    
    print("Testing data_processing function...")
    df = data_processing(test_ticker)
    print("Processed Data:")
    print(df.head())

    print("\nTesting retrieve_data function...")
    train_df, val_df, test_df = retrieve_data(test_ticker)
    
    print("\nTrain Data:")
    print(train_df.head())
    print("\nValidation Data:")
    print(val_df.head())
    print("\nTest Data:")
    print(test_df.head())

    print("\nTests completed successfully.")
