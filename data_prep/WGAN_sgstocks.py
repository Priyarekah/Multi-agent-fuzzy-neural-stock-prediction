import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# Define stock symbols with their names
stock_symbols = {
    "D05.SI": "DBS Group",
    "O39.SI": "OCBC Bank",
    "U11.SI": "UOB Bank",
    "Z74.SI": "SingTel",
    "F34.SI": "Wilmar International",
    "Q0F.SI": "IHH Healthcare",
    "C6L.SI": "Singapore Airlines",
    "S63.SI": "ST Engineering",
    "J36.SI": "Jardine Matheson",
    "S68.SI": "SGX",
    "C38U.SI": "CapitaLand Integrated Commercial Trust",
    "Y92.SI": "Thai Beverage",
    "H78.SI": "Hongkong Land",
    "9CI.SI": "CapitaLand Investment",
    "G07.SI": "Great Eastern",
    "BN4.SI": "Keppel Corp",
    "BS6.SI": "Yangzijiang Shipbuilding",
    "A17U.SI": "Ascendas REIT",
    "C07.SI": "Jardine C&C",
    "U96.SI": "Sembcorp Industries",
    "G13.SI": "Genting Singapore",
    "5E2.SI": "Seatrium",
    "N2IU.SI": "Mapletree Pan Asia Commercial Trust",
    "M44U.SI": "Mapletree Logistics Trust",
    "ME8U.SI": "Mapletree Industrial Trust",
    "S58.SI": "SATS",
    "AJBU.SI": "Keppel DC REIT"
}

# Function to fetch historical data and calculate peaks and troughs
def get_peak_trough_correlation(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="10y")["Close"]  # Change to 10 years
    peaks = hist[hist == hist.rolling(window=15, center=True).max()]  # Change to 15 days
    troughs = hist[hist == hist.rolling(window=15, center=True).min()]
    
    # Debugging: Print length of peaks and troughs
    print(f"{symbol} - Peaks: {len(peaks.dropna())}, Troughs: {len(troughs.dropna())}")
    
    return peaks.dropna(), troughs.dropna()

# Get all peaks and troughs
all_peaks = {}
all_troughs = {}

for symbol in stock_symbols.keys():
    peaks, troughs = get_peak_trough_correlation(symbol)
    if not peaks.empty and not troughs.empty:
        all_peaks[symbol] = peaks
        all_troughs[symbol] = troughs

# Calculate correlations
correlation_matrix = pd.DataFrame(index=all_peaks.keys(), columns=all_peaks.keys())

for sym1 in all_peaks.keys():
    for sym2 in all_peaks.keys():
        if sym1 != sym2:
            # Combine peaks and troughs into one series for correlation
            combined1 = pd.concat([all_peaks[sym1], all_troughs[sym1]]).sort_index()
            combined2 = pd.concat([all_peaks[sym2], all_troughs[sym2]]).sort_index()
            min_len = min(len(combined1), len(combined2))
            if min_len > 1:  # Ensure enough data points
                correlation, _ = pearsonr(combined1[:min_len], combined2[:min_len])
                correlation_matrix.loc[sym1, sym2] = correlation

# Debugging: Print correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Select stocks with low correlation
selected_stocks = []
for sym1 in correlation_matrix.index:
    if len(selected_stocks) < 10:
        correlations = correlation_matrix.loc[sym1].dropna().astype(float)
        if all(correlations < 0.85):  # Change from 0.7 to 0.8
            selected_stocks.append(sym1)

# Debugging: Print selected stocks
print("\nSelected Stocks with Low Correlation:")
print(selected_stocks)

# Fetch metrics for the selected stocks
def fetch_stock_metrics(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    try:
        metrics = {
            "Stock": symbol,
            "Name": stock_symbols[symbol],
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "P/B Ratio": info.get("priceToBook", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Debt-to-Equity": info.get("debtToEquity", "N/A"),
            "Earnings Growth": info.get("earningsGrowth", "N/A"),
        }
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        metrics = {
            "Stock": symbol,
            "Name": stock_symbols[symbol],
            "P/E Ratio": "N/A",
            "P/B Ratio": "N/A",
            "Dividend Yield": "N/A",
            "Debt-to-Equity": "N/A",
            "Earnings Growth": "N/A",
        }
    return metrics

# Fetch metrics and create DataFrame
stock_data = []
for symbol in selected_stocks:
    metrics = fetch_stock_metrics(symbol)
    stock_data.append(metrics)

df = pd.DataFrame(stock_data)

# Drop rows with any 'N/A' values
df = df.replace("N/A", np.nan).dropna()

# Convert Dividend Yield from decimal to percentage
df["Dividend Yield"] = df["Dividend Yield"].apply(lambda x: round(float(x) * 100, 2) if pd.notna(x) else "N/A")

# Display and save the DataFrame
print("\nFiltered Stock Metrics Data:")
print(df)

df.to_csv("filtered_singapore_stock_metrics.csv", index=False)
print("\nData saved to 'filtered_singapore_stock_metrics.csv'.")

# Preprocess the filtered data for WGAN
# Use numerical columns and normalize them
numeric_columns = ["P/E Ratio", "P/B Ratio", "Dividend Yield", "Debt-to-Equity", "Earnings Growth"]
data = df[numeric_columns].values.astype(np.float32)

# Normalize the data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Define the Generator
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(128, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(data.shape[1], activation='tanh')  # Output synthetic price data
    ])
    return model

# Define the Critic (Discriminator in WGAN)
def build_critic():
    model = models.Sequential([
        layers.Dense(512, input_shape=(data.shape[1],)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1)  # Output a score for Wasserstein distance
    ])
    return model

# WGAN Training Loop
def train_wgan(generator, critic, dataset, latent_dim, n_epochs=5000, batch_size=32, n_critic=5):
    critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

    for epoch in range(n_epochs):
        for _ in range(n_critic):
            # Sample real data
            real_data = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
            
            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
            fake_data = generator(noise, training=True)
            
            # Train critic
            with tf.GradientTape() as tape:
                real_output = critic(real_data, training=True)
                fake_output = critic(fake_data, training=True)
                critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

            critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

            # Clip critic weights correctly
            for var in critic.trainable_variables:
                if "kernel" in var.name:
                    var.assign(tf.clip_by_value(var, -0.01, 0.01))

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        with tf.GradientTape() as tape:
            fake_data = generator(noise, training=True)
            fake_output = critic(fake_data, training=True)
            generator_loss = -tf.reduce_mean(fake_output)

        generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - Critic Loss: {critic_loss.numpy():.6f}, Generator Loss: {generator_loss.numpy():.6f}")

            # Debugging: Check a few sample generated values
            sample_noise = np.random.normal(0, 1, (5, latent_dim)).astype(np.float32)
            sample_fake_data = generator(sample_noise, training=False).numpy()
            print(f"Sample Generated Data: {sample_fake_data.reshape(-1)}")

# Example usage
latent_dim = 100
generator = build_generator(latent_dim)
critic = build_critic()
train_wgan(generator, critic, data, latent_dim)

# import yfinance as yf
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr

# # Define stock symbols with their names
# stock_symbols = {
#     "D05.SI": "DBS Group", "O39.SI": "OCBC Bank", "U11.SI": "UOB Bank",
#     "Z74.SI": "SingTel", "F34.SI": "Wilmar International", "Q0F.SI": "IHH Healthcare",
#     "C6L.SI": "Singapore Airlines", "S63.SI": "ST Engineering", "J36.SI": "Jardine Matheson",
#     "S68.SI": "SGX", "C38U.SI": "CapitaLand Integrated Commercial Trust", "Y92.SI": "Thai Beverage",
#     "H78.SI": "Hongkong Land", "9CI.SI": "CapitaLand Investment", "G07.SI": "Great Eastern",
#     "BN4.SI": "Keppel Corp", "BS6.SI": "Yangzijiang Shipbuilding", "A17U.SI": "Ascendas REIT",
#     "C07.SI": "Jardine C&C", "U96.SI": "Sembcorp Industries", "G13.SI": "Genting Singapore",
#     "5E2.SI": "Seatrium", "N2IU.SI": "Mapletree Pan Asia Commercial Trust", "M44U.SI": "Mapletree Logistics Trust",
#     "ME8U.SI": "Mapletree Industrial Trust", "S58.SI": "SATS", "AJBU.SI": "Keppel DC REIT"
# }

# # Function to fetch historical data and calculate peaks and troughs
# def get_peak_trough_correlation(symbol):
#     stock = yf.Ticker(symbol)
#     hist = stock.history(period="10y")["Close"]  
#     if hist.empty:
#         return pd.Series(dtype=float), pd.Series(dtype=float)
    
#     peaks = hist[hist == hist.rolling(window=15, center=True).max()].dropna()
#     troughs = hist[hist == hist.rolling(window=15, center=True).min()].dropna()
    
#     print(f"{symbol} - Peaks: {len(peaks)}, Troughs: {len(troughs)}")
    
#     return peaks, troughs

# # Get all peaks and troughs
# all_peaks = {}
# all_troughs = {}

# for symbol in stock_symbols.keys():
#     peaks, troughs = get_peak_trough_correlation(symbol)
#     if not peaks.empty and not troughs.empty:
#         all_peaks[symbol] = peaks
#         all_troughs[symbol] = troughs

# # Ensure at least some valid data exists
# if not all_peaks:
#     print("No valid data collected. Exiting...")
#     exit()

# # Calculate correlations with added weightage for peaks
# correlation_matrix = pd.DataFrame(index=all_peaks.keys(), columns=all_peaks.keys(), dtype=float)

# for sym1 in all_peaks.keys():
#     for sym2 in all_peaks.keys():
#         if sym1 != sym2:
#             combined1 = pd.concat([all_peaks[sym1], all_troughs[sym1]]).sort_index()
#             combined2 = pd.concat([all_peaks[sym2], all_troughs[sym2]]).sort_index()
            
#             common_index = combined1.index.intersection(combined2.index)
#             if len(common_index) > 2:  
#                 combined1 = combined1.loc[common_index]
#                 combined2 = combined2.loc[common_index]
                
#                 weights1 = np.where(combined1.index.isin(all_peaks[sym1].index), 2, 1)
#                 weights2 = np.where(combined2.index.isin(all_peaks[sym2].index), 2, 1)
                
#                 weighted_combined1 = combined1 * weights1
#                 weighted_combined2 = combined2 * weights2
                
#                 try:
#                     correlation, _ = pearsonr(weighted_combined1, weighted_combined2)
#                     correlation_matrix.loc[sym1, sym2] = correlation
#                 except ValueError:
#                     correlation_matrix.loc[sym1, sym2] = np.nan

# # Print correlation matrix
# print("\nCorrelation Matrix:")
# print(correlation_matrix)

# # Fetch stock earnings growth
# earnings_growth_data = []

# for symbol in stock_symbols.keys():
#     stock = yf.Ticker(symbol)
#     try:
#         growth = stock.info.get("earningsGrowth", np.nan)
#         earnings_growth_data.append({"Stock": symbol, "Earnings Growth": growth})
#     except Exception as e:
#         print(f"Error fetching earnings growth for {symbol}: {e}")

# df = pd.DataFrame(earnings_growth_data).dropna()

# # Select stocks with low correlation and higher earnings growth
# selected_stocks = []
# for sym1 in correlation_matrix.index:
#     if len(selected_stocks) < 10:
#         correlations = correlation_matrix.loc[sym1].dropna().astype(float)
        
#         earnings_growth = df.loc[df["Stock"] == sym1, "Earnings Growth"].values
#         if len(earnings_growth) > 0 and all(correlations < 0.90):
#             selected_stocks.append(sym1)

# print("\nSelected Stocks with Low Correlation and Higher Earnings Growth:")
# print(selected_stocks)

# # Fetch metrics for the selected stocks
# def fetch_stock_metrics(symbol):
#     stock = yf.Ticker(symbol)
#     info = stock.info
#     try:
#         metrics = {
#             "Stock": symbol,
#             "Name": stock_symbols[symbol],
#             "P/E Ratio": info.get("trailingPE", np.nan),
#             "P/B Ratio": info.get("priceToBook", np.nan),
#             "Dividend Yield": info.get("dividendYield", np.nan),
#             "Debt-to-Equity": info.get("debtToEquity", np.nan),
#             "Earnings Growth": info.get("earningsGrowth", np.nan),
#         }
#     except Exception as e:
#         print(f"Error fetching data for {symbol}: {e}")
#         metrics = {key: np.nan for key in ["Stock", "Name", "P/E Ratio", "P/B Ratio", "Dividend Yield", "Debt-to-Equity", "Earnings Growth"]}
#         metrics["Stock"] = symbol
#         metrics["Name"] = stock_symbols[symbol]
    
#     return metrics

# # Fetch metrics and create DataFrame
# stock_data = [fetch_stock_metrics(symbol) for symbol in selected_stocks]
# df_metrics = pd.DataFrame(stock_data).dropna()

# # Convert Dividend Yield from decimal to percentage
# df_metrics["Dividend Yield"] = df_metrics["Dividend Yield"].apply(lambda x: round(float(x) * 100, 2) if pd.notna(x) else np.nan)

# # Display and save the DataFrame
# print("\nFiltered Stock Metrics Data:")
# print(df_metrics)

# df_metrics.to_csv("filtered_singapore_stock_metrics.csv", index=False)
# print("\nData saved to 'filtered_singapore_stock_metrics.csv'.")
