# import numpy as np
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
# from tensorflow.keras import layers, models

# # Define the Generator
# def build_generator(latent_dim):
#     model = models.Sequential([
#         layers.Dense(128, input_dim=latent_dim),
#         layers.LeakyReLU(alpha=0.2),
#         layers.BatchNormalization(),
#         layers.Dense(256),
#         layers.LeakyReLU(alpha=0.2),
#         layers.BatchNormalization(),
#         layers.Dense(512),
#         layers.LeakyReLU(alpha=0.2),
#         layers.BatchNormalization(),
#         layers.Dense(1, activation='tanh')  # Output synthetic price data
#     ])
#     return model

# # Define the Critic (Discriminator in WGAN)
# def build_critic():
#     model = models.Sequential([
#         layers.Dense(512, input_shape=(1,)),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Dense(256),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Dense(128),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Dense(1)  # Output a score for Wasserstein distance
#     ])
#     return model

# # WGAN Training Loop
# def train_wgan(generator, critic, dataset, latent_dim, n_epochs=5000, batch_size=32, n_critic=5):
#     critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
#     generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

#     for epoch in range(n_epochs):
#         for _ in range(n_critic):
#             # Sample real data
#             real_data = dataset[np.random.randint(0, dataset.shape[0], batch_size)]
#             real_data = real_data.reshape(-1, 1).astype(np.float32)  # Ensure correct shape
            
#             # Generate fake data
#             noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
#             fake_data = generator(noise, training=True)
            
#             # Train critic
#             with tf.GradientTape() as tape:
#                 real_output = critic(real_data, training=True)
#                 fake_output = critic(fake_data, training=True)
#                 critic_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

#             critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
#             critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

#             # Clip critic weights correctly
#             for var in critic.trainable_variables:
#                 if "kernel" in var.name:
#                     var.assign(tf.clip_by_value(var, -0.01, 0.01))

#         # Train generator
#         noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
#         with tf.GradientTape() as tape:
#             fake_data = generator(noise, training=True)
#             fake_output = critic(fake_data, training=True)
#             generator_loss = -tf.reduce_mean(fake_output)

#         generator_gradients = tape.gradient(generator_loss, generator.trainable_variables)
#         generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

#         # Print progress every 100 epochs
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch} - Critic Loss: {critic_loss.numpy():.6f}, Generator Loss: {generator_loss.numpy():.6f}")

#             # Debugging: Check a few sample generated values
#             sample_noise = np.random.normal(0, 1, (5, latent_dim)).astype(np.float32)
#             sample_fake_data = generator(sample_noise, training=False).numpy()
#             print(f"Sample Generated Data: {sample_fake_data.reshape(-1)}")

# # Example usage
# latent_dim = 100
# dataset = np.random.normal(0, 1, (10000, 1))  # Replace with real financial data
# dataset = (dataset - np.mean(dataset)) / np.std(dataset)  # Normalize data
# generator = build_generator(latent_dim)
# critic = build_critic()
# train_wgan(generator, critic, dataset, latent_dim)


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import os

# ---------------------- Load Stock Data ----------------------
stock_symbols = {
    "O39.SI": "OCBC Bank",
    "C38U.SI": "CapitaLand Integrated Commercial Trust",
    "Q0F.SI": "IHH Healthcare",
    "S68.SI": "SGX",
    "C6L.SI": "Singapore Airlines",
    "G07.SI": "Great Eastern",
}

all_stock_data = {}
for symbol in stock_symbols:
    df = yf.Ticker(symbol).history(period="10y")[["Close"]].dropna()
    df["RMA"] = df["Close"].ewm(alpha=1/14, adjust=False).mean()
    df["Returns"] = df["Close"].pct_change()
    df["Year"] = df.index.year
    df["Peak"] = df["RMA"] == df["RMA"].rolling(15, center=True).max()
    df["Trough"] = df["RMA"] == df["RMA"].rolling(15, center=True).min()
    all_stock_data[symbol] = df.dropna()

# ---------------------- GA Setup for Fundamentals ----------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

stocks = list(all_stock_data.keys())
df_fundamentals = pd.DataFrame({
    "Stock": stocks,
    "PE": np.random.rand(len(stocks)) * 30,
    "Risk": np.random.rand(len(stocks)) * 0.3,
    "Growth": np.random.rand(len(stocks)),
    "Health": np.random.rand(len(stocks))
})
scaler = MinMaxScaler()
df_fundamentals.iloc[:, 1:] = scaler.fit_transform(df_fundamentals.iloc[:, 1:])

def calculate_scores(df, weights):
    df = df.copy()
    df["Score"] = (
        weights["PE"] * (1 / df["PE"]) +
        weights["Risk"] * (1 / df["Risk"]) +
        weights["Growth"] * df["Growth"] +
        weights["Health"] * df["Health"]
    )
    df["Normalized_Score"] = df["Score"] / df["Score"].sum()
    return df

def evaluate(individual):
    weights = {
        "PE": individual[0],
        "Risk": -individual[1],
        "Growth": individual[2],
        "Health": individual[3]
    }
    scored = calculate_scores(df_fundamentals, weights)
    return scored["Normalized_Score"].sum(),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=50)
NGEN = 10
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

top_ind = tools.selBest(population, k=1)[0]
optimized_weights = {
    "PE": top_ind[0],
    "Risk": -top_ind[1],
    "Growth": top_ind[2],
    "Health": top_ind[3]
}

scores_df = calculate_scores(df_fundamentals, optimized_weights)

# ---------------------- LSTM Forecast ----------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = tf.keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------------- Forecast and Allocate ----------------------
alloc_by_year = {}
for symbol in stocks:
    df = all_stock_data[symbol]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])
    df["Scaled_Close"] = scaled
    X, y = create_sequences(scaled, 5)
    model = build_model((5, 1))
    model.fit(X, y, epochs=100, batch_size=16, verbose=0)
    preds_scaled = []
    input_seq = scaled[-5:].reshape(1, 5, 1)
    for _ in range(30):
        pred = model.predict(input_seq, verbose=0)
        preds_scaled.append(pred[0, 0])
        input_seq = np.roll(input_seq, -1)
        input_seq[0, -1, 0] = pred
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # Annual Allocation with Forecast Impact
    norm_score = scores_df[scores_df["Stock"] == symbol]["Normalized_Score"].values[0]
    for year in df["Year"].unique():
        year_df = df[df["Year"] == year]
        peak_factor = year_df["Peak"].sum()
        trough_factor = year_df["Trough"].sum()
        forecast_adjustment = np.mean(preds) / df["Close"].iloc[-30:].mean()
        score = norm_score * (1 + 0.2 * peak_factor + 0.1 * trough_factor) * forecast_adjustment
        if year not in alloc_by_year:
            alloc_by_year[year] = {}
        alloc_by_year[year][symbol] = score

# Normalize Allocation
for year in alloc_by_year:
    total = sum(alloc_by_year[year].values())
    for symbol in alloc_by_year[year]:
        alloc_by_year[year][symbol] /= total

# ---------------------- Plotting ----------------------
os.makedirs("plots", exist_ok=True)
for symbol in stocks:
    years = sorted(alloc_by_year.keys())
    values = [alloc_by_year[y].get(symbol, 0) for y in years]
    plt.figure(figsize=(8, 4))
    plt.plot(years, values, marker='o')
    plt.title(f"{stock_symbols[symbol]} ({symbol}) Annual Allocation")
    plt.xlabel("Year")
    plt.ylabel("Allocation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{symbol}_allocation.png")
    plt.close()

print("\nFinal Optimized Weights from GA:")
print(optimized_weights)
print("\nYearly Allocation (Peaks + Forecasting):")
for year in sorted(alloc_by_year):
    print(f"\nYear {year}:")
    for symbol, value in alloc_by_year[year].items():
        print(f"  {stock_symbols[symbol]} ({symbol}): {value:.4f}")
print("\nSaved plots to 'plots/' folder.")
