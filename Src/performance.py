import pandas as pd

# Define paths to your CSV files (make sure these paths are correct)
file_paths = [
    "/home/priya/Desktop/fyp/Src alwin/Src/performance_summary_Tp1.csv",
    "/home/priya/Desktop/fyp/Src alwin/Src/performance_summary_Tp13.csv",
    "/home/priya/Desktop/fyp/Agents/dl/performance_summary_Tp1.csv",
    "/home/priya/Desktop/fyp/Agents/performance_summary_Tp1.csv"
]

# Read and combine all CSVs
dfs = []
for path in file_paths:
    try:
        df = pd.read_csv(path)
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {path}")

# Combine all into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

# Sort by test_r2 in descending order and extract top 5
top_5 = combined_df.sort_values(by="test_r2", ascending=False).head(5)

# Display top 5 configurations
print("\nTop 5 Configurations Based on Test R² Score:")
print(top_5[["configuration", "val_r2", "val_rmse", "test_r2", "test_rmse"]])

# Optional: Save to CSV
top_5.to_csv("top5_transformer_configurations.csv", index=False)


import ast
import pandas as pd
# Re-import and re-define needed variables after reset
import ast

# Load previously uploaded top 5 transformer configuration file
top5_df = pd.read_csv('/home/priya/Desktop/fyp/Src alwin/top5_transformer_configurations.csv')

# Extract relevant hyperparameters
def extract_transformer_params(row):
    config = ast.literal_eval(row['configuration'])
    transformer = config.get('transformer', {})
    return pd.Series({
        'encoder_dim': transformer.get('d_model'),
        'hidden_layers': transformer.get('hidden_layers'),
        'activation': transformer.get('activation'),
        'hidden_dim': transformer.get('dim_feedforward'),
        'batch_size': transformer.get('batch_size'),
        'val_rmse': row['val_rmse'],
        'test_rmse': row['test_rmse']
    })

# Apply to top 5 configurations
summary_table = top5_df.apply(extract_transformer_params, axis=1)
summary_table.index = [f'Trial {i+1}' for i in range(len(summary_table))]

# Prepare numeric values for heatmap
heatmap_data = summary_table[['encoder_dim', 'hidden_layers', 'hidden_dim', 'batch_size', 'val_rmse', 'test_rmse']].copy()
heatmap_data = heatmap_data.apply(pd.to_numeric, errors='coerce')

# Plot heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Hyperparameter Heatmap of Top 5 Transformer Configurations")
plt.tight_layout()
plt.show()
