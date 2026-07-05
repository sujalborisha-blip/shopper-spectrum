import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Starting Data Preprocessing and Modeling...")

# Load dataset
print("Loading Data...")
df = pd.read_csv('dataset.csv', encoding='unicode_escape')

# --- 1. Data Cleaning ---
print("Cleaning Data...")
# Remove rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative/zero quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# --- 2. RFM Feature Engineering ---
print("Extracting RFM Features...")
# Current date for Recency (latest date in dataset + 1 day)
latest_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by CustomerID
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# --- 3. Clustering ---
print("Training KMeans Model...")
# Select features
features = rfm[['Recency', 'Frequency', 'Monetary']]

# Apply log transformation to handle skewness and outliers
features_log = np.log1p(features)

# Standardize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(features_log)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(rfm_scaled)
rfm['Cluster'] = kmeans.labels_

# Map clusters logically based on average RFM
cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
# Rank by Monetary value to assign segments
monetary_ranks = cluster_avg['Monetary'].rank(method='first').to_dict()
label_map = {
    4: 'High-Value',
    3: 'Regular',
    2: 'Occasional',
    1: 'At-Risk'
}
cluster_to_label = {cluster: label_map[rank] for cluster, rank in monetary_ranks.items()}

# Save model data
model_data = {
    'kmeans': kmeans,
    'scaler': scaler,
    'cluster_to_label': cluster_to_label
}

with open('rfm_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# --- 4. Collaborative Filtering ---
print("Generating Item Similarity Matrix...")
# Keep top 1000 items to avoid memory issues and focus on popular items
top_items = df['Description'].value_counts().head(1000).index
df_top_items = df[df['Description'].isin(top_items)]

# User-Item Matrix
user_item_matrix = df_top_items.pivot_table(
    index='CustomerID', 
    columns='Description', 
    values='Quantity', 
    aggfunc='sum', 
    fill_value=0
)

# Convert to 1/0 for bought/not-bought
user_item_matrix = user_item_matrix.map(lambda x: 1 if x > 0 else 0)

# Calculate Cosine Similarity
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Save Similarity Matrix
with open('item_similarity_df.pkl', 'wb') as f:
    pickle.dump(item_similarity_df, f)

print("Data Preprocessing and Modeling Completed successfully!")
