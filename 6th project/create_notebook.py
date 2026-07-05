import nbformat as nbf

nb = nbf.v4.new_notebook()

text_1 = """\
# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations
## 📌 Project Overview
This notebook contains the complete **Data Science and Machine Learning** pipeline for the Shopper Spectrum E-Commerce project.
The deliverables include:
* **Data Cleaning & Feature Engineering**
* **Exploratory Data Analysis (EDA) & Visualizations**
* **RFM Customer Segmentation (K-Means Clustering) with Evaluation**
* **Collaborative Filtering for Product Recommendations**
"""

code_1 = """\
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (10, 6)
"""

text_2 = """\
## 1️⃣ Data Collection and Understanding
Load the dataset and preview its structure.
"""

code_2 = """\
# Load dataset
df = pd.read_csv('dataset.csv', encoding='unicode_escape')
display(df.head())
display(df.info())
"""

text_3 = """\
## 2️⃣ Data Preprocessing
* Remove missing CustomerIDs
* Exclude cancelled invoices
* Remove negative quantities and prices
"""

code_3 = """\
# Remove missing CustomerID
df_clean = df.dropna(subset=['CustomerID']).copy()

# Remove cancelled invoices (starting with 'C')
df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]

# Remove negative/zero quantities and prices
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

# Convert Date
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")
"""

text_4 = """\
## 3️⃣ Exploratory Data Analysis (EDA)
### Transaction Volume by Country
"""

code_4 = """\
top_countries = df_clean['Country'].value_counts().head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
plt.title('Top 10 Countries by Transaction Volume')
plt.xlabel('Number of Transactions')
plt.ylabel('Country')
plt.show()
"""

text_5 = """\
### Top Selling Products
"""

code_5 = """\
top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_products.values, y=top_products.index, palette='magma')
plt.title('Top 10 Best-Selling Products by Quantity')
plt.xlabel('Total Quantity Sold')
plt.ylabel('Product')
plt.show()
"""

text_6 = """\
## 4️⃣ RFM Feature Engineering
Calculate Recency, Frequency, and Monetary metrics for Customer Segmentation.
"""

code_6 = """\
latest_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

display(rfm.head())
"""

text_7 = """\
## 5️⃣ Customer Segmentation (K-Means Clustering)
We will use the Elbow Method and Silhouette Score to evaluate the optimal number of clusters.
"""

code_7 = """\
# Apply log transformation to handle right-skewness and extreme outliers
features = rfm[['Recency', 'Frequency', 'Monetary']]
features_log = np.log1p(features)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(features_log)

# Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeanModel.fit(rfm_scaled)
    inertia.append(kmeanModel.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Inertia')
plt.title('The Elbow Method showing the optimal k')
plt.show()
"""

code_8 = """\
# Train the final KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Model Evaluation: Silhouette Score
sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
print(f"Silhouette Score for k=4: {sil_score:.4f}")

# Map clusters logically based on Monetary value
cluster_avg = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
monetary_ranks = cluster_avg['Monetary'].rank(method='first').to_dict()
label_map = {4: 'High-Value', 3: 'Regular', 2: 'Occasional', 1: 'At-Risk'}
rfm['Segment'] = rfm['Cluster'].map(lambda x: label_map[monetary_ranks[x]])

display(rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2))
"""

text_9 = """\
## 6️⃣ Product Recommendation System
Using Item-based Collaborative Filtering to recommend products based on Cosine Similarity.
"""

code_9 = """\
# Select top 1000 items to avoid memory issues
top_items = df_clean['Description'].value_counts().head(1000).index
df_top = df_clean[df_clean['Description'].isin(top_items)]

# User-Item Matrix
user_item_matrix = df_top.pivot_table(index='CustomerID', columns='Description', values='Quantity', aggfunc='sum', fill_value=0)
user_item_matrix = user_item_matrix.map(lambda x: 1 if x > 0 else 0)

# Cosine Similarity
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def recommend_products(product_name, num_recommendations=5):
    if product_name not in item_similarity_df.index:
        return "Product not found."
    scores = item_similarity_df[product_name].sort_values(ascending=False)[1:num_recommendations+1]
    return scores

# Test the recommendation engine
test_product = item_similarity_df.index[0]
print(f"Recommendations for '{test_product}':")
print(recommend_products(test_product))
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(text_2),
    nbf.v4.new_code_cell(code_2),
    nbf.v4.new_markdown_cell(text_3),
    nbf.v4.new_code_cell(code_3),
    nbf.v4.new_markdown_cell(text_4),
    nbf.v4.new_code_cell(code_4),
    nbf.v4.new_markdown_cell(text_5),
    nbf.v4.new_code_cell(code_5),
    nbf.v4.new_markdown_cell(text_6),
    nbf.v4.new_code_cell(code_6),
    nbf.v4.new_markdown_cell(text_7),
    nbf.v4.new_code_cell(code_7),
    nbf.v4.new_code_cell(code_8),
    nbf.v4.new_markdown_cell(text_9),
    nbf.v4.new_code_cell(code_9)
]

with open('Shopper_Spectrum_Notebook.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
    
print("Notebook 'Shopper_Spectrum_Notebook.ipynb' restored successfully for ML workflow!")
