import pytest
import pickle
import numpy as np
import pandas as pd

# Load the models globally for testing
@pytest.fixture(scope="module")
def rfm_model():
    with open('rfm_model.pkl', 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope="module")
def item_sim_df():
    with open('item_similarity_df.pkl', 'rb') as f:
        return pickle.load(f)

# --- K-Means RFM Testing (Clustering Logic) ---

def test_1_centroid_dimensionality(rfm_model):
    """1. Verify K-Means has exactly 4 clusters in a 3-dimensional space."""
    kmeans = rfm_model['kmeans']
    assert kmeans.n_clusters == 4
    assert kmeans.cluster_centers_.shape == (4, 3)

def test_2_scaler_transformation_validity(rfm_model):
    """2. Programmatically test StandardScaler combined with log1p."""
    scaler = rfm_model['scaler']
    # Extreme low vs extreme high input
    low_input = pd.DataFrame([[1, 1, 1]], columns=['Recency', 'Frequency', 'Monetary'])
    high_input = pd.DataFrame([[1000, 1000, 100000]], columns=['Recency', 'Frequency', 'Monetary'])
    
    low_scaled = scaler.transform(np.log1p(low_input))[0]
    high_scaled = scaler.transform(np.log1p(high_input))[0]
    
    # High input should have higher mathematically scaled scores than low input
    assert all(high_scaled > low_scaled)

def test_3_high_value_logical_constraint(rfm_model):
    """3. Assert a perfect high-value matrix correctly maps to 'High-Value'."""
    kmeans = rfm_model['kmeans']
    scaler = rfm_model['scaler']
    label_map = rfm_model['cluster_to_label']
    
    inp = pd.DataFrame([[1, 500, 50000.0]], columns=['Recency', 'Frequency', 'Monetary'])
    scaled = scaler.transform(np.log1p(inp))
    pred = kmeans.predict(scaled)[0]
    
    assert label_map[pred] == 'High-Value'

def test_4_at_risk_logical_constraint(rfm_model):
    """4. Assert a decaying profile correctly maps to 'At-Risk'."""
    kmeans = rfm_model['kmeans']
    scaler = rfm_model['scaler']
    label_map = rfm_model['cluster_to_label']
    
    inp = pd.DataFrame([[365, 1, 5.0]], columns=['Recency', 'Frequency', 'Monetary'])
    scaled = scaler.transform(np.log1p(inp))
    pred = kmeans.predict(scaled)[0]
    
    assert label_map[pred] == 'At-Risk'

def test_5_label_mapping_integrity(rfm_model):
    """5. Ensure cluster dictionary possesses exactly 4 distinct keys and values."""
    label_map = rfm_model['cluster_to_label']
    assert len(label_map) == 4
    assert set(label_map.keys()) == {0, 1, 2, 3}
    assert set(label_map.values()) == {'High-Value', 'Regular', 'Occasional', 'At-Risk'}

# --- Cosine Similarity Testing (Recommendation Flow) ---

def test_6_cosine_reflexivity(item_sim_df):
    """6. Algorithmically verify self-similarity score is perfectly 1.0."""
    # Diagonal should be entirely 1.0 (with a tiny floating point tolerance)
    diagonal = np.diag(item_sim_df)
    assert np.allclose(diagonal, 1.0, atol=1e-5)

def test_7_cosine_symmetry_validation(item_sim_df):
    """7. Ensure Similarity(A->B) == Similarity(B->A)."""
    # Pick two random products
    product_a = item_sim_df.index[0]
    product_b = item_sim_df.columns[1]
    
    score_ab = item_sim_df.loc[product_a, product_b]
    score_ba = item_sim_df.loc[product_b, product_a]
    assert np.isclose(score_ab, score_ba, atol=1e-5)

def test_8_recommendation_count_assertion(item_sim_df):
    """8. Query matrix and assert algorithm outputs precisely 5 limits."""
    product = item_sim_df.index[0]
    scores = item_sim_df[product].sort_values(ascending=False)[1:6]
    assert len(scores) == 5

def test_9_self_exclusion_rule(item_sim_df):
    """9. Validate top recommendations exclude the queried item itself."""
    product = item_sim_df.index[0]
    # Get top 5 recommendations
    recommendations = item_sim_df[product].sort_values(ascending=False)[1:6].index.tolist()
    assert product not in recommendations

def test_10_vector_bound_constraints(item_sim_df):
    """10. Scan matrix and assert no values fall outside 0.0 to 1.0 boundaries."""
    # Checking for values < -0.01 or > 1.01 to allow minor floating point errors
    out_of_bounds = (item_sim_df < -0.01) | (item_sim_df > 1.01)
    assert not out_of_bounds.any().any(), "Found Cosine Similarity score outside mathematical boundaries."
