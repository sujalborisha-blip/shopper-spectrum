import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    :root {
        --primary: #6366f1;
        --background: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --text-color: #f8fafc;
    }
    
    .stApp {
        background-color: var(--background);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }
    
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(99, 102, 241, 0.2);
    }
    
    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }
    
    /* Input field styling overrides */
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.9);
        color: white;
    }
    .stNumberInput input {
        background-color: rgba(30, 41, 59, 0.9) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Shopper Spectrum Dashboard")
st.markdown("Discover Customer Segments & Personalized Product Recommendations")

def load_models():
    try:
        with open('rfm_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        with open('item_similarity_df.pkl', 'rb') as f:
            item_sim = pickle.load(f)
        return model_data, item_sim
    except FileNotFoundError:
        return None, None

model_data, item_sim = load_models()

if model_data is None:
    st.error("⚠️ Models not found. Please ensure 'train_models.py' has been run successfully to generate the `.pkl` files.")
    st.stop()

kmeans = model_data['kmeans']
scaler = model_data['scaler']
cluster_to_label = model_data['cluster_to_label']

tab1, tab2 = st.tabs(["🎁 Product Recommendations", "👥 Customer Segmentation"])

with tab1:
    st.header("Product Recommendation Engine")
    st.markdown("Select a product to get AI-driven recommendations based on collaborative filtering (Cosine Similarity).")
    
    product_list = item_sim.index.tolist()
    selected_product = st.selectbox("Search for a Product", [""] + product_list)
    
    if st.button("Get Recommendations"):
        if selected_product:
            if selected_product in item_sim.index:
                similar_scores = item_sim[selected_product]
                # Sort and exclude the item itself
                recommendations = similar_scores.sort_values(ascending=False)[1:6]
                
                st.subheader("Top 5 Recommended Products:")
                cols = st.columns(5)
                for i, (prod, score) in enumerate(recommendations.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="font-size: 15px; margin-bottom: 5px; line-height: 1.4;">{prod}</h4>
                            <p style="color:#38bdf8; font-size: 13px; margin: 0; font-weight: 600;">Match Score: {score:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("Product not found in the matrix.")
        else:
            st.warning("Please select a product from the dropdown.")

with tab2:
    st.header("Predict Customer Segment")
    st.markdown("Enter RFM (Recency, Frequency, Monetary) metrics to classify a customer's behavior segment using K-Means Clustering.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (Number of purchases)", min_value=1, value=5, step=1)
    with col3:
        monetary = st.number_input("Monetary (Total Spend $)", min_value=0.0, value=500.0, step=10.0)
        
    if st.button("Predict Cluster"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        # Apply log transformation (same as training pipeline)
        input_log = np.log1p(input_data)
        scaled_input = scaler.transform(input_log)
        
        cluster_id = kmeans.predict(scaled_input)[0]
        segment_label = cluster_to_label[cluster_id]
        
        # Color logic based on segment
        color_map = {
            'High-Value': '#22c55e', # Green
            'Regular': '#3b82f6',    # Blue
            'Occasional': '#eab308', # Yellow
            'At-Risk': '#ef4444'     # Red
        }
        seg_color = color_map.get(segment_label, 'var(--primary)')
        
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-left: 5px solid {seg_color}; padding: 30px;">
            <h3 style="margin-top: 0;">Predicted Customer Segment</h3>
            <div style="font-size: 42px; font-weight: 800; color: {seg_color}; margin: 15px 0;">{segment_label}</div>
            <p style="color: #94a3b8; font-size: 16px; margin-bottom: 0;">Assigned to K-Means Cluster {cluster_id}</p>
        </div>
        """, unsafe_allow_html=True)
