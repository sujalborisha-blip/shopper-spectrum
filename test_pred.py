import pickle
import numpy as np
import pandas as pd
with open('rfm_model.pkl', 'rb') as f: model = pickle.load(f)
kmeans = model['kmeans']
scaler = model['scaler']
print('Cluster to Label:', model['cluster_to_label'])
for r, f, m in [[30, 5, 500], [1, 50, 5000], [300, 1, 10], [15, 10, 1000]]:
    inp = pd.DataFrame([[r, f, m]], columns=['Recency', 'Frequency', 'Monetary'])
    inp_log = np.log1p(inp)
    scaled = scaler.transform(inp_log)
    pred = kmeans.predict(scaled)[0]
    label = model['cluster_to_label'][pred]
    print(f'Input: R={r} F={f} M={m} -> Scaled: {scaled} -> Cluster: {pred} -> Label: {label}')
