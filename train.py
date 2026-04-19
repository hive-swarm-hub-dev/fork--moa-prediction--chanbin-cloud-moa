"""
Logistic Regression with PCA features.
LightGBM overfits badly on sparse targets (pred_max=1.0), blowing up log loss.
Logistic regression is better calibrated for this sparse multi-label problem.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import time

start = time.time()

# Load data
train_features = pd.read_csv("data/train_features.csv")
train_targets = pd.read_csv("data/train_targets.csv")
test_features = pd.read_csv("data/test_features.csv")

target_cols = [c for c in train_targets.columns if c != "sig_id"]
gene_cols = [c for c in train_features.columns if c.startswith("g-")]
cell_cols = [c for c in train_features.columns if c.startswith("c-")]

# Fit PCA on combined train+test (more data = better PCA)
all_feats = pd.concat([train_features, test_features], ignore_index=True)
n_train = len(train_features)

pca_g = PCA(n_components=80, random_state=42)
pca_c = PCA(n_components=30, random_state=42)
g_all = pca_g.fit_transform(all_feats[gene_cols].values)
c_all = pca_c.fit_transform(all_feats[cell_cols].values)

# Encode categorical features
cp_time_map = {24: 0.0, 48: 0.5, 72: 1.0}
cp_dose_map = {"D1": 0.0, "D2": 1.0}
cp_type_bin = (all_feats["cp_type"].values == "trt_cp").astype(float)
cp_time_enc = np.array([cp_time_map[t] for t in all_feats["cp_time"].values])
cp_dose_enc = np.array([cp_dose_map[d] for d in all_feats["cp_dose"].values])

X_all = np.hstack([
    cp_type_bin.reshape(-1, 1),
    cp_time_enc.reshape(-1, 1),
    cp_dose_enc.reshape(-1, 1),
    g_all,
    c_all,
])

X_train = X_all[:n_train]
X_test = X_all[n_train:]
y_train = train_targets[target_cols].values

# Control samples always have all-zero targets
is_ctrl_train = train_features["cp_type"].values == "ctl_vehicle"
is_ctrl_test = test_features["cp_type"].values == "ctl_vehicle"

# Initialize predictions with a small constant (controls stay at this)
preds = np.full((len(test_features), len(target_cols)), 1e-4)

# Train on treatment samples only
trt_mask = ~is_ctrl_train
trt_test_mask = ~is_ctrl_test
X_trt = X_train[trt_mask]
y_trt = y_train[trt_mask]
X_test_trt = X_test[trt_test_mask]
n_trt = int(trt_mask.sum())

print(f"Treatment train: {n_trt}, Treatment test: {int(trt_test_mask.sum())}")
print(f"Fitting {len(target_cols)} LightGBM models...")

for i, col in enumerate(target_cols):
    y = y_trt[:, i]
    n_pos = int(y.sum())
    pos_frac = n_pos / n_trt

    if n_pos < 3:
        # Too few positives — just use the prior
        preds[trt_test_mask, i] = max(pos_frac, 1e-5)
        continue

    model = LogisticRegression(
        C=0.5,
        solver="liblinear",
        max_iter=200,
        random_state=42,
    )
    model.fit(X_trt, y)
    prob = model.predict_proba(X_test_trt)[:, 1]
    preds[trt_test_mask, i] = np.clip(prob, 1e-5, 1 - 1e-5)

    if (i + 1) % 50 == 0:
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(target_cols)}] elapsed: {elapsed:.1f}s")

elapsed = time.time() - start
print(f"Training completed in {elapsed:.1f}s")

submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Submission saved: {len(submission)} rows x {len(target_cols)} targets")
