"""
Two-stage stacking LR with whitened PCA.

Best so far: whitened PCA (80/30, C=0.5) = -0.016141.

Stage 1: per-target whitened-PCA LR — same as best model.
  Also stores training-set predictions as meta-features.
Stage 2: per-target LR on [original 113 features + 206 stage-1 predictions].
  Captures strong target correlations (e.g., nfkb/proteasome: r=0.92,
  pdgfr/kit: r=0.91) — if stage-1 predicts proteasome inhibition high,
  stage-2 boosts nfkb_inhibitor prediction.

Timing estimate: ~64s (stage 1) + ~180s (stage 2) = ~244s total.
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

all_feats = pd.concat([train_features, test_features], ignore_index=True)
n_train = len(train_features)

# Whitened PCA
pca_g = PCA(n_components=80, whiten=True, random_state=42)
pca_c = PCA(n_components=30, whiten=True, random_state=42)
g_pca = pca_g.fit_transform(all_feats[gene_cols].values)
c_pca = pca_c.fit_transform(all_feats[cell_cols].values)

cp_time_map = {24: 0.0, 48: 0.5, 72: 1.0}
cp_dose_map = {"D1": 0.0, "D2": 1.0}
cp_type_bin = (all_feats["cp_type"].values == "trt_cp").astype(float)
cp_time_enc = np.array([cp_time_map[t] for t in all_feats["cp_time"].values])
cp_dose_enc = np.array([cp_dose_map[d] for d in all_feats["cp_dose"].values])

X_all = np.hstack([
    cp_type_bin.reshape(-1, 1),
    cp_time_enc.reshape(-1, 1),
    cp_dose_enc.reshape(-1, 1),
    g_pca, c_pca,
])

X_train = X_all[:n_train]
X_test = X_all[n_train:]
y_train = train_targets[target_cols].values

is_ctrl_train = train_features["cp_type"].values == "ctl_vehicle"
is_ctrl_test = test_features["cp_type"].values == "ctl_vehicle"
trt_mask = ~is_ctrl_train
trt_test_mask = ~is_ctrl_test
X_trt = X_train[trt_mask]
y_trt = y_train[trt_mask]
X_test_trt = X_test[trt_test_mask]
n_trt = int(trt_mask.sum())

print(f"Treatment train: {n_trt}, Treatment test: {int(trt_test_mask.sum())}")
print(f"Stage 1 features: {X_trt.shape[1]}, Stage 2 features: {X_trt.shape[1] + len(target_cols)}")

# ── Stage 1: baseline whitened-PCA LR ──────────────────────────────────────
print(f"\nStage 1: fitting {len(target_cols)} LR models (whitened PCA)...")
meta_train = np.zeros((n_trt, len(target_cols)))        # stage-1 preds on training set
meta_test = np.full((int(trt_test_mask.sum()), len(target_cols)), 1e-4)

for i, col in enumerate(target_cols):
    y = y_trt[:, i]
    n_pos = int(y.sum())
    pos_frac = n_pos / n_trt

    if n_pos < 3:
        meta_train[:, i] = pos_frac
        meta_test[:, i] = pos_frac
        continue

    lr = LogisticRegression(C=0.5, solver="liblinear", max_iter=200, random_state=42)
    lr.fit(X_trt, y)
    meta_train[:, i] = lr.predict_proba(X_trt)[:, 1]
    meta_test[:, i] = lr.predict_proba(X_test_trt)[:, 1]

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(target_cols)}] elapsed: {time.time()-start:.1f}s")

t1 = time.time() - start
print(f"Stage 1 done in {t1:.1f}s")

# ── Stage 2: LR on [original features + stage-1 predictions] ───────────────
# Stage-1 predictions encode target correlations: high P(proteasome) → high P(nfkb)
X_trt2 = np.hstack([X_trt, meta_train])
X_test2 = np.hstack([X_test_trt, meta_test])
print(f"\nStage 2: fitting {len(target_cols)} LR models ({X_trt2.shape[1]} features)...")

preds = np.full((len(test_features), len(target_cols)), 1e-4)

for i, col in enumerate(target_cols):
    y = y_trt[:, i]
    n_pos = int(y.sum())
    pos_frac = n_pos / n_trt

    if n_pos < 3:
        preds[trt_test_mask, i] = max(pos_frac, 1e-5)
        continue

    lr = LogisticRegression(C=0.5, solver="liblinear", max_iter=200, random_state=42)
    lr.fit(X_trt2, y)
    prob = lr.predict_proba(X_test2)[:, 1]
    preds[trt_test_mask, i] = np.clip(prob, 1e-5, 1 - 1e-5)

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(target_cols)}] elapsed: {time.time()-start:.1f}s")

elapsed = time.time() - start
print(f"Stage 2 done. Total: {elapsed:.1f}s")

submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Saved: {len(submission)} rows x {len(target_cols)} targets")
