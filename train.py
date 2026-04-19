"""
Two-stage stacking LR: whitened PCA + 2-fold OOF SVD-compressed meta-features.
Parallelized with joblib (threading backend; liblinear releases GIL) for 48-core use.

Score history:
  baseline (no stacking, whitened PCA C=0.5):  -0.016141
  uncompressed stacking (319 features):         -0.016100
  SVD-20 stacking (133 features, in-sample):    -0.016092
  OOF + PCA-whitened meta 200 + Stage2 C=20:    -0.015907  ← committed best
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
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
pca_g = PCA(n_components=85, whiten=True, random_state=42)
pca_c = PCA(n_components=35, whiten=True, random_state=42)
g_pca = pca_g.fit_transform(all_feats[gene_cols].values)
c_pca = pca_c.fit_transform(all_feats[cell_cols].values)

cp_dose_map = {"D1": 0.0, "D2": 1.0}
cp_type_bin = (all_feats["cp_type"].values == "trt_cp").astype(float)
cp_time_24 = (all_feats["cp_time"].values == 24).astype(float)
cp_time_48 = (all_feats["cp_time"].values == 48).astype(float)
cp_time_72 = (all_feats["cp_time"].values == 72).astype(float)
cp_dose_enc = np.array([cp_dose_map[d] for d in all_feats["cp_dose"].values])

X_all = np.hstack([
    cp_type_bin.reshape(-1, 1),
    cp_time_24.reshape(-1, 1),
    cp_time_48.reshape(-1, 1),
    cp_time_72.reshape(-1, 1),
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
print(f"Stage 1 features: {X_trt.shape[1]}")


# ── Parallel worker functions (module-level for pickling) ─────────────────────

def _s1_oof(i, X_tr, X_val, y_col, n_tr):
    n_pos = int(y_col.sum())
    if n_pos < 2:
        return i, np.full(len(X_val), n_pos / n_tr)
    lr = LogisticRegression(C=5.0, solver="liblinear", max_iter=100, random_state=42)
    lr.fit(X_tr, y_col)
    return i, lr.predict_proba(X_val)[:, 1]


def _s1_full(i, X_tr, y_col, X_te, n_tr):
    n_pos = int(y_col.sum())
    if n_pos < 3:
        return i, np.full(len(X_te), n_pos / n_tr)
    lr = LogisticRegression(C=5.0, solver="liblinear", max_iter=100, random_state=42)
    lr.fit(X_tr, y_col)
    return i, lr.predict_proba(X_te)[:, 1]


def _s2(i, X_tr, y_col, X_te, n_tr):
    n_pos = int(y_col.sum())
    if n_pos < 3:
        return i, np.full(len(X_te), max(n_pos / n_tr, 1e-5))
    lr = LogisticRegression(C=5.0, penalty="l1", solver="liblinear", max_iter=200, random_state=42)
    lr.fit(X_tr, y_col)
    return i, np.clip(lr.predict_proba(X_te)[:, 1], 1e-5, 1 - 1e-5)


# ── Stage 1: 2-fold OOF meta-predictions (parallel) ──────────────────────────
kf = KFold(n_splits=2, shuffle=True, random_state=42)
meta_train = np.zeros((n_trt, len(target_cols)))
meta_test_trt = np.full((int(trt_test_mask.sum()), len(target_cols)), 1e-4)

print(f"\nStage 1 OOF: fitting {len(target_cols)} LR models per fold [parallel]...")
for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_trt)):
    X_tr_f = X_trt[tr_idx]
    X_val_f = X_trt[val_idx]
    y_tr_f = y_trt[tr_idx]
    n_tr_f = len(tr_idx)

    res = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_s1_oof)(i, X_tr_f, X_val_f, y_tr_f[:, i], n_tr_f)
        for i in range(len(target_cols))
    )
    for i, pred in res:
        meta_train[val_idx, i] = pred
    print(f"  Fold {fold_i + 1} done: {time.time() - start:.1f}s")

# Full Stage 1 on all training data → test meta-predictions (parallel)
print(f"\nStage 1 full (test preds): fitting {len(target_cols)} LR models [parallel]...")
res = Parallel(n_jobs=-1, prefer="threads")(
    delayed(_s1_full)(i, X_trt, y_trt[:, i], X_test_trt, n_trt)
    for i in range(len(target_cols))
)
for i, pred in res:
    meta_test_trt[:, i] = pred

t1 = time.time() - start
print(f"Stage 1 done in {t1:.1f}s")

# ── Stage 2: LR on [original features + OOF PCA-whitened meta-features] ──────
pca_meta = PCA(n_components=35, whiten=True, random_state=42)
meta_train_m = pca_meta.fit_transform(meta_train)
meta_test_m = pca_meta.transform(meta_test_trt)

# Include control training samples (y=0, zero meta) to improve Stage 2 boundary
n_ctrl = int(is_ctrl_train.sum())
X_ctrl = X_train[is_ctrl_train]
y_ctrl = y_train[is_ctrl_train]  # all zeros
meta_ctrl = np.zeros((n_ctrl, 35))
X_s2_train = np.vstack([np.hstack([X_trt, meta_train_m]),
                         np.hstack([X_ctrl, meta_ctrl])])
y_s2_train = np.vstack([y_trt, y_ctrl])
n_s2_train = len(X_s2_train)

n_ctrl_test = int(is_ctrl_test.sum())
X_ctrl_test = X_test[is_ctrl_test]
meta_ctrl_test = np.zeros((n_ctrl_test, 35))
# Full test: treatment + control (aligned to test_features order)
X_test2_full = np.empty((len(test_features), X_trt.shape[1] + 35))
X_test2_full[trt_test_mask] = np.hstack([X_test_trt, meta_test_m])
X_test2_full[is_ctrl_test] = np.hstack([X_ctrl_test, meta_ctrl_test])
print(f"PCA meta: {pca_meta.n_components} components (OOF+whitened), var={pca_meta.explained_variance_ratio_.sum():.3f}")
print(f"\nStage 2: fitting {len(target_cols)} LR models ({X_s2_train.shape[1]} features, +{n_ctrl} ctrl) [parallel]...")

preds = np.full((len(test_features), len(target_cols)), 1e-4)

res = Parallel(n_jobs=-1, prefer="threads")(
    delayed(_s2)(i, X_s2_train, y_s2_train[:, i], X_test2_full, n_s2_train)
    for i in range(len(target_cols))
)
for i, pred in res:
    preds[:, i] = pred  # predict for ALL test samples

elapsed = time.time() - start
print(f"Stage 2 done. Total: {elapsed:.1f}s")

submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "sig_id", test_features["sig_id"].values)
submission.to_csv("submission.csv", index=False)
print(f"Saved: {len(submission)} rows x {len(target_cols)} targets")
