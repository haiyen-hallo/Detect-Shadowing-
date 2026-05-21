"""
Train.py — Stacking: XGBoost + RF + MLP(LeakyReLU) → LR meta
==============================================================

CHIẾN LƯỢC DỮ LIỆU (dữ liệu nhỏ):
  ┌─────────────────────────────────────────────────────────────┐
  │  Toàn bộ dataset                                            │
  │  ├── 80% Train+Val (image-level split, không leak augment)  │
  │  │    └── 5-fold CV: tune tham số theo IoU/Dice trên Val    │
  │  └── 20% Test (giữ lại, CHỈ dùng để báo cáo cuối cùng)     │
  └─────────────────────────────────────────────────────────────┘
  Lý do: tập test không được dùng để tune → tránh overfitting metric.
  Hyperparameter tuning dựa trên Dice + IoU từ validation folds.

STACKING (XGBoost + RF + MLP → LR meta):
  - XGBoost   : boosting mạnh, xử lý imbalance tốt qua scale_pos_weight
  - RF         : ổn định, feature importance, giảm variance
  - MLP        : học pattern phi tuyến qua Leaky ReLU (torch nếu có, else sklearn)
  - LR meta    : kết hợp tối ưu vote của 3 model

PROBE-AWARE DERIVED FEATURES (8 features thêm vào 18 gốc):
  above_below_ratio, shadow_drop, dark_bright_above,
  lateral_contrast_nm, depth_darkness, vert_dark_depth,
  anti_reverb_dark, phys_shadow_score

Chạy:
    python Train.py                          # Stacking (mặc định)
    python Train.py --model rf               # RF baseline nhanh
    python Train.py --model both --cv        # Cả hai + CV tuning
    python Train.py --model stack --smote    # SMOTE

VỀ Noise.py: KHÔNG cần thay đổi.
  Noise.py tạo ra 18 features gốc → lưu vào npz.
  Derived features được tính TRONG Train.py từ 18 features đó.
"""

# ══════════════════════════════════════════════════════════════════
# ENV VARS — set TRƯỚC khi import sklearn/joblib (Windows fix)
# ══════════════════════════════════════════════════════════════════
import os, sys

if os.name == "nt":
    os.environ["LOKY_MAX_CPU_COUNT"]  = "14"
    os.environ["JOBLIB_BACKEND"]      = "threading"
    os.environ["OMP_NUM_THREADS"]     = "14"
    os.environ["MKL_NUM_THREADS"]     = "14"

import json, argparse, time, warnings
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    roc_curve,
    f1_score, ConfusionMatrixDisplay,
)
import joblib

warnings.filterwarnings("ignore")

# ── XGBoost ────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost không có — pip install xgboost")

# ── PyTorch (cho MLP Leaky ReLU) ───────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[warn] PyTorch không có — MLP dùng sklearn (không có LeakyReLU)")

# ── Optional deps ──────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ══════════════════════════════════════════════════════════════════
# CONSTANTS (chỉ những gì Train cần)
# ══════════════════════════════════════════════════════════════════
PATCH_H   = 16    # đồng bộ với Predict.py
PATCH_W   = 16
GLCM_BINS = 32

N_JOBS_RF       = 14
SHADOW_MAX_MEAN = 0.5   # ngưỡng vật lý cho derived features

# Tên nhóm features
GEO_FEATURE_NAMES = {
    "dist_origin_norm", "angle_axis_norm",
    "ray_mean_above", "lateral_drop",
    "vert_continuity", "reverb_score",
}
BASE_FEATURE_NAMES = [
    "contrast", "homogeneity", "correlation", "energy",
    "mean", "std", "skewness", "entropy",
    "row_ratio", "col_ratio", "local_contrast", "below_bright",
    "dist_origin_norm", "angle_axis_norm", "ray_mean_above",
    "lateral_drop", "vert_continuity", "reverb_score",
]
DERIVED_FEATURE_NAMES = [
    "above_below_ratio",
    "shadow_drop",
    "dark_bright_above",
    "lateral_contrast_nm",
    "depth_darkness",
    "vert_dark_depth",
    "anti_reverb_dark",
    "phys_shadow_score",
]


# ═══════════════════════════════════════════════════════════════════
# PROBE-AWARE FEATURE ENGINEERING
# (dùng chung cho Train.py và Predict.py)
# ═══════════════════════════════════════════════════════════════════

def engineer_probe_features(X: np.ndarray, feature_names: list) -> tuple:
    """
    Thêm 8 derived features encode mối quan hệ đầu dò → bóng cản.

    Nguyên lý vật lý được mã hóa:
      1. above_below_ratio : vùng gần đầu dò phải SÁNG HƠN bóng
      2. shadow_drop       : có bước nhảy intensity qua reflector
      3. dark_bright_above : bóng TỐI + vùng trên SÁNG (dấu hiệu mạnh nhất)
      4. lateral_contrast  : bóng TỐI HƠN vùng xung quanh ngang
      5. depth_darkness    : sâu + tối = bóng cản
      6. vert_dark_depth   : tối liên tục xuống sâu (không phải noise)
      7. anti_reverb_dark  : tối + KHÔNG có reverb → bóng thật
      8. phys_shadow_score : composite score tổng hợp

    Returns: (X_expanded, expanded_feature_names)
    """
    fn  = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6

    def col(name):
        return X[:, fn[name]].astype(np.float64) if name in fn \
               else np.zeros(len(X))

    mean      = col("mean")
    ray_above = col("ray_mean_above")
    lat_drop  = col("lateral_drop")
    vert_cont = col("vert_continuity")
    reverb    = col("reverb_score")
    dist_norm = col("dist_origin_norm")

    # 1. Tỷ lệ sáng/tối trên-dưới — shadow: ratio >> 1
    above_below = np.clip(ray_above / (mean + eps), 0.0, 10.0)

    # 2. Độ chênh intensity qua reflector (clipped ≥ 0)
    shadow_drop = np.clip(ray_above - mean, 0.0, 1.0)

    # 3. Tối × Sáng phía trên — phân biệt bóng cản vs vùng tối BTP
    darkness   = np.clip(SHADOW_MAX_MEAN - mean, 0.0, SHADOW_MAX_MEAN) / SHADOW_MAX_MEAN
    dark_bright = darkness * np.clip(ray_above, 0.0, 1.0)

    # 4. Normalized lateral drop
    lat_norm = np.clip(lat_drop / (mean + eps), -2.0, 5.0)

    # 5. Sâu + Tối
    depth_dark = dist_norm * np.clip(1.0 - mean, 0.0, 1.0)

    # 6. Tối liên tục theo chiều sâu
    vert_depth = vert_cont * dist_norm

    # 7. Tối + không reverb
    anti_reverb = np.clip(1.0 - reverb, 0.0, 1.0) * np.clip(1.0 - mean, 0.0, 1.0)

    # 8. Composite physics score
    phys = np.clip(above_below / 5.0, 0.0, 1.0) * shadow_drop * darkness
    phys = np.clip(phys, 0.0, 1.0)

    new_cols = np.column_stack([
        above_below, shadow_drop, dark_bright, lat_norm,
        depth_dark, vert_depth, anti_reverb, phys,
    ]).astype(np.float32)
    new_cols  = np.nan_to_num(new_cols, nan=0., posinf=1., neginf=-1.)
    X_new     = np.hstack([X, new_cols])
    new_names = feature_names + DERIVED_FEATURE_NAMES
    return X_new, new_names


# ═══════════════════════════════════════════════════════════════════
# MLP VỚI LEAKY RELU (PyTorch wrapper, sklearn-compatible)
# ═══════════════════════════════════════════════════════════════════

class _LeakyReLUNet(nn.Module if HAS_TORCH else object):
    """MLP với Leaky ReLU + BatchNorm + Dropout."""
    def __init__(self, input_dim, hidden_sizes, negative_slope=0.01, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(negative_slope=negative_slope),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # output: 1 logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible MLP với Leaky ReLU activation (PyTorch).
    Hỗ trợ imbalance qua pos_weight.

    Nếu PyTorch không có → tự động fallback sang sklearn MLPClassifier.
    """
    def __init__(self, hidden_sizes=(128, 64, 32),
                 lr=1e-3, epochs=150, batch_size=512,
                 negative_slope=0.01, dropout=0.30,
                 pos_weight="auto", patience=15,
                 validation_fraction=0.10, tol=1e-5,
                 random_state=42):
        self.hidden_sizes    = hidden_sizes
        self.lr              = lr
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.negative_slope  = negative_slope
        self.dropout         = dropout
        self.pos_weight      = pos_weight
        self.patience        = patience
        self.validation_fraction = validation_fraction
        self.tol             = tol
        self.random_state    = random_state
        self._model          = None
        self.classes_        = np.array([0, 1])

    def fit(self, X, y):
        if not HAS_TORCH:
            self._sklearn_mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_sizes,
                activation="relu", max_iter=300,
                random_state=self.random_state,
                early_stopping=True, validation_fraction=0.1,
            )
            self._sklearn_mlp.fit(X, y)
            return self

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        idx_all = np.arange(len(y))
        val_idx = []
        tr_idx  = []
        frac = float(self.validation_fraction)
        if 0.0 < frac < 0.5 and len(y) >= 10 and len(np.unique(y)) == 2:
            for cls in [0.0, 1.0]:
                cls_idx = idx_all[y == cls]
                rng.shuffle(cls_idx)
                n_val = int(round(len(cls_idx) * frac))
                n_val = min(max(n_val, 1), max(len(cls_idx) - 1, 0))
                val_idx.extend(cls_idx[:n_val])
                tr_idx.extend(cls_idx[n_val:])
            tr_idx = np.array(tr_idx, dtype=np.int64)
            val_idx = np.array(val_idx, dtype=np.int64)
            rng.shuffle(tr_idx); rng.shuffle(val_idx)
        else:
            tr_idx = idx_all
            val_idx = np.array([], dtype=np.int64)

        net = _LeakyReLUNet(
            X.shape[1], self.hidden_sizes,
            self.negative_slope, self.dropout,
        ).to(device)

        y_train_np = y[tr_idx].astype(np.int32)
        if self.pos_weight == "auto" or self.pos_weight is None:
            n_pos = int(y_train_np.sum())
            n_neg = int(len(y_train_np) - n_pos)
            pos_w = min(n_neg / max(n_pos, 1), 15.0)
        else:
            pos_w = float(self.pos_weight)
        self.pos_weight_ = float(pos_w)

        pw     = torch.tensor([self.pos_weight_], device=device)
        loss_fn= nn.BCEWithLogitsLoss(pos_weight=pw)
        opt    = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        train_ds = TensorDataset(X_t[tr_idx], y_t[tr_idx])
        val_ds   = TensorDataset(X_t[val_idx], y_t[val_idx]) if len(val_idx) else None
        drop_last = len(train_ds) > self.batch_size and len(train_ds) % self.batch_size == 1
        loader = DataLoader(train_ds, batch_size=self.batch_size,
                            shuffle=True, drop_last=drop_last)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                shuffle=False) if val_ds is not None else None

        best_loss = float("inf"); wait = 0; best_state = None
        for epoch in range(self.epochs):
            net.train()
            ep_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward(); opt.step()
                ep_loss += loss.item() * len(xb)
            ep_loss /= max(len(train_ds), 1)

            monitor_loss = ep_loss
            if val_loader is not None:
                net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        loss = loss_fn(net(xb), yb)
                        val_loss += loss.item() * len(xb)
                monitor_loss = val_loss / max(len(val_ds), 1)

            sched.step()
            if monitor_loss < best_loss - self.tol:
                best_loss = monitor_loss; wait = 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            net.load_state_dict(best_state)
        self._model  = net
        self._device = device
        return self

    def predict_proba(self, X):
        if not HAS_TORCH:
            return self._sklearn_mlp.predict_proba(X)
        self._model.eval()
        with torch.no_grad():
            X_t  = torch.tensor(X, dtype=torch.float32).to(self._device)
            logit= self._model(X_t).cpu().numpy()
        logit = np.clip(logit, -80.0, 80.0)
        prob1 = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - prob1, prob1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS (IoU / Dice ở cấp patch)
# ═══════════════════════════════════════════════════════════════════

def patch_iou(y_true, y_pred):
    inter = int(((y_pred == 1) & (y_true == 1)).sum())
    union = int(((y_pred == 1) | (y_true == 1)).sum())
    return inter / max(union, 1)


def patch_dice(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return 2 * tp / max(int(y_pred.sum()) + int(y_true.sum()), 1)


def class_ratio(y):
    y = np.asarray(y).astype(np.int32)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)


# Helper chung: dùng cho cả metric phụ ở patch-level và metric chính
# ở image-level sau khi group các patch theo từng ảnh.
def _binary_stats(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    acc       = (tp + tn) / max(len(y_true), 1)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": acc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def _image_groups(stems, n_samples):
    if stems is None:
        return [np.arange(n_samples, dtype=np.int64)]
    stems = np.asarray(stems).astype(str)
    if len(stems) != n_samples:
        raise ValueError("stems phải cùng độ dài với y_true/y_pred")
    return [np.where(stems == s)[0] for s in np.unique(stems)]


def image_mask_metrics(y_true, y_pred, stems=None, groups=None):
    """
    IoU/Dice ở cấp image mask: mỗi ảnh là một mask patch-grid riêng,
    sau đó lấy trung bình đều theo ảnh thay vì gộp toàn bộ patch.
    """
    y_true = np.asarray(y_true).astype(np.int32)
    y_pred = np.asarray(y_pred).astype(np.int32)
    groups = groups or _image_groups(stems, len(y_true))

    ious, dices, precisions, recalls = [], [], [], []
    for idx in groups:
        s = _binary_stats(y_true[idx], y_pred[idx])
        ious.append(s["iou"])
        dices.append(s["dice"])
        precisions.append(s["precision"])
        recalls.append(s["recall"])

    return {
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "std_iou": float(np.std(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dices)) if dices else 0.0,
        "std_dice": float(np.std(dices)) if dices else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "n_images": int(len(groups)),
    }


# ═══════════════════════════════════════════════════════════════════
# IMAGE-LEVEL SPLIT
# ═══════════════════════════════════════════════════════════════════

def _base_stem(stem: str) -> str:
    changed = True
    suffixes = ["_flip","_tx+5","_tx-5","_tx+10","_tx-10",
                "_left","_right","_L","_R"]
    while changed:
        changed = False
        for suf in suffixes:
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                changed = True
                break
    return stem


def image_level_split(stems_all, test_ratio=0.20, seed=42):
    """Chia 80/20 theo ảnh gốc, tránh data leak từ augmentation."""
    base_stems   = np.array([_base_stem(str(s)) for s in stems_all])
    unique_bases = np.unique(base_stems)
    n_test       = max(1, int(len(unique_bases) * test_ratio))

    rng        = np.random.default_rng(seed)
    idx        = rng.permutation(len(unique_bases))
    test_bases = set(unique_bases[idx[:n_test]])
    train_bases= set(unique_bases[idx[n_test:]])

    train_mask = np.array([_base_stem(str(s)) in train_bases for s in stems_all])
    test_mask  = np.array([_base_stem(str(s)) in test_bases  for s in stems_all])

    print(f"[split] {len(unique_bases)} ảnh gốc → "
          f"train+val={len(train_bases)} | test={len(test_bases)}")
    print(f"[split] Không có ảnh nào vừa train vừa test ✓")
    return train_mask, test_mask, list(test_bases)


def image_level_kfold(stems_trainval, n_splits=5, seed=42):
    """
    5-fold split theo ảnh gốc trong tập train+val.
    Đảm bảo augmented patches từ cùng ảnh luôn cùng 1 fold.
    Returns: list of (train_idx, val_idx) tuples
    """
    base_stems   = np.array([_base_stem(str(s)) for s in stems_trainval])
    unique_bases = np.unique(base_stems)
    rng          = np.random.default_rng(seed)
    shuffled     = rng.permutation(unique_bases)

    folds = np.array_split(shuffled, n_splits)
    splits= []
    for k in range(n_splits):
        val_bases  = set(folds[k])
        train_bases= set(shuffled) - val_bases
        tr_idx = np.where([b in train_bases for b in base_stems])[0]
        va_idx = np.where([b in val_bases   for b in base_stems])[0]
        splits.append((tr_idx, va_idx))
    return splits


# ═══════════════════════════════════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════════════════════════════════

def load_dataset(npz_path: str):
    d    = np.load(npz_path, allow_pickle=True)
    X    = d["X"].astype(np.float64)
    Y    = d["Y"].astype(np.int32)
    feature_names = [str(s) for s in d["feature_names"]]
    stems = d["stems"] if "stems" in d \
            else np.array([str(i) for i in range(len(Y))])

    n_pos = int(Y.sum()); n_neg = len(Y) - n_pos
    ratio = n_neg / max(n_pos, 1)

    print(f"\n[load] {npz_path}")
    print(f"[load] X={X.shape}   n_features={len(feature_names)}")
    geo     = [f for f in feature_names if f in GEO_FEATURE_NAMES]
    non_geo = [f for f in feature_names if f not in GEO_FEATURE_NAMES]
    print(f"[load] Non-geometric ({len(non_geo)}): {non_geo}")
    print(f"[load] Geometric     ({len(geo)}): {geo}")
    print(f"[load] shadow={n_pos:,} ({n_pos/len(Y)*100:.1f}%)  "
          f"bg={n_neg:,}  ratio=1:{ratio:.1f}")

    bad = int(np.isnan(X).sum()) + int(np.isinf(X).sum())
    if bad > 0:
        print(f"  {bad} NaN/Inf → 0")
        X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)
    return X, Y, feature_names, ratio, stems


# ═══════════════════════════════════════════════════════════════════
# SMOTE
# ═══════════════════════════════════════════════════════════════════

def apply_smote(X_tr, y_tr, target_ratio=2):
    if not HAS_SMOTE:
        print("[SMOTE] Bỏ qua — pip install imbalanced-learn"); return X_tr, y_tr
    n_sh = int(y_tr.sum()); n_bg = len(y_tr) - n_sh
    s    = min(n_sh * target_ratio / max(n_bg, 1), 1.0)
    print(f"\n[SMOTE] target 1:{target_ratio}  sh={n_sh:,} bg={n_bg:,}")
    sm  = SMOTE(sampling_strategy=s, k_neighbors=5, random_state=42)
    X_r, y_r = sm.fit_resample(X_tr.astype(np.float32), y_tr)
    print(f"[SMOTE] after: sh={y_r.sum():,} bg={(~y_r.astype(bool)).sum():,}")
    return X_r.astype(np.float64), y_r


# ═══════════════════════════════════════════════════════════════════
# BUILD MODELS
# ═══════════════════════════════════════════════════════════════════

def build_rf(ratio=None) -> RandomForestClassifier:
    # RF xử lý imbalance bằng class_weight="balanced"; ratio giữ để API đồng nhất.
    return RandomForestClassifier(
        n_estimators=600, max_depth=20, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        n_jobs=N_JOBS_RF, random_state=42, oob_score=True,
    )


def build_xgb(ratio, n_estimators=300, max_depth=6,
              learning_rate=0.05, subsample=0.8,
              colsample=0.8, reg_alpha=0.1, reg_lambda=1.0):
    """
    XGBoost: xử lý imbalance tốt qua scale_pos_weight = ratio.
    tree_method='hist' → nhanh hơn nhiều so với 'exact'.
    """
    if not HAS_XGB:
        raise ImportError("pip install xgboost")
    spw = min(float(ratio), 20.0)
    return XGBClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        learning_rate     = learning_rate,
        subsample         = subsample,
        colsample_bytree  = colsample,
        reg_alpha         = reg_alpha,
        reg_lambda        = reg_lambda,
        scale_pos_weight  = spw,        # xử lý class imbalance
        tree_method       = "hist",     # nhanh hơn 'exact'
        n_jobs            = N_JOBS_RF,
        eval_metric       = "logloss",
        random_state      = 42,
        verbosity         = 0,
    )


def build_mlp(ratio=None, hidden_sizes=(128, 64, 32),
              lr=1e-3, epochs=150, dropout=0.30):
    """
    MLP với Leaky ReLU (PyTorch) hoặc ReLU (sklearn fallback).
    pos_weight = ratio để xử lý imbalance trong loss function.
    """
    return TorchMLPClassifier(
        hidden_sizes   = hidden_sizes,
        lr             = lr,
        epochs         = epochs,
        dropout        = dropout,
        negative_slope = 0.01,   # Leaky ReLU slope
        pos_weight     = "auto",
        patience       = 20,
        random_state   = 42,
    )


def build_stacking(ratio,
                   xgb_params: dict = None,
                   rf_params:  dict = None,
                   mlp_params: dict = None,
                   stack_cv: int = 3) -> Pipeline:
    """
    Stacking: StandardScaler → [XGBoost, RF, MLP(LeakyReLU)] → LR meta.

    ┌─────────────────┬──────────┬──────────────────────────────┐
    │ Estimator       │ n_jobs   │ Lý do                        │
    ├─────────────────┼──────────┼──────────────────────────────┤
    │ StackingClass.  │    1     │ FIX WinError 87              │
    │ XGBoost         │   14     │ native threading             │
    │ RF              │   14     │ joblib threading             │
    │ MLP (PyTorch)   │  GPU/CPU │ batch training               │
    └─────────────────┴──────────┴──────────────────────────────┘
    """
    xp = xgb_params or {}
    rp = rf_params  or {}
    mp = mlp_params or {}

    xgb_est = build_xgb(ratio, **xp)
    rf_est  = build_rf(ratio)
    if rp:
        for k, v in rp.items():
            setattr(rf_est, k, v)
    mlp_est = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", build_mlp(ratio, **mp)),
    ])

    spw = min(ratio, 15.0)
    meta_lr = LogisticRegression(
        C=1.0, class_weight={0:1, 1:min(spw, 10.0)},
        max_iter=1000, random_state=42,
    )

    stacking = StackingClassifier(
        estimators     = [("xgb", xgb_est), ("rf", rf_est), ("mlp", mlp_est)],
        final_estimator= meta_lr,
        cv             = stack_cv,
        passthrough    = False,
        n_jobs         = 1,   # FIX Windows nested multiprocessing
    )
    return Pipeline([
        ("classifier", stacking),
    ])


# ═══════════════════════════════════════════════════════════════════
# THRESHOLD ANALYSIS + IMAGE-LEVEL CV
# ═══════════════════════════════════════════════════════════════════

def _threshold_grid(y_score, max_points=301):
    y_score = np.asarray(y_score, dtype=np.float64)
    finite = y_score[np.isfinite(y_score)]
    if finite.size == 0:
        return np.array([0.5], dtype=np.float64)
    qn = min(max_points, max(3, finite.size))
    quantiles = np.quantile(finite, np.linspace(0.0, 1.0, qn))
    grid = np.concatenate([
        np.linspace(0.0, 1.0, 201),
        quantiles,
        np.array([0.5], dtype=np.float64),
    ])
    return np.unique(np.clip(grid, 0.0, 1.0))


def analyze_thresholds(y_score, y_te, stems=None):
    """
    Image-level threshold search: compare predicted patch mask with doctor
    patch-label mask per image, then average IoU/Dice across images.
    """
    y_score = np.asarray(y_score, dtype=np.float64)
    y_te = np.asarray(y_te).astype(np.int32)
    groups = _image_groups(stems, len(y_te))
    rows = []

    for thr in _threshold_grid(y_score):
        yp = (y_score >= thr).astype(np.int32)
        patch_s = _binary_stats(y_te, yp)
        img_s = image_mask_metrics(y_te, yp, groups=groups)
        rows.append({
            "threshold": float(thr),
            "precision": img_s["mean_precision"],
            "recall": img_s["mean_recall"],
            "f1": img_s["mean_dice"],
            "iou_image": img_s["mean_iou"],
            "dice_image": img_s["mean_dice"],
            "patch_precision": patch_s["precision"],
            "patch_recall": patch_s["recall"],
            "patch_f1": patch_s["f1"],
            "patch_iou": patch_s["iou"],
            "patch_dice": patch_s["dice"],
            "accuracy": patch_s["accuracy"],
            "tn": patch_s["tn"], "fp": patch_s["fp"],
            "fn": patch_s["fn"], "tp": patch_s["tp"],
        })

    def _best_index(candidates):
        return max(candidates, key=lambda i: (
            rows[i]["dice_image"], rows[i]["iou_image"], rows[i]["patch_f1"]
        ))

    def _pack(row):
        packed = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)):
                packed[key] = round(float(value), 4)
            else:
                packed[key] = int(value)
        return packed

    idx_best = _best_index(range(len(rows)))

    def _find(min_r):
        candidates = [i for i, row in enumerate(rows) if row["recall"] >= min_r]
        if not candidates:
            candidates = [max(range(len(rows)), key=lambda i: rows[i]["recall"])]
        return _pack(rows[_best_index(candidates)])

    prec = np.array([row["precision"] for row in rows], dtype=np.float64)
    rec  = np.array([row["recall"] for row in rows], dtype=np.float64)
    thrs = np.array([row["threshold"] for row in rows], dtype=np.float64)
    f1   = np.array([row["f1"] for row in rows], dtype=np.float64)

    return {
        "best_f1": _pack(rows[idx_best]),
        "sens80" : _find(0.80),
        "sens90" : _find(0.90),
    }, (prec, rec, thrs, f1)


def cross_validate_iou_dice(model_fn, X_tv, y_tv, stems_tv,
                             n_splits=5, do_smote=False):
    """
    Image-level CV: split by image, fit on fold train, tune threshold on
    validation patch labels, and report IoU/Dice as image-mask averages.
    """
    stems_tv = np.asarray(stems_tv)
    splits = image_level_kfold(stems_tv, n_splits=n_splits)
    results = []

    for k, (tr_idx, va_idx) in enumerate(splits):
        X_tr, y_tr = X_tv[tr_idx], y_tv[tr_idx]
        X_va, y_va = X_tv[va_idx], y_tv[va_idx]
        stems_va = stems_tv[va_idx]

        if do_smote:
            X_tr, y_tr = apply_smote(X_tr, y_tr)

        fold_ratio = class_ratio(y_tr)
        try:
            model = model_fn(fold_ratio)
        except TypeError:
            model = model_fn()
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_va)[:, 1]
        thr_info, _ = analyze_thresholds(y_prob, y_va, stems_va)
        best_thr = float(thr_info["best_f1"]["threshold"])
        y_pred = (y_prob >= best_thr).astype(np.int32)

        img_m = image_mask_metrics(y_va, y_pred, stems_va)
        iou_patch = patch_iou(y_va, y_pred)
        dice_patch = patch_dice(y_va, y_pred)
        auc = float(roc_auc_score(y_va, y_prob)) if len(np.unique(y_va)) == 2 else 0.0
        f1 = float(f1_score(y_va, y_pred, zero_division=0))

        results.append({
            "fold": k + 1,
            "threshold": best_thr,
            "iou": img_m["mean_iou"],
            "dice": img_m["mean_dice"],
            "iou_patch": iou_patch,
            "dice_patch": dice_patch,
            "auc": auc,
            "f1": f1,
            "n_images": img_m["n_images"],
        })
        print(f"  fold {k+1}/{n_splits}  thr={best_thr:.4f}  "
              f"ImgIoU={img_m['mean_iou']:.4f}  ImgDice={img_m['mean_dice']:.4f}  "
              f"PatchIoU={iou_patch:.4f}  PatchDice={dice_patch:.4f}  AUC={auc:.4f}")

    avg_iou = float(np.mean([r["iou"] for r in results]))
    avg_dice = float(np.mean([r["dice"] for r in results]))
    avg_iou_patch = float(np.mean([r["iou_patch"] for r in results]))
    avg_dice_patch = float(np.mean([r["dice_patch"] for r in results]))
    avg_auc = float(np.mean([r["auc"] for r in results]))
    avg_f1 = float(np.mean([r["f1"] for r in results]))
    med_thr = float(np.median([r["threshold"] for r in results]))

    print("\n  --- CV Mean (validation patch labels -> image mask) ---")
    print(f"  Image IoU={avg_iou:.4f}  Image Dice={avg_dice:.4f}")
    print(f"  Patch IoU={avg_iou_patch:.4f}  Patch Dice={avg_dice_patch:.4f}")
    print(f"  AUC={avg_auc:.4f}  F1={avg_f1:.4f}  median_thr={med_thr:.4f}")

    return {
        "folds": results,
        "mean_iou": round(avg_iou, 4),
        "mean_dice": round(avg_dice, 4),
        "mean_iou_patch": round(avg_iou_patch, 4),
        "mean_dice_patch": round(avg_dice_patch, 4),
        "mean_auc": round(avg_auc, 4),
        "mean_f1": round(avg_f1, 4),
        "median_best_threshold": round(med_thr, 4),
    }


# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def get_feature_importance(model):
    try:
        clf = model.named_steps.get("classifier", model) \
              if hasattr(model, "named_steps") else model
        if hasattr(clf, "estimators_"):
            for est in clf.estimators_:
                cand = est.steps[-1][1] if hasattr(est, "steps") else est
                if hasattr(cand, "feature_importances_"):
                    return cand.feature_importances_, cand
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_, clf
    except Exception:
        pass
    return None, None


def print_feature_importance(imp, feature_names):
    pairs  = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    geo_t  = derived_t = base_t = 0.0

    print(f"\n  Feature Importance  [★=geometric  ◆=probe-derived]:")
    print(f"  {'Feature':<26} {'Imp':>8}   Bar")
    print(f"  {'─'*60}")
    for fname, imp_v in pairs:
        bar  = "█" * int(imp_v * 40)
        if fname in DERIVED_FEATURE_NAMES:
            mark = " ◆"; derived_t += imp_v
        elif fname in GEO_FEATURE_NAMES:
            mark = " ★"; geo_t += imp_v
        else:
            mark = "  "; base_t += imp_v
        print(f"  {fname:<26}{mark} {imp_v:>6.4f}  {bar}")

    total = geo_t + derived_t + base_t
    print(f"\n  ── Đóng góp theo nhóm ──")
    print(f"  Probe-derived : {derived_t:.4f} ({derived_t/max(total,1e-6)*100:.1f}%)")
    print(f"  Geometric     : {geo_t:.4f}     ({geo_t/max(total,1e-6)*100:.1f}%)")
    print(f"  GLCM+Intensity: {base_t:.4f}  ({base_t/max(total,1e-6)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════

def evaluate(name, model, X_te, y_te, out_dir, feature_names=None,
             stems_te=None, decision_threshold=0.5):
    print(f"\n{'═'*70}")
    print(f"  [{name}] — Kết quả trên tập TEST (20%)")
    print(f"{'═'*70}")

    y_sc   = model.predict_proba(X_te)[:, 1]
    y_pred = (y_sc >= decision_threshold).astype(int)
    print(classification_report(y_te, y_pred,
          target_names=["bg(0)","shadow(1)"], digits=4))

    auc  = roc_auc_score(y_te, y_sc)
    ap   = average_precision_score(y_te, y_sc)
    cm   = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn,fp,fn,tp = cm.ravel()
    acc  = (tp+tn)/len(y_te)
    sens = tp/max(tp+fn,1)
    spec = tn/max(tn+fp,1)
    f1v  = f1_score(y_te, y_pred, zero_division=0)
    iou  = patch_iou(y_te, y_pred)
    dice = patch_dice(y_te, y_pred)
    img_m = image_mask_metrics(y_te, y_pred, stems_te)

    print(f"  Acc={acc:.4f}  AUC={auc:.4f}  AvgP={ap:.4f}")
    print(f"  Sens={sens:.4f}  Spec={spec:.4f}  F1={f1v:.4f}  thr={decision_threshold:.4f}")
    print(f"  IoU/Dice (image-mask)={img_m['mean_iou']:.4f}/{img_m['mean_dice']:.4f} "
          f"(n={img_m['n_images']})")
    print(f"  IoU/Dice (patch-label)={iou:.4f}/{dice:.4f}")

    thr_info, pr_data = analyze_thresholds(y_sc, y_te, stems_te)
    print(f"\n  Threshold gợi ý cho Predict.py:")
    print(f"  {'Key':<10} {'thr':>6} {'Prec':>6} {'Sens':>6} {'F1':>6} {'Acc':>6}")
    print(f"  {'─'*48}")
    results_thr = {}
    for key, t in thr_info.items():
        yp2 = (y_sc >= t["threshold"]).astype(int)
        cm2 = confusion_matrix(y_te, yp2, labels=[0, 1]); tn2,fp2,fn2,tp2 = cm2.ravel()
        acc2 = (tp2+tn2)/len(y_te)
        print(f"  {key:<10} {t['threshold']:>6.4f} {t['precision']:>6.4f} "
              f"{t['recall']:>6.4f} {t['f1']:>6.4f}  acc={acc2:.4f}")
        results_thr[key] = {**t,"accuracy":round(acc2,4),
                            "tn":int(tn2),"fp":int(fp2),
                            "fn":int(fn2),"tp":int(tp2)}

    imp_arr, _ = get_feature_importance(model)
    if imp_arr is not None and feature_names:
        print_feature_importance(imp_arr, feature_names)

    if HAS_MPL:
        _plot_eval(name, y_sc, y_te, cm, auc, ap, acc, f1v,
                   thr_info, pr_data, out_dir, feature_names, imp_arr)

    return {
        "model": name,
        "accuracy": round(float(acc),4), "roc_auc": round(float(auc),4),
        "avg_prec": round(float(ap),4),  "sensitivity": round(float(sens),4),
        "specificity": round(float(spec),4), "f1": round(float(f1v),4),
        "iou_image": round(float(img_m["mean_iou"]),4),
        "dice_image": round(float(img_m["mean_dice"]),4),
        "iou_image_std": round(float(img_m["std_iou"]),4),
        "dice_image_std": round(float(img_m["std_dice"]),4),
        "iou_patch": round(float(iou),4), "dice_patch": round(float(dice),4),
        "decision_threshold": round(float(decision_threshold),4),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "thresholds": results_thr,
    }


def _plot_eval(name, y_sc, y_te, cm, auc, ap, acc, f1v,
               thr_info, pr_data, out_dir, feature_names, imp_arr):
    fig  = plt.figure(figsize=(28, 5))
    gs   = gridspec.GridSpec(1, 5, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    ConfusionMatrixDisplay(cm, display_labels=["bg","shadow"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"{name}\nAcc={acc:.3f} F1={f1v:.3f}", fontsize=9)

    p_a, r_a, _, _ = pr_data
    axes[1].plot(r_a, p_a, lw=2.5, color="#4e79a7")
    clr = {"best_f1":"red","sens80":"#f28e2b","sens90":"#59a14f"}
    for key, t in thr_info.items():
        axes[1].scatter([t["recall"]],[t["precision"]],s=100,zorder=6,
                        c=clr[key],edgecolors="k",lw=0.5,
                        label=f"{key} thr={t['threshold']} F1={t['f1']:.3f}")
    axes[1].fill_between([0.8,1],[0.8,0.8],[1,1],alpha=0.08,color="green")
    axes[1].axhline(0.8,color="gray",ls="--",lw=1,alpha=0.5)
    axes[1].axvline(0.8,color="gray",ls="--",lw=1,alpha=0.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  AP={ap:.4f}",fontsize=9)
    axes[1].legend(fontsize=6); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)

    fpr_a,tpr_a,_ = roc_curve(y_te, y_sc)
    axes[2].plot(fpr_a,tpr_a,lw=2.5,color="#59a14f")
    axes[2].plot([0,1],[0,1],"k--",lw=1)
    axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
    axes[2].set_title(f"ROC  AUC={auc:.4f}",fontsize=9)
    axes[2].grid(alpha=0.3)

    if imp_arr is not None and feature_names:
        sidx   = np.argsort(imp_arr)
        names  = [feature_names[i] for i in sidx]
        vals   = imp_arr[sidx]
        colors = ["#e15759" if n in DERIVED_FEATURE_NAMES else
                  "#f28e2b" if n in GEO_FEATURE_NAMES else "#4e79a7"
                  for n in names]
        axes[3].barh(names, vals, color=colors)
        axes[3].set_title("Feature Importance\n(đỏ=probe-derived  cam=geometric)",fontsize=8)
        axes[3].grid(alpha=0.2,axis="x")

        d_t = sum(imp_arr[i] for i,n in enumerate(feature_names) if n in DERIVED_FEATURE_NAMES)
        g_t = sum(imp_arr[i] for i,n in enumerate(feature_names) if n in GEO_FEATURE_NAMES)
        b_t = 1.0 - d_t - g_t
        axes[4].pie([d_t,g_t,b_t],
                    labels=[f"Probe-derived\n{d_t:.1%}",
                            f"Geometric\n{g_t:.1%}",
                            f"GLCM+Int\n{b_t:.1%}"],
                    colors=["#e15759","#f28e2b","#4e79a7"],
                    autopct="%1.1f%%",startangle=90)
        axes[4].set_title("Feature Group",fontsize=9)
    else:
        for ax in axes[3:]:
            ax.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax.transAxes)

    fig.suptitle(f"Model: {name} | Test set",fontsize=11,y=1.02)
    plt.tight_layout()
    sn   = name.lower().replace(" ","_").replace("(","").replace(")","")
    path = os.path.join(out_dir,f"{sn}_eval.png")
    plt.savefig(path,dpi=130,bbox_inches="tight"); plt.close()
    print(f"  [plot] → {path}")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

def print_summary(all_metrics, cv_results=None):
    print(f"\n{'═'*95}")
    print(f"  TỔNG KẾT")
    print(f"{'═'*95}")
    H = (f"  {'Model':<36} {'Acc':>7} {'AUC':>7} {'F1':>7} "
         f"{'Sens':>7} {'ImgIoU':>7} {'ImgDice':>7}")
    print(H); print(f"  {'─'*85}")
    for m in all_metrics:
        ok_s = "✅" if m["sensitivity"]  >= 0.80 else "  "
        ok_f = "✅" if m["f1"]           >= 0.75 else "  "
        ok_i = "✅" if m.get("iou_image",0) >= 0.40 else "  "
        print(f"  {m['model']:<36} {m['accuracy']:>7.4f} {m['roc_auc']:>7.4f} "
              f"{m['f1']:>7.4f} {m['sensitivity']:>7.4f} "
              f"{m.get('iou_image',0):>7.4f} {m.get('dice_image',0):>7.4f}  "
              f"S{ok_s} F{ok_f} I{ok_i}")

    if cv_results:
        print(f"\n  CV Validation (5-fold IoU/Dice trên train+val):")
        for name, cv in cv_results.items():
            print(f"  {name:<36} IoU={cv['mean_iou']:.4f}  "
                  f"Dice={cv['mean_dice']:.4f}  "
                  f"AUC={cv['mean_auc']:.4f}")

    best = max(all_metrics,
               key=lambda m:(m["sensitivity"]+m["f1"]+m.get("iou_image",0))/3)
    print(f"\n  ★ Best model (sens+F1+IoU): {best['model']}")

    print(f"\n  Ngưỡng cho Predict.py (RF_DIRECT_THR):")
    print(f"  {'Model':<22} {'Key':<10} {'--thr':>6}  (sens / prec / f1)")
    print(f"  {'─'*60}")
    for m in all_metrics:
        for key in ["best_f1","sens80","sens90"]:
            if key in m.get("thresholds",{}):
                t = m["thresholds"][key]
                print(f"  {m['model'][:20]:<22} {key:<10} {t['threshold']:>6.4f}  "
                      f"(sens={t['recall']:.3f}  prec={t['precision']:.3f}  "
                      f"f1={t['f1']:.3f})")
    print(f"{'═'*95}")


# ═══════════════════════════════════════════════════════════════════
# MAIN RUN
# ═══════════════════════════════════════════════════════════════════

def run(dataset_path, model_choice="stack", test_size=0.2,
        do_cv=True, do_smote=False, n_cv_folds=5):

    out_dir = str(Path(dataset_path).parent / "models")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load + feature engineering ─────────────────────────────
    X, Y, feature_names, ratio, stems = load_dataset(dataset_path)

    print(f"\n[features] Thêm {len(DERIVED_FEATURE_NAMES)} probe-aware derived features...")
    X, feature_names = engineer_probe_features(X, feature_names)
    print(f"[features] Tổng: {X.shape[1]} features "
          f"({len(BASE_FEATURE_NAMES)} gốc + {len(DERIVED_FEATURE_NAMES)} derived)")

    feat_path = os.path.join(out_dir, "feature_names.json")
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump({"all_features": feature_names,
                   "base_features": BASE_FEATURE_NAMES,
                   "derived_features": DERIVED_FEATURE_NAMES,
                   "n_total": len(feature_names)}, f, indent=2)
    print(f"[features] → {feat_path}")

    # ── 2. Train+Val / Test split (80/20 theo ảnh) ────────────────
    trainval_mask, test_mask, test_bases = image_level_split(
        stems, test_ratio=test_size)
    X_tv, y_tv = X[trainval_mask], Y[trainval_mask]
    X_te, y_te = X[test_mask],    Y[test_mask]
    stems_tv   = stems[trainval_mask]
    stems_te   = stems[test_mask]

    test_split_path = os.path.join(out_dir, "test_split_images.json")
    test_eval_stems = sorted(np.unique(stems[test_mask]).astype(str).tolist())
    with open(test_split_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_images": test_bases,
            "test_samples": test_eval_stems,
        }, f, indent=2)
    print(f"\n[Save] {len(test_bases)} ảnh test → {test_split_path}")
    print(f"[split] Train+Val={len(X_tv):,}  Test={len(X_te):,}")
    print(f"[split] Shadow TV: {y_tv.sum():,} ({y_tv.mean()*100:.1f}%)  "
          f"Test: {y_te.sum():,} ({y_te.mean()*100:.1f}%)")

    all_metrics = []; fitted = {}; cv_results = {}

    # ── 3a. RF baseline ────────────────────────────────────────────
    if model_choice in ("rf", "both"):
        print(f"\n{'─'*60}")
        print(f"[RF] Baseline RandomForest")
        print(f"{'─'*60}")

        if do_cv:
            print(f"[RF-CV] 5-fold validation trên Train+Val (IoU/Dice)...")
            cv = cross_validate_iou_dice(
                lambda fold_ratio: build_rf(fold_ratio), X_tv, y_tv, stems_tv,
                n_splits=n_cv_folds, do_smote=do_smote)
            cv_results["RF"] = cv
            rf_eval_thr = cv.get("median_best_threshold", 0.5)
        else:
            rf_eval_thr = 0.5

        # Train on full train+val
        X_tr_final, y_tr_final = X_tv.copy(), y_tv.copy()
        if do_smote:
            X_tr_final, y_tr_final = apply_smote(X_tr_final, y_tr_final)

        t0 = time.time()
        rf = build_rf(class_ratio(y_tr_final))
        rf.fit(X_tr_final, y_tr_final)
        print(f"[RF] OOB={rf.oob_score_:.4f}  {time.time()-t0:.1f}s")

        m = evaluate("RandomForest", rf, X_te, y_te, out_dir, feature_names,
                     stems_te=stems_te, decision_threshold=rf_eval_thr)
        all_metrics.append(m); fitted["RF"] = rf
        joblib.dump(rf, os.path.join(out_dir, "rf_model.pkl"))
        print(f"[RF] ✅ rf_model.pkl")

    # ── 3b. Stacking (XGBoost + RF + MLP) ─────────────────────────
    if model_choice in ("stack", "both"):
        print(f"\n{'─'*60}")
        print(f"[Stack] XGBoost + RF + MLP(LeakyReLU) → LR meta")
        print(f"{'─'*60}")
        print(f"  PyTorch: {'✅' if HAS_TORCH else '❌ fallback sklearn MLP'}")
        print(f"  XGBoost: {'✅' if HAS_XGB  else '❌ cần pip install xgboost'}")

        if not HAS_XGB:
            print("[ERROR] Cần cài: pip install xgboost")
            sys.exit(1)

        if do_cv:
            print(f"\n[Stack-CV] 5-fold validation trên Train+Val...")
            cv = cross_validate_iou_dice(
                lambda fold_ratio: build_stacking(fold_ratio), X_tv, y_tv, stems_tv,
                n_splits=n_cv_folds, do_smote=do_smote)
            cv_results["Stacking"] = cv
            stack_eval_thr = cv.get("median_best_threshold", 0.5)
        else:
            stack_eval_thr = 0.5

        # Train on full train+val
        X_tr_final, y_tr_final = X_tv.copy(), y_tv.copy()
        if do_smote:
            X_tr_final, y_tr_final = apply_smote(X_tr_final, y_tr_final)

        print(f"\n[Stack] Training trên toàn Train+Val={len(X_tr_final):,} samples...")
        print(f"        n_features={X_tr_final.shape[1]}  "
              f"ước tính: 15–40 phút...")
        t0 = time.time()
        stack = build_stacking(class_ratio(y_tr_final))
        stack.fit(X_tr_final, y_tr_final)
        elapsed = time.time() - t0
        print(f"[Stack] Done {elapsed:.1f}s ({elapsed/60:.1f} phút)")

        m = evaluate("Stacking(XGB+RF+MLP)", stack,
                     X_te, y_te, out_dir, feature_names,
                     stems_te=stems_te, decision_threshold=stack_eval_thr)
        all_metrics.append(m); fitted["Stack"] = stack

        joblib.dump(stack, os.path.join(out_dir, "stacking_model.pkl"))
        print(f"[Stack] ✅ stacking_model.pkl")

        try:
            # Lưu inner RF để Predict.py --model rf dùng được
            clf   = stack.named_steps["classifier"]
            rf_in = clf.named_estimators_["rf"]
            joblib.dump(rf_in, os.path.join(out_dir, "rf_model.pkl"))
            print(f"[Stack] ✅ Inner RF → rf_model.pkl")
        except Exception as e:
            print(f"[Stack] ⚠ Inner RF: {e}")

    print_summary(all_metrics, cv_results if do_cv else None)

    # ── 4. Lưu kết quả ────────────────────────────────────────────
    clean = [{k:v for k,v in m.items() if k!="y_score"} for m in all_metrics]
    if do_cv:
        for name, cv in cv_results.items():
            for m in clean:
                model_l = m["model"].lower()
                if name.lower() in model_l or (name == "RF" and "randomforest" in model_l):
                    m["cv_validation"] = cv

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  [results] → {results_path}")

    print(f"\n  ⚠  Predict.py phải gọi engineer_probe_features() trước predict!")
    print(f"     from Train import engineer_probe_features")
    return all_metrics


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train shadow — XGBoost+RF+MLP stacking, IoU/Dice CV tuning"
    )
    parser.add_argument("--dataset",
        default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed\dataset.npz")
    parser.add_argument("--model", default="stack",
                        choices=["rf","stack","both"])
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--cv",        action="store_true",
                        help="Chạy 5-fold CV validation (tune tham số theo IoU/Dice)")
    parser.add_argument("--smote",     action="store_true")
    parser.add_argument("--n_folds",   default=5, type=int)
    args = parser.parse_args()

    print("=" * 70)
    print("  TRAIN — Shadow Detection")
    print("  Stacking: XGBoost + RF + MLP(Leaky ReLU) → LR meta")
    print("  Features: 18 gốc + 8 probe-derived = 26 features")
    print("  Tuning: 5-fold CV theo IoU/Dice (KHÔNG dùng tập test)")
    print("=" * 70)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Model     : {args.model}")
    print(f"  Test 20%  : giữ lại, chỉ dùng để báo cáo cuối")
    print(f"  CV 5-fold : {'on' if args.cv else 'off'} (tune theo IoU/Dice)")
    print(f"  SMOTE     : {'on' if args.smote else 'off'}")
    print(f"  PyTorch   : {'✅ LeakyReLU MLP' if HAS_TORCH else '❌ sklearn MLP fallback'}")
    print(f"  XGBoost   : {'✅' if HAS_XGB else '❌ pip install xgboost'}")
    print(f"  RF n_jobs : {N_JOBS_RF} core  |  Stack n_jobs=1 (Windows fix)")
    print("=" * 70 + "\n")

    run(args.dataset, args.model, args.test_size,
        do_cv=args.cv, do_smote=args.smote, n_cv_folds=args.n_folds)
