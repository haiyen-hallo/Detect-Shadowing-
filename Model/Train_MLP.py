import os, sys
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")   # CUDA báo lỗi đúng vị trí

if os.name == "nt":
    os.environ["LOKY_MAX_CPU_COUNT"]  = "14"
    os.environ["JOBLIB_BACKEND"]      = "threading"
    os.environ["OMP_NUM_THREADS"]     = "14"
    os.environ["MKL_NUM_THREADS"]     = "14"

import json, argparse, time, warnings
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# FIX 11: BalancedRandomForest (undersampling) — thay thế SMOTE
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_BRF = True
except ImportError:
    HAS_BRF = False
    print("[warn] pip install imbalanced-learn  (cần BalancedRandomForest)")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, roc_curve,
    f1_score, ConfusionMatrixDisplay,
)
import joblib

# ── PyTorch ────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[torch] device = {_DEVICE}  "
      f"({'CUDA ' + torch.version.cuda if torch.cuda.is_available() else 'CPU'})")

warnings.filterwarnings("ignore")

# ── XGBoost ────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] pip install xgboost")

# ── SMOTE ──────────────────────────────────────────────────────────
# FIX 11: SMOTE bị LOẠI BỎ theo yêu cầu (oversampling TB phá vỡ logic vật lý).
# Giữ lại import để tương thích ngược, nhưng KHÔNG bao giờ được kích hoạt.
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
HAS_SMOTE = False  # FIX 11: ép tắt SMOTE ở mọi nơi

# ── Matplotlib ─────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
PATCH_H   = 16
PATCH_W   = 16
GLCM_BINS = 32
N_JOBS_RF = 14

# FIX 5+11: SHADOW_MAX_MEAN = 0.38 (ngưỡng darkness mới theo yêu cầu).
# Predict.py phải import giá trị này từ Train.py:
#   from Train import SHADOW_MAX_MEAN, engineer_probe_features, SELECTED_FEATURES, select_features
SHADOW_MAX_MEAN = 0.34

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

# FIX 6: 12 derived features (tăng từ 8 lên 12)
DERIVED_FEATURE_NAMES = [
    # Nhóm 1 – vật lý cơ bản (8)
    "above_below_ratio",    # ray_mean_above / mean  →  shadow: >> 1
    "shadow_drop",          # clip(above-mean, 0,1)  →  bước nhảy intensity
    "dark_bright_above",    # darkness × above        →  dấu hiệu mạnh nhất
    "lateral_contrast_nm",  # lateral_drop / mean    →  tối hơn hàng xóm
    "depth_darkness",       # dist_norm × (1-mean)   →  sâu + tối
    "vert_dark_depth",      # vert_cont × dist_norm  →  tối liên tục xuống
    "anti_reverb_dark",     # (1-reverb)×(1-mean)    →  tối, không reverb
    "phys_shadow_score",    # composite vật lý
    # Nhóm 2 – column context (2) — bắt bóng rộng BTP
    "col_dark_score",       # darkness × row_weight  →  proxy cho cột tối
    "depth_col_dark",       # dist_norm × col_dark   →  cột tối ở độ sâu
    # Nhóm 3 – discriminator (2) — phân biệt shadow vs vùng tối thường
    "is_shadow_candidate",  # dark AND bright_above  →  discriminator quan trọng nhất
    "shadow_contrast",      # above_ratio × lat_pos  →  contrast 2 chiều
]
N_FEATURES_TOTAL = len(BASE_FEATURE_NAMES) + len(DERIVED_FEATURE_NAMES)  # 30 (pool)

# ════════════════════════════════════════════════════════════════════
# FIX 11: SELECTED FEATURES — chỉ giữ 8 features có ý nghĩa vật lý nhất
#   row_ratio, angle_axis_norm, col_ratio, ray_mean_above,
#   depth_darkness (derived), mean, contrast, is_shadow_candidate (derived)
# Lý do: 30 features quá nhiều → model không tư duy được với mask khó.
# ════════════════════════════════════════════════════════════════════
SELECTED_FEATURES = [
    "row_ratio",
    "angle_axis_norm",
    "col_ratio",
    "ray_mean_above",
    "depth_darkness",        # derived (Nhóm 1)
    "mean",
    "contrast",
    "is_shadow_candidate",   # derived (Nhóm 3) — discriminator chính
]
N_FEATURES_SELECTED = len(SELECTED_FEATURES)  # 8


def select_features(X: np.ndarray, feature_names: list,
                    selected: list = None) -> tuple[np.ndarray, list]:
    """
    FIX 11: Slice ma trận đặc trưng theo SELECTED_FEATURES.
    PHẢI gọi SAU khi engineer_probe_features() để derived features đã có sẵn.

    Predict.py PHẢI import và gọi đúng thứ tự:
        X, fns = engineer_probe_features(X, fns)
        X, fns = select_features(X, fns)
    """
    if selected is None:
        selected = SELECTED_FEATURES
    fn  = {name: i for i, name in enumerate(feature_names)}
    missing = [s for s in selected if s not in fn]
    if missing:
        raise ValueError(
            f"[select_features] thiếu features trong pool: {missing}\n"
            f"Pool hiện có: {feature_names}"
        )
    idx = [fn[s] for s in selected]
    return X[:, idx].astype(X.dtype, copy=False), list(selected)


# ═══════════════════════════════════════════════════════════════════
# FIX 5+6: PROBE-AWARE FEATURE ENGINEERING
# Hàm này dùng chung Train.py và Predict.py → import từ Train
# ═══════════════════════════════════════════════════════════════════

def engineer_probe_features(X: np.ndarray,
                             feature_names: list) -> tuple[np.ndarray, list]:
    """
    Thêm 12 derived features encode mối quan hệ đầu dò → bóng cản.

    QUAN TRỌNG:
      - Predict.py PHẢI import hàm này: from Train import engineer_probe_features
      - SHADOW_MAX_MEAN phải giống nhau ở cả 2 file (đang = 0.38 — FIX 11)
      - Nếu khác → features Train ≠ features Predict → model predict sai hoàn toàn

    is_shadow_candidate = darkness × above_signal:
      Kết hợp 2 dấu hiệu vật lý cơ bản nhất:
        1. Patch phải TỐI (mean < SHADOW_MAX_MEAN)
        2. Vùng phía trên (gần đầu dò) phải SÁNG HƠN rõ rệt (above_below > 1.0)
      Bóng thật       → cả 2 đều cao → is_shadow_candidate cao
      Vùng tối thường → darkness cao nhưng above_below ≈ 1 (không có reflector)
      → is_shadow_candidate thấp
    """
    fn  = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6

    def _col(name: str) -> np.ndarray:
        return X[:, fn[name]].astype(np.float64) if name in fn \
               else np.zeros(len(X), np.float64)

    mean      = _col("mean")
    ray_above = _col("ray_mean_above")
    lat_drop  = _col("lateral_drop")
    vert_cont = _col("vert_continuity")
    reverb    = _col("reverb_score")
    dist_norm = _col("dist_origin_norm")
    row_ratio = _col("row_ratio")

    # ── Nhóm 1: vật lý cơ bản ─────────────────────────────────────
    darkness    = np.clip((SHADOW_MAX_MEAN - mean) / SHADOW_MAX_MEAN, 0.0, 1.0)
    above_below = np.clip(ray_above / (mean + eps), 0.0, 10.0)
    shadow_drop = np.clip(ray_above - mean, 0.0, 1.0)
    dark_bright = darkness * np.clip(ray_above, 0.0, 1.0)
    lat_norm    = np.clip(lat_drop / (mean + eps), -2.0, 5.0)
    depth_dark  = dist_norm * np.clip(1.0 - mean, 0.0, 1.0)
    vert_depth  = vert_cont * dist_norm
    anti_reverb = np.clip(1.0 - reverb, 0.0, 1.0) * np.clip(1.0 - mean, 0.0, 1.0)
    phys = np.clip(
        np.clip(above_below / 5.0, 0.0, 1.0) * shadow_drop * darkness,
        0.0, 1.0)

    # ── Nhóm 2: column context ────────────────────────────────────
    # Patches sâu hơn (row_ratio lớn) được weight nhiều hơn
    # vì bóng cản trải dài từ reflector xuống đáy fan
    row_weight   = np.clip(0.4 + 0.6 * row_ratio, 0.4, 1.0)
    col_dark     = darkness * row_weight
    depth_col    = dist_norm * col_dark

    # ── Nhóm 3: discriminator ─────────────────────────────────────
    above_signal   = np.clip((above_below - 1.0) / 2.0, 0.0, 1.0)
    is_shadow_cand = darkness * above_signal               # KEY feature
    lat_pos        = np.clip(lat_drop, 0.0, 1.0)
    shadow_cont    = np.clip(above_below / 5.0, 0.0, 1.0) * lat_pos

    new_cols = np.column_stack([
        above_below, shadow_drop, dark_bright, lat_norm,
        depth_dark,  vert_depth,  anti_reverb, phys,
        col_dark,    depth_col,
        is_shadow_cand, shadow_cont,
    ]).astype(np.float32)
    new_cols = np.nan_to_num(new_cols, nan=0.0, posinf=1.0, neginf=-1.0)

    return np.hstack([X, new_cols]), feature_names + DERIVED_FEATURE_NAMES


# ═══════════════════════════════════════════════════════════════════
# FIX 1: FOCAL LOSS STABLE
# ═══════════════════════════════════════════════════════════════════

class _FocalLoss(nn.Module):
    """
    FIX 1: Focal Loss numerically stable.

    Cũ (unstable):
        p_t = torch.exp(-bce)   ← khi bce >> 0, p_t → 0, gây underflow
        focal = (1-p_t)^gamma * bce  ← NaN → CUDA crash

    Mới (stable):
        p = sigmoid(logits)     ← dùng sigmoid trực tiếp, không qua exp(-bce)
        p_t = p*y + (1-p)*(1-y)← xác suất class đúng, luôn trong [0,1]
        .clamp(1e-6, 1-1e-6)   ← tránh log(0) và pow(0, gamma)
        focal = (1-p_t)^gamma * bce ← giờ an toàn

    Tại sao Focal Loss tốt hơn BCE+pos_weight cho shadow detection:
      BCE+pos_weight: reweight theo class (shadow vs background)
      Focal: thêm (1-p_t)^gamma → DOWN-weight easy examples
      → Model tập trung vào hard examples: shadow patches khó phân biệt
        với dark tissue (đây chính là lý do model hay bị nhầm)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # FIX: dùng sigmoid → p_t stable, không qua exp(-bce)
        p     = torch.sigmoid(logits)
        p_t   = (p * targets + (1.0 - p) * (1.0 - targets)).clamp(1e-6, 1.0 - 1e-6)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal = (1.0 - p_t).pow(self.gamma) * bce
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * focal).mean()


# ═══════════════════════════════════════════════════════════════════
# FIX 2+3: MLP STABLE CUDA + VRAM MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

class _LeakyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple,
                 slope: float = 0.01, drop: float = 0.35):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.BatchNorm1d(h),   # FIX 3: cần batch >= 2
                nn.LeakyReLU(slope),
                nn.Dropout(drop),
            ]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    MLP với Leaky ReLU + Focal Loss + CUDA safe.

    FIX 2: VRAM management:
      fit()          → train trên GPU → model.cpu() → empty_cache()
      predict_proba() → model lên device → infer → model.cpu() → empty_cache()
      Đảm bảo mỗi MLP không giữ VRAM sau khi xong,
      quan trọng vì StackingClassifier tạo nhiều MLP trong CV.

    FIX 3: BatchNorm safety:
      batch_size = min(bs, n_train//2)  đảm bảo batch >= 2 samples
      drop_last=True mọi lúc để tránh batch size = 1 với BatchNorm1d

    FIX 1: Focal Loss stable (xem _FocalLoss doc)
    """
    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=150,
                 batch_size=128, slope=0.01, dropout=0.35,
                 focal_gamma=2.0, val_frac=0.12,
                 patience=20, tol=1e-5, seed=42):
        self.hidden      = hidden
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.slope       = slope
        self.dropout     = dropout
        self.focal_gamma = focal_gamma
        self.val_frac    = val_frac
        self.patience    = patience
        self.tol         = tol
        self.seed        = seed
        self._model      = None
        self._dev_str    = "cpu"
        self.classes_    = np.array([0, 1])

    # ── helpers ────────────────────────────────────────────────────
    @staticmethod
    def _free_vram():
        """FIX 2: Giải phóng VRAM sau mỗi MLP training."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ── fit ────────────────────────────────────────────────────────
    def fit(self, X, y):
        torch.manual_seed(self.seed)
        rng    = np.random.default_rng(self.seed)
        device = _DEVICE

        # FIX 2: clear cache trước khi bắt đầu
        self._free_vram()

        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)

        # Stratified val split
        idx = np.arange(len(y))
        tr_i, va_i = [], []
        for cls in [0.0, 1.0]:
            ci = idx[y == cls]; rng.shuffle(ci)
            nv = max(1, int(len(ci) * self.val_frac))
            nv = min(nv, len(ci) - 1)
            va_i.extend(ci[:nv]); tr_i.extend(ci[nv:])
        tr_i = np.array(tr_i, np.int64)
        va_i = np.array(va_i, np.int64)
        rng.shuffle(tr_i); rng.shuffle(va_i)

        # FIX 3: batch_size an toàn (đảm bảo batch >= 2 cho BatchNorm)
        n_tr  = len(tr_i)
        bs    = min(self.batch_size, max(n_tr // 2, 2))

        # Focal alpha từ class ratio của fold (không dùng global)
        n_pos = int(y[tr_i].sum()); n_neg = n_tr - n_pos
        focal_alpha = float(np.clip(n_neg / max(n_tr, 1), 0.5, 0.95))

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)

        net     = _LeakyMLP(X.shape[1], self.hidden,
                             self.slope, self.dropout).to(device)
        loss_fn = _FocalLoss(alpha=focal_alpha, gamma=self.focal_gamma)
        opt     = torch.optim.AdamW(net.parameters(),
                                    lr=self.lr, weight_decay=1e-3)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                      opt, T_max=self.epochs)

        # FIX 3: drop_last=True LUÔN LUÔN để tránh batch size=1 → BatchNorm crash
        tr_dl = DataLoader(TensorDataset(Xt[tr_i], yt[tr_i]),
                           batch_size=bs, shuffle=True, drop_last=True)
        va_dl = DataLoader(TensorDataset(Xt[va_i], yt[va_i]),
                           batch_size=min(bs * 2, 512), shuffle=False)

        best_val  = float("inf"); wait = 0; best_w = None

        for _ in range(self.epochs):
            # Train
            net.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                opt.step()

            # Val loss (FIX 1: stable focal loss không crash)
            net.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, yb in va_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    vl += loss_fn(net(xb), yb).item() * len(xb)
            vl /= max(len(va_i), 1)
            sched.step()

            if vl < best_val - self.tol:
                best_val = vl; wait = 0
                best_w = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience: break

        if best_w:
            net.load_state_dict(best_w)

        # FIX 2: model về CPU ngay → giải phóng VRAM cho fold tiếp theo
        self._model   = net.cpu()
        self._dev_str = str(device)
        self._free_vram()
        return self

    def predict_proba(self, X):
        self._model.eval()
        device = torch.device(self._dev_str)

        # FIX 2: lên device chỉ trong khi infer, về CPU ngay sau
        self._model.to(device)
        try:
            all_logits = []
            Xt = torch.tensor(np.asarray(X, np.float32))
            bs = 1024
            with torch.no_grad():
                for i in range(0, len(Xt), bs):
                    xb = Xt[i:i+bs].to(device)
                    all_logits.append(self._model(xb).cpu())
            log = torch.cat(all_logits).numpy()
        finally:
            # FIX 2: luôn về CPU dù có lỗi hay không
            self._model.cpu()
            self._free_vram()

        log = np.clip(log, -80.0, 80.0)
        p1  = 1.0 / (1.0 + np.exp(-log))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def patch_iou(yt, yp) -> float:
    yt = np.asarray(yt, np.int32); yp = np.asarray(yp, np.int32)
    return int(((yp==1)&(yt==1)).sum()) / max(int(((yp==1)|(yt==1)).sum()), 1)

def patch_dice(yt, yp) -> float:
    yt = np.asarray(yt, np.int32); yp = np.asarray(yp, np.int32)
    tp = int(((yp==1)&(yt==1)).sum())
    return 2*tp / max(int(yp.sum())+int(yt.sum()), 1)

def class_ratio(y) -> float:
    y = np.asarray(y, np.int32)
    return int((y==0).sum()) / max(int((y==1).sum()), 1)

def _bin_stats(yt, yp) -> dict:
    yt = np.asarray(yt, np.int32); yp = np.asarray(yp, np.int32)
    tp=int(((yp==1)&(yt==1)).sum()); fp=int(((yp==1)&(yt==0)).sum())
    fn=int(((yp==0)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
    pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1)
    return {"tp":tp,"fp":fp,"fn":fn,"tn":tn,
            "precision":pr,"recall":rc,
            "f1":2*pr*rc/max(pr+rc,1e-9),
            "accuracy":(tp+tn)/max(len(yt),1),
            "iou":tp/max(tp+fp+fn,1),
            "dice":2*tp/max(2*tp+fp+fn,1)}

def _img_groups(stems, n) -> list:
    if stems is None: return [np.arange(n, np.int64)]
    stems = np.asarray(stems, str)
    return [np.where(stems==s)[0] for s in np.unique(stems)]

def image_mask_metrics(yt, yp, stems=None, groups=None) -> dict:
    """IoU/Dice per-image → trung bình. Metric chính để so sánh với mask bác sĩ."""
    yt = np.asarray(yt, np.int32); yp = np.asarray(yp, np.int32)
    groups = groups or _img_groups(stems, len(yt))
    ious,dices,precs,recs = [],[],[],[]
    for idx in groups:
        s = _bin_stats(yt[idx], yp[idx])
        ious.append(s["iou"]); dices.append(s["dice"])
        precs.append(s["precision"]); recs.append(s["recall"])
    def _m(lst): return float(np.mean(lst)) if lst else 0.0
    def _s(lst): return float(np.std(lst))  if lst else 0.0
    return {"mean_iou":_m(ious),"std_iou":_s(ious),
            "mean_dice":_m(dices),"std_dice":_s(dices),
            "mean_precision":_m(precs),"mean_recall":_m(recs),
            "n_images":len(groups)}


# ═══════════════════════════════════════════════════════════════════
# FIX 7: THRESHOLD SEARCH (image-level, chỉ 1 phiên bản)
# ═══════════════════════════════════════════════════════════════════

def _thr_grid(y_score, max_pts=301) -> np.ndarray:
    fin = y_score[np.isfinite(y_score)]
    if fin.size == 0: return np.array([0.5])
    q = np.quantile(fin, np.linspace(0, 1, min(max_pts, max(3, fin.size))))
    return np.unique(np.clip(np.concatenate([np.linspace(0,1,201), q, [0.5]]), 0, 1))


def analyze_thresholds(y_score, y_te, stems=None):
    """
    FIX 7: Image-level threshold search (bỏ phiên bản patch-level cũ).
    Tìm threshold tối ưu theo Image Dice (không phải Patch F1).
    """
    y_score = np.asarray(y_score, np.float64)
    y_te    = np.asarray(y_te, np.int32)
    groups  = _img_groups(stems, len(y_te))
    rows    = []

    for thr in _thr_grid(y_score):
        yp = (y_score >= thr).astype(np.int32)
        ps = _bin_stats(y_te, yp)
        im = image_mask_metrics(y_te, yp, groups=groups)
        rows.append({
            "threshold":   float(thr),
            "precision":   im["mean_precision"],
            "recall":      im["mean_recall"],
            "f1":          im["mean_dice"],
            "iou_image":   im["mean_iou"],
            "dice_image":  im["mean_dice"],
            **{f"patch_{k}": ps[k]
               for k in ("precision","recall","f1","iou","dice","accuracy",
                         "tn","fp","fn","tp")},
        })

    def _score(i):
        return (rows[i]["dice_image"], rows[i]["iou_image"], rows[i]["patch_f1"])
    def _best(cs): return max(cs, key=_score)
    def _pack(r):
        return {k: round(float(v),4) if isinstance(v,(float,np.floating)) else int(v)
                for k,v in r.items()}
    def _find(min_r):
        cs = [i for i,r in enumerate(rows) if r["recall"] >= min_r]
        if not cs: cs = [max(range(len(rows)), key=lambda i:rows[i]["recall"])]
        return _pack(rows[_best(cs)])

    idx_best = _best(range(len(rows)))
    prec = np.array([r["precision"] for r in rows])
    rec  = np.array([r["recall"]    for r in rows])
    thrs = np.array([r["threshold"] for r in rows])
    f1s  = np.array([r["f1"]        for r in rows])

    return {"best_f1":_pack(rows[idx_best]),
            "sens80":_find(0.80), "sens90":_find(0.90)}, (prec, rec, thrs, f1s)


# ═══════════════════════════════════════════════════════════════════
# IMAGE-LEVEL SPLIT & K-FOLD
# ═══════════════════════════════════════════════════════════════════

def _base_stem(stem: str) -> str:
    """FIX 10: while-loop xử lý chồng suffix augmentation."""
    suffixes = ["_flip","_tx+5","_tx-5","_tx+10","_tx-10",
                "_left","_right","_L","_R"]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if stem.endswith(suf):
                stem = stem[:-len(suf)]; changed = True; break
    return stem


def image_level_split(stems_all, test_ratio=0.20, seed=42):
    base = np.array([_base_stem(str(s)) for s in stems_all])
    uniq = np.unique(base)
    n_te = max(1, int(len(uniq) * test_ratio))
    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(uniq))
    te_b = set(uniq[idx[:n_te]]); tr_b = set(uniq[idx[n_te:]])
    tr_m = np.array([_base_stem(str(s)) in tr_b for s in stems_all])
    te_m = np.array([_base_stem(str(s)) in te_b for s in stems_all])
    print(f"[split] {len(uniq)} ảnh → train+val={len(tr_b)} | test={len(te_b)}")
    return tr_m, te_m, list(te_b)


def image_level_kfold(stems_tv, n_splits=5, seed=42) -> list:
    base  = np.array([_base_stem(str(s)) for s in stems_tv])
    uniq  = np.unique(base)
    rng   = np.random.default_rng(seed)
    shuf  = rng.permutation(uniq)
    folds = np.array_split(shuf, n_splits)
    splits = []
    for k in range(n_splits):
        vb = set(folds[k]); tb = set(shuf) - vb
        splits.append((
            np.where([b in tb for b in base])[0],
            np.where([b in vb for b in base])[0],
        ))
    return splits


# ═══════════════════════════════════════════════════════════════════
# LOAD DATASET + SMOTE
# ═══════════════════════════════════════════════════════════════════

def load_dataset(npz_path: str):
    d     = np.load(npz_path, allow_pickle=True)
    X     = d["X"].astype(np.float64)
    Y     = d["Y"].astype(np.int32)
    fns   = [str(s) for s in d["feature_names"]]
    stems = d["stems"] if "stems" in d \
            else np.array([str(i) for i in range(len(Y))])
    n_pos = int(Y.sum()); n_neg = len(Y) - n_pos
    ratio = n_neg / max(n_pos, 1)
    print(f"\n[load] {npz_path}")
    print(f"[load] X={X.shape}  shadow={n_pos:,}({n_pos/len(Y)*100:.1f}%)"
          f"  bg={n_neg:,}  ratio=1:{ratio:.1f}")
    bad = int(np.isnan(X).sum()) + int(np.isinf(X).sum())
    if bad:
        X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)
        print(f"  {bad} NaN/Inf → 0")
    return X, Y, fns, ratio, stems


def apply_smote(X_tr, y_tr, target_ratio=2):
    """
    FIX 11: SMOTE BỊ TẮT — giữ lại stub để không vỡ import/CLI cũ.
    Imbalance được xử lý bằng BalancedRandomForest (undersampling)
    và scale_pos_weight trong XGBoost.
    """
    print("[SMOTE] ⛔ Đã tắt theo FIX 11 (dùng BalancedRandomForest + scale_pos_weight thay thế).")
    return X_tr, y_tr


# ═══════════════════════════════════════════════════════════════════
# BUILD MODELS
# ═══════════════════════════════════════════════════════════════════

def build_rf():
    """
    FIX 11: Dùng BalancedRandomForestClassifier (undersampling background
    theo từng bootstrap) thay cho RandomForest+class_weight.
    Lý do: với 8 features, class_weight không đủ; undersampling giúp
    model nhìn thấy nhiều shadow hơn → tư duy được mask khó.
    Fallback về RandomForestClassifier(class_weight='balanced_subsample')
    nếu imbalanced-learn không có sẵn.
    """
    if HAS_BRF:
        # imbalanced-learn ≥0.11: sampling_strategy='auto' = undersample bg về = #shadow
        try:
            return BalancedRandomForestClassifier(
                n_estimators=500, max_depth=20,
                min_samples_leaf=2, min_samples_split=4,
                max_features="sqrt",
                sampling_strategy="auto",  # undersample bg
                replacement=False, bootstrap=True,
                n_jobs=N_JOBS_RF, random_state=42,
                oob_score=True,
            )
        except TypeError:
            # Bản imblearn cũ không có replacement/bootstrap params
            return BalancedRandomForestClassifier(
                n_estimators=500, max_depth=20,
                min_samples_leaf=2, min_samples_split=4,
                max_features="sqrt",
                sampling_strategy="auto",
                n_jobs=N_JOBS_RF, random_state=42,
                oob_score=True,
            )
    # Fallback (không khuyến nghị)
    print("[RF] ⚠ BalancedRandomForest không có — fallback class_weight='balanced_subsample'")
    return RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=2,
        min_samples_split=4, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=N_JOBS_RF,
        random_state=42, oob_score=True,
    )


def build_xgb(ratio) -> "XGBClassifier":
    """
    FIX 11: XGBoost tuned theo yêu cầu:
      • max_depth=4         : cây nông hơn → ít overfit khi chỉ có 8 features
      • colsample_bytree=1.0: dùng đủ 8/8 cột mỗi cây (không random drop)
      • max_delta_step=1    : giới hạn bước cập nhật → giảm thiên vị về
                              class background đa số khi scale_pos_weight lớn
    """
    if not HAS_XGB: raise ImportError("pip install xgboost")
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=1.0,
        min_child_weight=3, gamma=0.1,
        max_delta_step=1,                         # FIX 11
        reg_alpha=0.1, reg_lambda=2.0,
        scale_pos_weight=min(float(ratio), 20.0),
        tree_method="hist", n_jobs=N_JOBS_RF,
        eval_metric="logloss", random_state=42, verbosity=0,
    )


def build_mlp() -> TorchMLPClassifier:
    """pos_weight tính tự động từ fold data (pos_weight='auto')."""
    return TorchMLPClassifier(
        hidden=(64, 32), lr=1e-3, epochs=150,
        batch_size=128, slope=0.01, dropout=0.35,
        focal_gamma=2.0, val_frac=0.12,
        patience=20, seed=42,
    )


def build_stacking(ratio, stack_cv: int = 3) -> Pipeline:
    """
    FIX 8: stack_cv=2 trong outer CV (ít data/fold), =3 khi train final.
    StandardScaler CHỈ cho MLP (XGBoost + RF là tree-based, không cần scale).
    n_jobs=1: FIX WinError 87 (Windows nested multiprocessing).
    """
    spw  = min(float(ratio), 15.0)
    meta = LogisticRegression(
        C=1.0, class_weight={0:1, 1:min(spw, 10.0)},
        max_iter=1000, random_state=42,
    )
    mlp_pipe = Pipeline([("scaler", StandardScaler()), ("mlp", build_mlp())])
    stacking = StackingClassifier(
        estimators=[("xgb", build_xgb(ratio)),
                    ("rf",  build_rf()),
                    ("mlp", mlp_pipe)],
        final_estimator=meta, cv=stack_cv,
        passthrough=False, n_jobs=1,
    )
    return Pipeline([("classifier", stacking)])


# ═══════════════════════════════════════════════════════════════════
# FIX 7+9: 5-FOLD CV (image-level, 1 phiên bản, không có ratio param)
# ═══════════════════════════════════════════════════════════════════

def cross_validate_iou_dice(model_fn, X_tv, y_tv, stems_tv,
                             n_splits=5, do_smote=False) -> dict:
    """
    FIX 9: bỏ param ratio (tính fold_ratio từ y_tr trong từng fold).
    FIX 7: chỉ 1 phiên bản image-level (bỏ patch-level cũ).
    FIX 11: SMOTE bị ép False bất kể caller truyền gì.

    model_fn(fold_ratio) → model mới chưa fit.
    Threshold tối ưu theo Image Dice (mục tiêu: overlap mask bác sĩ).
    """
    # FIX 11: SMOTE bị tắt cứng
    do_smote = False
    stems_tv = np.asarray(stems_tv)
    splits   = image_level_kfold(stems_tv, n_splits=n_splits)
    results  = []

    for k, (tr_i, va_i) in enumerate(splits):
        X_tr, y_tr = X_tv[tr_i], y_tv[tr_i]
        X_va, y_va = X_tv[va_i], y_tv[va_i]
        st_va      = stems_tv[va_i]

        if do_smote:
            X_tr, y_tr = apply_smote(X_tr, y_tr)

        # FIX 9: fold_ratio từ training fold, không dùng global
        fold_ratio = class_ratio(y_tr)
        try:   m = model_fn(fold_ratio)
        except TypeError: m = model_fn()
        m.fit(X_tr, y_tr)

        y_prob   = m.predict_proba(X_va)[:, 1]
        thr_info, _ = analyze_thresholds(y_prob, y_va, st_va)
        best_thr = float(thr_info["best_f1"]["threshold"])
        y_pred   = (y_prob >= best_thr).astype(np.int32)

        im  = image_mask_metrics(y_va, y_pred, st_va)
        auc = float(roc_auc_score(y_va, y_prob)) \
              if len(np.unique(y_va)) == 2 else 0.0
        f1  = float(f1_score(y_va, y_pred, zero_division=0))

        results.append({
            "fold":k+1, "threshold":best_thr,
            "iou":im["mean_iou"], "dice":im["mean_dice"],
            "iou_patch":patch_iou(y_va,y_pred),
            "dice_patch":patch_dice(y_va,y_pred),
            "auc":auc, "f1":f1, "n_images":im["n_images"],
        })
        print(f"  fold {k+1}/{n_splits}  thr={best_thr:.4f}  "
              f"ImgIoU={im['mean_iou']:.4f}  ImgDice={im['mean_dice']:.4f}  "
              f"PatchIoU={results[-1]['iou_patch']:.4f}  AUC={auc:.4f}")

    def _avg(key): return float(np.mean([r[key] for r in results]))
    med_thr = float(np.median([r["threshold"] for r in results]))
    print(f"\n  ── CV Mean ───────────────────────────────────────────────────")
    print(f"  Image: IoU={_avg('iou'):.4f}  Dice={_avg('dice'):.4f}")
    print(f"  Patch: IoU={_avg('iou_patch'):.4f}  Dice={_avg('dice_patch'):.4f}")
    print(f"  AUC={_avg('auc'):.4f}  F1={_avg('f1'):.4f}  "
          f"median_thr={med_thr:.4f}")

    return {"folds":results,
            "mean_iou":round(_avg("iou"),4),
            "mean_dice":round(_avg("dice"),4),
            "mean_iou_patch":round(_avg("iou_patch"),4),
            "mean_dice_patch":round(_avg("dice_patch"),4),
            "mean_auc":round(_avg("auc"),4),
            "mean_f1":round(_avg("f1"),4),
            "median_best_threshold":round(med_thr,4)}


# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def get_feature_importance(model):
    try:
        clf = model.named_steps.get("classifier", model) \
              if hasattr(model, "named_steps") else model
        if hasattr(clf, "estimators_"):
            for est in clf.estimators_:
                cand = est.steps[-1][1] if hasattr(est,"steps") else est
                if hasattr(cand,"feature_importances_"):
                    return cand.feature_importances_, cand
        if hasattr(clf,"feature_importances_"):
            return clf.feature_importances_, clf
    except Exception: pass
    return None, None


def print_feature_importance(imp, feature_names):
    pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    geo_t = der_t = base_t = 0.0
    print(f"\n  Feature Importance  [★=geometric  ◆=probe-derived]:")
    print(f"  {'Feature':<26} {'Imp':>8}   Bar")
    print(f"  {'─'*62}")
    for fn, iv in pairs:
        bar = "█" * int(iv * 40)
        if fn in DERIVED_FEATURE_NAMES:  mark=" ◆"; der_t  += iv
        elif fn in GEO_FEATURE_NAMES:    mark=" ★"; geo_t   += iv
        else:                            mark="  "; base_t  += iv
        print(f"  {fn:<26}{mark} {iv:>6.4f}  {bar}")
    tot = geo_t + der_t + base_t + 1e-9
    print(f"\n  Probe-derived : {der_t:.4f}  ({der_t/tot*100:.1f}%)")
    print(f"  Geometric     : {geo_t:.4f}  ({geo_t/tot*100:.1f}%)")
    print(f"  GLCM+Intensity: {base_t:.4f}  ({base_t/tot*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════

def evaluate(name, model, X_te, y_te, out_dir,
             feature_names=None, stems_te=None, decision_threshold=0.5):
    print(f"\n{'═'*70}\n  [{name}] — TEST SET\n{'═'*70}")

    y_sc   = model.predict_proba(X_te)[:, 1]
    y_pred = (y_sc >= decision_threshold).astype(int)
    print(classification_report(y_te, y_pred,
          target_names=["bg(0)","shadow(1)"], digits=4))

    auc  = roc_auc_score(y_te, y_sc)
    ap   = average_precision_score(y_te, y_sc)
    cm   = confusion_matrix(y_te, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    acc  = (tp+tn)/len(y_te); sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1)
    f1v  = f1_score(y_te, y_pred, zero_division=0)
    iou  = patch_iou(y_te, y_pred); dice = patch_dice(y_te, y_pred)
    img  = image_mask_metrics(y_te, y_pred, stems_te)

    print(f"  Acc={acc:.4f}  AUC={auc:.4f}  AvgP={ap:.4f}")
    print(f"  Sens={sens:.4f}  Spec={spec:.4f}  F1={f1v:.4f}  "
          f"thr={decision_threshold:.4f}")
    print(f"  Image IoU/Dice = {img['mean_iou']:.4f} / {img['mean_dice']:.4f}"
          f"  ±{img['std_iou']:.4f}  (n={img['n_images']})")
    print(f"  Patch IoU/Dice = {iou:.4f} / {dice:.4f}")

    thr_info, pr_data = analyze_thresholds(y_sc, y_te, stems_te)
    print(f"\n  Threshold gợi ý cho Predict.py:")
    print(f"  {'Key':<10} {'thr':>6} {'ImgDice':>8} {'Sens':>6}")
    print(f"  {'─'*38}")
    results_thr = {}
    for key, t in thr_info.items():
        yp2  = (y_sc >= t["threshold"]).astype(int)
        cm2  = confusion_matrix(y_te, yp2, labels=[0,1])
        tn2,fp2,fn2,tp2 = cm2.ravel()
        acc2 = (tp2+tn2)/len(y_te)
        print(f"  {key:<10} {t['threshold']:>6.4f} "
              f"{t.get('dice_image', t['f1']):>8.4f} "
              f"{t['recall']:>6.4f}  acc={acc2:.4f}")
        results_thr[key] = {**t, "accuracy":round(acc2,4),
                            "tn":int(tn2),"fp":int(fp2),
                            "fn":int(fn2),"tp":int(tp2)}

    imp_arr, _ = get_feature_importance(model)
    if imp_arr is not None and feature_names:
        print_feature_importance(imp_arr, feature_names)

    if HAS_MPL:
        _plot_eval(name, y_sc, y_te, cm, auc, ap, acc, f1v,
                   thr_info, pr_data, out_dir, feature_names, imp_arr)

    return {"model":name,
            "accuracy":round(float(acc),4), "roc_auc":round(float(auc),4),
            "avg_prec":round(float(ap),4),  "sensitivity":round(float(sens),4),
            "specificity":round(float(spec),4), "f1":round(float(f1v),4),
            "iou_image":round(float(img["mean_iou"]),4),
            "dice_image":round(float(img["mean_dice"]),4),
            "iou_image_std":round(float(img["std_iou"]),4),
            "iou_patch":round(float(iou),4), "dice_patch":round(float(dice),4),
            "decision_threshold":round(float(decision_threshold),4),
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
            "thresholds":results_thr}


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
                        label=f"{key} thr={t['threshold']:.3f}")
    axes[1].fill_between([0.8,1],[0.8,0.8],[1,1],alpha=0.08,color="green")
    for v in [0.8]:
        axes[1].axhline(v,color="gray",ls="--",lw=1,alpha=0.5)
        axes[1].axvline(v,color="gray",ls="--",lw=1,alpha=0.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  AP={ap:.4f}", fontsize=9)
    axes[1].legend(fontsize=6); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)

    fpr, tpr, _ = roc_curve(y_te, y_sc)
    axes[2].plot(fpr, tpr, lw=2.5, color="#59a14f")
    axes[2].plot([0,1],[0,1],"k--",lw=1)
    axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
    axes[2].set_title(f"ROC  AUC={auc:.4f}", fontsize=9)
    axes[2].grid(alpha=0.3)

    if imp_arr is not None and feature_names:
        sidx   = np.argsort(imp_arr)
        names_ = [feature_names[i] for i in sidx]
        vals_  = imp_arr[sidx]
        colors = ["#e15759" if n in DERIVED_FEATURE_NAMES else
                  "#f28e2b" if n in GEO_FEATURE_NAMES else "#4e79a7"
                  for n in names_]
        axes[3].barh(names_, vals_, color=colors)
        axes[3].set_title("Feature Importance\n(đỏ=derived  cam=geometric)", fontsize=8)
        axes[3].grid(alpha=0.2, axis="x")

        d_t = sum(imp_arr[i] for i,n in enumerate(feature_names) if n in DERIVED_FEATURE_NAMES)
        g_t = sum(imp_arr[i] for i,n in enumerate(feature_names) if n in GEO_FEATURE_NAMES)
        b_t = max(1.0 - d_t - g_t, 0.0)
        axes[4].pie([d_t,g_t,b_t],
                    labels=[f"Derived\n{d_t:.1%}",
                            f"Geo\n{g_t:.1%}",
                            f"GLCM\n{b_t:.1%}"],
                    colors=["#e15759","#f28e2b","#4e79a7"],
                    autopct="%1.1f%%", startangle=90)
        axes[4].set_title("Feature Group", fontsize=9)
    else:
        for ax in axes[3:]:
            ax.text(0.5,0.5,"N/A",ha="center",va="center",
                    transform=ax.transAxes)

    fig.suptitle(f"Model: {name}", fontsize=11, y=1.02)
    plt.tight_layout()
    sn   = name.lower().replace(" ","_").replace("(","").replace(")","")
    path = os.path.join(out_dir, f"{sn}_eval.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [plot] → {path}")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

def print_summary(all_metrics, cv_results=None):
    print(f"\n{'═'*96}\n  TỔNG KẾT\n{'═'*96}")
    H = (f"  {'Model':<36} {'Acc':>7} {'AUC':>7} {'F1':>7} "
         f"{'Sens':>7} {'ImgIoU':>8} {'ImgDice':>8}")
    print(H); print(f"  {'─'*88}")
    for m in all_metrics:
        ok_s = "✅" if m["sensitivity"]     >= 0.80 else "  "
        ok_f = "✅" if m["f1"]              >= 0.75 else "  "
        ok_i = "✅" if m.get("iou_image",0) >= 0.40 else "  "
        print(f"  {m['model']:<36} {m['accuracy']:>7.4f} {m['roc_auc']:>7.4f} "
              f"{m['f1']:>7.4f} {m['sensitivity']:>7.4f} "
              f"{m.get('iou_image',0):>8.4f} {m.get('dice_image',0):>8.4f}  "
              f"S{ok_s} F{ok_f} I{ok_i}")
    if cv_results:
        print(f"\n  CV (5-fold image-level):")
        for nm, cv in cv_results.items():
            print(f"  {nm:<36} IoU={cv['mean_iou']:.4f}  "
                  f"Dice={cv['mean_dice']:.4f}  "
                  f"AUC={cv['mean_auc']:.4f}  "
                  f"thr={cv['median_best_threshold']:.4f}")
    best = max(all_metrics,
               key=lambda m:(m["sensitivity"]+m["f1"]+m.get("iou_image",0))/3)
    print(f"\n  ★ Best: {best['model']}")
    print(f"\n  Threshold → Predict.py:")
    for m in all_metrics:
        for key in ["best_f1","sens80","sens90"]:
            if key in m.get("thresholds",{}):
                t = m["thresholds"][key]
                d = t.get("dice_image", t.get("f1",0))
                print(f"  {m['model'][:18]:<20} {key:<10} "
                      f"thr={t['threshold']:.4f}  "
                      f"sens={t['recall']:.3f}  ImgDice={d:.3f}")
    print(f"{'═'*96}")


# ═══════════════════════════════════════════════════════════════════
# MAIN RUN
# ═══════════════════════════════════════════════════════════════════

def run(dataset_path, model_choice="stack", test_size=0.20,
        do_cv=True, do_smote=False, n_cv_folds=5):

    # FIX 11: ép tắt SMOTE ngay khi vào run()
    if do_smote:
        print("[run] ⛔ FIX 11: SMOTE bị tắt cứng — sẽ chạy KHÔNG SMOTE.")
    do_smote = False

    out_dir = str(Path(dataset_path).parent / "models")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load + feature engineering + SELECT 8 features ─────────
    X, Y, fns, ratio, stems = load_dataset(dataset_path)
    print(f"\n[features] +{len(DERIVED_FEATURE_NAMES)} derived "
          f"(SHADOW_MAX_MEAN={SHADOW_MAX_MEAN})")
    X, fns = engineer_probe_features(X, fns)
    print(f"[features] Pool sau engineering: {X.shape[1]} cột "
          f"({len(BASE_FEATURE_NAMES)} base + {len(DERIVED_FEATURE_NAMES)} derived)")

    # FIX 11: chọn 8 features
    X, fns = select_features(X, fns, SELECTED_FEATURES)
    print(f"[features] ✅ SELECTED {len(fns)} features: {fns}")

    # Tính lại ratio sau khi select (không đổi nhưng để chắc chắn)
    feat_info = {"all_features":fns,
                 "selected_features":SELECTED_FEATURES,
                 "base_features":BASE_FEATURE_NAMES,
                 "derived_features":DERIVED_FEATURE_NAMES,
                 "n_total":len(fns),
                 "SHADOW_MAX_MEAN":SHADOW_MAX_MEAN}
    with open(os.path.join(out_dir,"feature_names.json"),"w") as f:
        json.dump(feat_info, f, indent=2)

    # ── 2. 80/20 split theo ảnh ────────────────────────────────────
    tv_m, te_m, te_bases = image_level_split(stems, test_size)
    X_tv, y_tv = X[tv_m], Y[tv_m]; stems_tv = stems[tv_m]
    X_te, y_te = X[te_m], Y[te_m]; stems_te = stems[te_m]

    with open(os.path.join(out_dir,"test_split_images.json"),"w") as f:
        json.dump({"test_images":te_bases,
                   "test_stems":sorted(np.unique(stems_te).astype(str).tolist())},
                  f, indent=2)
    print(f"\n[split] Train+Val={len(X_tv):,}  Test={len(X_te):,}")
    print(f"[split] Shadow TV={y_tv.sum():,}({y_tv.mean()*100:.1f}%)  "
          f"Te={y_te.sum():,}({y_te.mean()*100:.1f}%)")

    all_metrics=[]; cv_results={}

    # ── 3a. RF ─────────────────────────────────────────────────────
    if model_choice in ("rf","both"):
        print(f"\n{'─'*60}\n[RF] Baseline RandomForest\n{'─'*60}")
        rf_thr = 0.5
        if do_cv:
            print(f"[RF-CV] {n_cv_folds}-fold image-level CV...")
            # FIX 9: bỏ ratio
            cv = cross_validate_iou_dice(
                lambda _: build_rf(), X_tv, y_tv, stems_tv,
                n_splits=n_cv_folds, do_smote=do_smote)
            cv_results["RF"] = cv
            rf_thr = cv.get("median_best_threshold", 0.5)

        # FIX 11: không SMOTE
        Xf, yf = X_tv.copy(), y_tv.copy()
        t0 = time.time()
        rf = build_rf(); rf.fit(Xf, yf)
        oob = getattr(rf, "oob_score_", float("nan"))
        print(f"[RF] OOB={oob:.4f}  {time.time()-t0:.1f}s")
        m = evaluate("RandomForest", rf, X_te, y_te, out_dir, fns,
                     stems_te=stems_te, decision_threshold=rf_thr)
        all_metrics.append(m)
        joblib.dump(rf, os.path.join(out_dir,"rf_model.pkl"))
        print(f"[RF] ✅ rf_model.pkl")

    # ── 3b. Stacking ───────────────────────────────────────────────
    if model_choice in ("stack","both"):
        if not HAS_XGB: print("[ERROR] pip install xgboost"); sys.exit(1)
        print(f"\n{'─'*60}")
        print(f"[Stack] XGBoost + RF + MLP(LeakyReLU+FocalLoss) → LR")
        print(f"  GPU : {_DEVICE}  "
              f"({'CUDA' if torch.cuda.is_available() else 'CPU'})")
        print(f"{'─'*60}")

        stack_thr = 0.5
        if do_cv:
            print(f"[Stack-CV] {n_cv_folds}-fold image-level CV (inner cv=2)...")
            # FIX 8: stack_cv=2 trong outer CV
            cv = cross_validate_iou_dice(
                lambda r: build_stacking(r, stack_cv=2),
                X_tv, y_tv, stems_tv,
                n_splits=n_cv_folds, do_smote=do_smote)
            cv_results["Stacking"] = cv
            stack_thr = cv.get("median_best_threshold", 0.5)

        # FIX 11: không SMOTE
        Xf, yf = X_tv.copy(), y_tv.copy()
        print(f"\n[Stack] Training full Train+Val "

              f"({len(Xf):,} patches, {X_tv.shape[1]} features, inner cv=3)...")
        print(f"        Ước tính: 20–50 phút...")
        t0 = time.time()
        # FIX 8: stack_cv=3 cho final training (full 80% data)
        stack = build_stacking(class_ratio(yf), stack_cv=3)
        stack.fit(Xf, yf)
        elapsed = time.time() - t0
        print(f"[Stack] Done {elapsed:.1f}s ({elapsed/60:.1f} phút)")

        m = evaluate("Stacking(XGB+RF+MLP)", stack, X_te, y_te, out_dir, fns,
                     stems_te=stems_te, decision_threshold=stack_thr)
        all_metrics.append(m)
        joblib.dump(stack, os.path.join(out_dir,"stacking_model.pkl"))
        print(f"[Stack] ✅ stacking_model.pkl")

        try:
            clf   = stack.named_steps["classifier"]
            rf_in = clf.named_estimators_["rf"]
            joblib.dump(rf_in, os.path.join(out_dir,"rf_model.pkl"))
            print(f"[Stack] ✅ inner RF → rf_model.pkl")
        except Exception as e:
            print(f"[Stack] ⚠ inner RF: {e}")

    print_summary(all_metrics, cv_results if do_cv else None)

    # ── 4. Lưu ─────────────────────────────────────────────────────
    clean = [{k:v for k,v in m.items() if k!="y_score"}
             for m in all_metrics]
    if do_cv:
        for nm, cv in cv_results.items():
            for m in clean:
                ml = m["model"].lower()
                if nm.lower() in ml or (nm=="RF" and "randomforest" in ml):
                    m["cv_validation"] = {k:v for k,v in cv.items()
                                          if k!="folds"}
    with open(os.path.join(out_dir,"results.json"),"w",encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  [results] → {out_dir}/results.json")
    print(f"\n  ⚠ Predict.py phải import từ Train.py:")
    print(f"    from Train import engineer_probe_features, SHADOW_MAX_MEAN")
    return all_metrics


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train shadow: XGB+RF+MLP(FocalLoss), image-level IoU CV")
    ap.add_argument("--dataset",
        default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed\dataset.npz")
    ap.add_argument("--model",     default="stack",
                    choices=["rf","stack","both"])
    ap.add_argument("--test_size", default=0.2,  type=float)
    ap.add_argument("--cv",        action="store_true",
        help="5-fold CV image-level (tune theo Image IoU/Dice)")
    ap.add_argument("--smote",     action="store_true",
        help="(LEGACY) FIX 11: cờ này bị bỏ qua — SMOTE luôn tắt.")
    ap.add_argument("--n_folds",   default=5, type=int)
    args = ap.parse_args()
    if args.smote:
        print("[CLI] ⛔ --smote bị bỏ qua (FIX 11: SMOTE đã được loại bỏ).")
        args.smote = False

    print("=" * 70)
    print("  TRAIN — Shadow Detection (11 fixes applied)")
    print("  Stacking: XGBoost + BalancedRF + MLP(LeakyReLU+FocalLoss) → LR")
    print(f"  Features: SELECTED {N_FEATURES_SELECTED} / pool {N_FEATURES_TOTAL}")
    print(f"     → {SELECTED_FEATURES}")
    print(f"  SHADOW_MAX_MEAN = {SHADOW_MAX_MEAN} (FIX 11)")
    print("=" * 70)
    print(f"  dataset   : {args.dataset}")
    print(f"  model     : {args.model}")
    print(f"  CV        : {'5-fold image-level' if args.cv else 'off'}")
    print(f"  SMOTE     : OFF (FIX 11 — luôn tắt)")
    print(f"  GPU       : {_DEVICE}")
    print(f"  XGBoost   : {'✅ max_depth=4 colsample=1.0 max_delta_step=1' if HAS_XGB else '❌ pip install xgboost'}")
    print(f"  RF        : {'BalancedRandomForest (undersampling)' if HAS_BRF else 'RF balanced_subsample (fallback)'}")
    print(f"  MLP       : hidden=(64,32)  batch=128  FocalLoss(γ=2)")
    print(f"  Stack-CV  : inner=2(outer CV) / 3(final train)")
    print("=" * 70 + "\n")

    run(args.dataset, args.model, args.test_size,
        do_cv=args.cv, do_smote=args.smote, n_cv_folds=args.n_folds)