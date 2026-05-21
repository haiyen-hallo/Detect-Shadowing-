"""
Train.py — Tối ưu hóa: Khung Stacking (HGBT + GBT + RF)
Phiên bản sửa lỗi toàn diện:
  [FIX-1] Thêm absolute_dark gate: chặn false positive speckle sáng
  [FIX-2] Thêm vray_dark_score: phát hiện bóng rộng (không phụ thuộc lateral_drop)
  [FIX-3] Bỏ class_weight=balanced khỏi RF (xung đột với FHL sample_weight)
  [FIX-4] Mở rộng quét threshold 0.30→0.65 bước 0.01
  [FIX-5] Metric kết hợp Dice+IoU khi dò threshold
  [FIX-6] passthrough=True cho StackingClassifier
  [FIX-7] Post-processing morphological closing per image
"""

import os, sys

if os.name == "nt":
    os.environ["LOKY_MAX_CPU_COUNT"]  = "14"
    os.environ["JOBLIB_BACKEND"]      = "threading"
    os.environ["OMP_NUM_THREADS"]     = "14"
    os.environ["MKL_NUM_THREADS"]     = "14"

import json, argparse, time, warnings
import numpy as np
from pathlib import Path

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
import joblib

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ═══════════════════════════════════════════════════════════════════
# HẰNG SỐ CONFIG
# ═══════════════════════════════════════════════════════════════════

# [FIX-1] Ngưỡng tối tuyệt đối: patch phải có mean < ngưỡng này mới là bóng
ABSOLUTE_DARK_THRESHOLD = 0.38

# 13 FEATURES (10 cũ + 3 mới)
SELECTED_FEATURES = [
    # --- Positional ---
    "row_ratio", "angle_axis_norm", "col_ratio",
    # --- Core intensity signals ---
    "mean", "contrast", "ray_mean_above",
    # --- Continuity signals ---
    "vert_continuity",
    # --- Derived (cũ) ---
    "depth_darkness", "is_shadow_candidate", "ray_dark_context",
    # --- [NEW] Absolute darkness gate (chặn speckle sáng) ---
    "absolute_dark",
    # --- [NEW] Physics compound: sáng trên + tối dưới + tối liên tục (phát hiện bóng rộng) ---
    "vray_dark_score",
    # --- [NEW] Shadow zone: kết hợp ray_above * vert_continuity * (1-mean) ---
    "shadow_zone_score",
]

GEO_FEATURE_NAMES = {
    "dist_origin_norm", "angle_axis_norm",
    "ray_mean_above",   "lateral_drop",
    "vert_continuity",  "reverb_score",
}

# Tên derived features (cũ + mới)
DERIVED_FEATURE_NAMES = [
    # --- 13 features cũ ---
    "above_below_ratio", "shadow_drop", "dark_bright_above",
    "lateral_contrast_nm", "depth_darkness", "vert_dark_depth",
    "anti_reverb_dark", "phys_shadow_score",
    "col_dark_score", "depth_col_dark",
    "is_shadow_candidate", "shadow_contrast",
    "ray_dark_context",
    # --- [NEW] 3 features mới ---
    "absolute_dark",       # Hard gate: mean < ABSOLUTE_DARK_THRESHOLD
    "vray_dark_score",     # bright_above × current_dark × vert_cont (độc lập với lateral)
    "shadow_zone_score",   # ray_above × vert_cont × (1-mean)
]

ALPHA_POS   = 1.5   # Giảm từ 2.0 → ít aggressive hơn, tránh overfit shadow mạnh
NEG_WEIGHT  = 0.5
N_JOBS_RF   = 14


# ═══════════════════════════════════════════════════════════════════
# SPLITTING
# ═══════════════════════════════════════════════════════════════════

def _base_stem(stem: str) -> str:
    for suf in ["_flip", "_tx+5", "_tx-5", "_tx+10", "_tx-10"]:
        if stem.endswith(suf): return stem[: -len(suf)]
    return stem

def image_level_split(stems_all, test_ratio: float = 0.2, seed: int = 42):
    base_stems   = np.array([_base_stem(str(s)) for s in stems_all])
    unique_bases = np.unique(base_stems)
    n_total      = len(unique_bases)
    n_test       = max(1, int(n_total * test_ratio))

    rng         = np.random.default_rng(seed)
    idx         = rng.permutation(n_total)
    test_bases  = set(unique_bases[idx[:n_test]])
    train_bases = set(unique_bases[idx[n_test:]])

    train_mask = np.array([_base_stem(str(s)) in train_bases for s in stems_all])
    test_mask  = np.array([_base_stem(str(s)) in test_bases  for s in stems_all])

    print(f"[split] {n_total} ảnh gốc → train={len(train_bases)}  test={len(test_bases)}")
    return train_mask, test_mask, list(test_bases)

def image_level_kfold(stems_tv, n_splits=5, seed=42):
    base  = np.array([_base_stem(str(s)) for s in stems_tv])
    uniq  = np.unique(base)
    rng   = np.random.default_rng(seed)
    shuf  = rng.permutation(uniq)
    folds = np.array_split(shuf, n_splits)

    splits = []
    for k in range(n_splits):
        vb = set(folds[k])
        splits.append((
            np.where([b not in vb for b in base])[0],
            np.where([b in vb     for b in base])[0]
        ))
    return splits


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — ĐÃ SỬA LỖI
# ═══════════════════════════════════════════════════════════════════

def engineer_probe_features(X: np.ndarray, feature_names: list, stems: np.ndarray = None) -> tuple:
    fn  = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6
    def _c(name):
        return X[:, fn[name]].astype(np.float64) if name in fn else np.zeros(len(X), np.float64)

    mean      = _c("mean")
    ray_above = _c("ray_mean_above")
    lat_drop  = _c("lateral_drop")
    vert_cont = _c("vert_continuity")
    reverb    = _c("reverb_score")
    dist_norm = _c("dist_origin_norm")
    row_ratio = _c("row_ratio")

    # ─── Adaptive max mean (theo từng ảnh) ────────────────────────
    adaptive_max_mean = np.zeros_like(mean)
    if stems is not None:
        unique_stems = np.unique(stems)
        for stem in unique_stems:
            idx = (stems == stem)
            adaptive_max_mean[idx] = np.clip(np.median(mean[idx]) * 0.85, 0.15, 0.50)
    else:
        adaptive_max_mean[:] = np.clip(np.median(mean) * 0.85, 0.15, 0.50)

    darkness     = np.clip((adaptive_max_mean - mean) / (adaptive_max_mean + eps), 0.0, 1.0)
    above_below  = np.clip(ray_above / (mean + eps), 0.0, 10.0)
    shadow_drop  = np.clip(ray_above - mean, 0.0, 1.0)
    dark_bright  = darkness * np.clip(ray_above, 0.0, 1.0)
    lat_norm     = np.clip(lat_drop / (mean + eps), -2.0, 5.0)
    depth_dark   = dist_norm * np.clip(1.0 - mean, 0.0, 1.0)
    vert_depth   = vert_cont * dist_norm
    anti_reverb  = np.clip(1.0 - reverb, 0.0, 1.0) * np.clip(1.0 - mean, 0.0, 1.0)
    phys         = np.clip(np.clip(above_below / 5.0, 0.0, 1.0) * shadow_drop * darkness, 0.0, 1.0)
    col_dark     = darkness * np.clip(0.4 + 0.6 * row_ratio, 0.4, 1.0)
    depth_col    = dist_norm * col_dark

    # [FIX-1] is_shadow_candidate: thêm absolute_dark gate
    # OLD: darkness * clip((above_below-1)/2, 0, 1)  ← bị false positive speckle
    # NEW: thêm điều kiện mean phải thực sự tối (<= ABSOLUTE_DARK_THRESHOLD)
    abs_gate  = np.clip((ABSOLUTE_DARK_THRESHOLD - mean) / (ABSOLUTE_DARK_THRESHOLD + eps), 0.0, 1.0)
    is_shadow = abs_gate * darkness * np.clip((above_below - 1.0) / 2.0, 0.0, 1.0)

    shadow_cont       = np.clip(above_below / 5.0, 0.0, 1.0) * np.clip(lat_drop, 0.0, 1.0)
    ray_dark_context  = np.clip(vert_cont * darkness * np.clip(ray_above, 0.0, 1.0), 0.0, 1.0)

    # ─── [NEW] 3 features mới ─────────────────────────────────────

    # [FIX-1] absolute_dark: cứng — patch phải có mean < ngưỡng tuyệt đối
    # Chặn bright speckle (ảnh 1): patch sáng sẽ có absolute_dark ≈ 0
    absolute_dark = np.clip((ABSOLUTE_DARK_THRESHOLD - mean) / (ABSOLUTE_DARK_THRESHOLD + eps), 0.0, 1.0)

    # [FIX-2] vray_dark_score: tín hiệu vật lý thuần túy
    # Phát hiện: "có cấu trúc sáng phía trên" VÀ "patch hiện tại tối" VÀ "bên dưới tiếp tục tối"
    # ★ Không phụ thuộc lateral_drop → hoạt động với cả bóng rộng lẫn hẹp (ảnh 2)
    bright_above_sig = np.clip((ray_above - 0.35) / (0.35 + eps), 0.0, 1.0)
    current_dark_sig = np.clip((0.40 - mean) / (0.40 + eps), 0.0, 1.0)
    vray_dark_score  = np.clip(bright_above_sig * current_dark_sig * vert_cont, 0.0, 1.0)

    # [FIX-2] shadow_zone_score: phiên bản đơn giản hơn, bắt cả bóng rộng
    # ray_above × vert_continuity × (1-mean): cao khi có sáng trên + tối liên tục xuống
    shadow_zone_score = np.clip(ray_above * vert_cont * np.clip(1.0 - mean, 0.0, 1.0), 0.0, 1.0)

    new = np.column_stack([
        above_below, shadow_drop, dark_bright, lat_norm, depth_dark, vert_depth, anti_reverb,
        phys, col_dark, depth_col, is_shadow, shadow_cont, ray_dark_context,
        # 3 features mới
        absolute_dark, vray_dark_score, shadow_zone_score,
    ]).astype(np.float32)

    return np.hstack([X, np.nan_to_num(new, nan=0., posinf=1., neginf=-1.)]), feature_names + DERIVED_FEATURE_NAMES


def select_features(X: np.ndarray, feature_names: list, selected: list = None) -> tuple:
    if selected is None: selected = SELECTED_FEATURES
    fn_map = {name: i for i, name in enumerate(feature_names)}
    missing = [s for s in selected if s not in fn_map]
    if missing:
        print(f"  [WARN] Features không tìm thấy, bỏ qua: {missing}")
        selected = [s for s in selected if s in fn_map]
    idx = [fn_map[s] for s in selected]
    return X[:, idx].astype(X.dtype, copy=False), list(selected)


def load_dataset(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int32)
    stems         = d["stems"] if "stems" in d else np.array([str(i) for i in range(len(Y))])
    feature_names = [str(s) for s in d["feature_names"]]
    Y_soft        = d["Y_soft"].astype(np.float32) if "Y_soft" in d else Y.astype(np.float32)

    if np.isnan(X).sum() + np.isinf(X).sum() > 0:
        X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)

    X, feature_names = engineer_probe_features(X, feature_names, stems)
    X, feature_names = select_features(X, feature_names, SELECTED_FEATURES)
    print(f"\n[Features] Sử dụng {len(feature_names)} đặc trưng: {feature_names}")

    n_pos = int(Y.sum())
    ratio = (len(Y) - n_pos) / max(n_pos, 1)
    return X, Y, Y_soft, feature_names, ratio, stems


# ═══════════════════════════════════════════════════════════════════
# MODELS & FHL WEIGHTS
# ═══════════════════════════════════════════════════════════════════

def compute_fhl_weights(Y_soft, Y_binary, alpha=ALPHA_POS, neg_weight=NEG_WEIGHT):
    w = np.zeros(len(Y_soft), dtype=np.float64)
    pos_idx = np.where(Y_binary == 1)[0]
    neg_idx = np.where(Y_binary == 0)[0]
    n_neg   = max(len(neg_idx), 1)

    for idx in pos_idx:
        w[idx] = float(np.clip(Y_soft[idx], 1e-7, 1.0)) ** alpha
    for idx in neg_idx:
        w[idx] = (1.0 - float(np.clip(Y_soft[idx], 0, 1.0))) * neg_weight / n_neg

    w_max = w.max()
    return np.clip(w / w_max if w_max > 1e-10 else w, 1e-6, 1.0).astype(np.float32)


def build_rf(ratio=None):
    # [FIX-3] Bỏ class_weight="balanced" — xung đột với FHL sample_weight
    # class_weight và sample_weight cùng điều chỉnh trọng số → model under-confident
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        # class_weight="balanced",  ← ĐÃ XÓA: xung đột với FHL
        n_jobs=N_JOBS_RF,
        random_state=42,
        oob_score=True,
    )


def build_stacking(ratio, meta_c=0.5):
    # HGBT: giữ class_weight vì không dùng sample_weight
    hgbt = HistGradientBoostingClassifier(
        max_iter=500,
        max_leaf_nodes=31,
        learning_rate=0.03,       # Chậm hơn → học tốt hơn
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )

    # GBT: thêm n_estimators nhiều hơn để bù class_weight=None
    gbt = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        min_samples_leaf=8,
        subsample=0.8,
        random_state=42,
    )

    # [FIX-3] RF trong stacking: bỏ class_weight vì stacking không pass sample_weight
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",   # Giữ vì stacking không dùng FHL
        n_jobs=N_JOBS_RF,
        random_state=42,
        oob_score=True,
    )

    meta_lr = LogisticRegression(C=meta_c, class_weight="balanced", max_iter=2000, random_state=42)

    stacking = StackingClassifier(
        estimators=[("hgbt", hgbt), ("gbt", gbt), ("rf", rf)],
        final_estimator=meta_lr,
        cv=5,
        passthrough=True,   # [FIX-6] Meta-learner thấy cả features gốc → thông tin phong phú hơn
        n_jobs=1,
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", stacking)])


def fit_with_weight(model, X_tr, y_tr, sw):
    """FHL chỉ áp dụng cho RF đơn lẻ. Stacking tự cân bằng qua class_weight."""
    if isinstance(model, Pipeline) and "classifier" in model.named_steps:
        if isinstance(model.named_steps["classifier"], StackingClassifier):
            print("  [Info] Stacking: dùng class_weight nội bộ (không dùng FHL sample_weight).")
            model.fit(X_tr, y_tr)
            return

    if sw is None:
        model.fit(X_tr, y_tr)
        return

    try:
        if hasattr(model, "named_steps"):
            model.fit(X_tr, y_tr, classifier__sample_weight=sw)
        else:
            model.fit(X_tr, y_tr, sample_weight=sw)
    except TypeError:
        model.fit(X_tr, y_tr)


# ═══════════════════════════════════════════════════════════════════
# METRICS & THRESHOLD SEARCH — ĐÃ SỬA
# ═══════════════════════════════════════════════════════════════════

def _bin_stats(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {
        "precision": pr, "recall": rc,
        "iou":  tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def image_mask_metrics(yt, yp, stems):
    unique = np.unique(np.asarray(stems, str))
    ious, dices = [], []
    for s in unique:
        idx = np.where(np.asarray(stems, str) == s)[0]
        st  = _bin_stats(yt[idx], yp[idx])
        ious.append(st["iou"]); dices.append(st["dice"])
    return {
        "mean_iou":   float(np.mean(ious)),
        "mean_dice":  float(np.mean(dices)),
        "n_images":   len(unique),
    }


def find_best_threshold(y_score, y_te, stems, thr_min=0.30, thr_max=0.65):
    """
    [FIX-4] Quét bước 0.01 (thay vì 0.025), dải rộng hơn (0.30→0.65)
    [FIX-5] Metric kết hợp 0.6×Dice + 0.4×IoU (thay vì chỉ Dice)
    """
    n_steps = int(round((thr_max - thr_min) / 0.01)) + 1
    grid    = np.linspace(thr_min, thr_max, n_steps)

    best_thr, best_score = thr_min, 0.0
    for thr in grid:
        yp      = (y_score >= thr).astype(np.int32)
        im      = image_mask_metrics(y_te, yp, stems)
        # [FIX-5] Kết hợp cả Dice và IoU tránh bias về 1 metric
        combined = 0.6 * im["mean_dice"] + 0.4 * im["mean_iou"]
        if combined > best_score:
            best_score, best_thr = combined, thr

    return best_thr


def cross_validate_and_find_threshold(
    model_builder, X_tv, y_tv, ys_tv, stems_tv, ratio,
    n_splits=5, use_fhl=True, meta_c=0.5, thr_min=0.30, thr_max=0.65
):
    print(f"\n[Validation] {n_splits}-Fold CV | Quét threshold [{thr_min:.2f}, {thr_max:.2f}] bước 0.01 ...")
    splits = image_level_kfold(stems_tv, n_splits=n_splits)
    thrs, ious, dices = [], [], []

    for k, (tr_idx, va_idx) in enumerate(splits):
        X_tr_, y_tr_, ys_tr_ = X_tv[tr_idx], y_tv[tr_idx], ys_tv[tr_idx]
        X_va_, y_va_, st_va_ = X_tv[va_idx], y_tv[va_idx], stems_tv[va_idx]

        sw = compute_fhl_weights(ys_tr_, y_tr_) if use_fhl else None

        if model_builder.__name__ == "build_rf":
            m = model_builder(ratio)
        else:
            m = model_builder(ratio, meta_c=meta_c)

        fit_with_weight(m, X_tr_, y_tr_, sw)

        y_prob   = m.predict_proba(X_va_)[:, 1]
        best_thr = find_best_threshold(y_prob, y_va_, st_va_, thr_min=thr_min, thr_max=thr_max)
        y_pred   = (y_prob >= best_thr).astype(np.int32)
        im       = image_mask_metrics(y_va_, y_pred, st_va_)

        thrs.append(best_thr); ious.append(im["mean_iou"]); dices.append(im["mean_dice"])
        print(f"  Fold {k+1}/{n_splits} | Thr={best_thr:.3f} | ImgIoU={im['mean_iou']:.4f} | ImgDice={im['mean_dice']:.4f}")

    median_thr = float(np.median(thrs))
    mean_iou   = float(np.mean(ious))
    mean_dice  = float(np.mean(dices))
    print(f"  => [CHỐT] Median Threshold = {median_thr:.4f} | CV ImgIoU={mean_iou:.4f} | CV ImgDice={mean_dice:.4f}")
    return median_thr


# ═══════════════════════════════════════════════════════════════════
# POST-PROCESSING — [MỚI] Morphological closing trên mask dự đoán
# ═══════════════════════════════════════════════════════════════════

def postprocess_mask_per_image(y_pred, y_score, stems, patch_h=16, patch_w=16, min_patch_area=4):
    """
    [FIX-7] Áp dụng morphological closing theo grid patch cho từng ảnh.
    Mục đích: lấp lỗ hổng giữa các cột shadow (ảnh 2) và loại speckle nhỏ (ảnh 1).

    Args:
        y_pred  : binary prediction (N_patches,)
        y_score : probability score (N_patches,)
        stems   : stem của từng patch
        patch_h/w: kích thước patch (để ước tính grid)
        min_patch_area: loại connected component nhỏ hơn số này (đơn vị: patches)
    Returns:
        y_pred_pp: prediction sau post-processing
    """
    try:
        from scipy import ndimage
    except ImportError:
        return y_pred  # Không có scipy → bỏ qua

    y_out = y_pred.copy()
    unique_stems = np.unique(np.asarray(stems, str))

    for s in unique_stems:
        idx  = np.where(np.asarray(stems, str) == s)[0]
        n    = len(idx)
        if n == 0: continue

        # Ước tính grid shape từ số patches và tỉ lệ ảnh siêu âm (~4:3 hoặc ~16:9)
        # Thử các tỉ lệ phổ biến: n_cols/n_rows ∈ [1.2, 1.5, 1.6, 2.0]
        best_shape = None
        for ratio_wh in [1.25, 1.33, 1.5, 1.6, 2.0]:
            n_rows_try = int(round((n / ratio_wh) ** 0.5))
            if n_rows_try < 1: continue
            n_cols_try = int(round(n / n_rows_try))
            if n_rows_try * n_cols_try == n:
                best_shape = (n_rows_try, n_cols_try)
                break
        # Fallback: tìm phân tích gần nhất
        if best_shape is None:
            for nr in range(int(n**0.5), 0, -1):
                if n % nr == 0:
                    best_shape = (nr, n // nr)
                    break
        if best_shape is None:
            continue

        n_rows, n_cols = best_shape
        grid = y_pred[idx].reshape(n_rows, n_cols)

        # Closing: lấp lỗ hổng theo chiều ngang (kernel nhỏ = 1×3)
        struct_h = ndimage.generate_binary_structure(2, 1)
        grid_closed = ndimage.binary_closing(grid, structure=struct_h, iterations=2).astype(np.int32)

        # Loại bỏ connected component nhỏ (false positive speckle đơn lẻ)
        labeled, n_comp = ndimage.label(grid_closed)
        for comp_id in range(1, n_comp + 1):
            if (labeled == comp_id).sum() < min_patch_area:
                grid_closed[labeled == comp_id] = 0

        y_out[idx] = grid_closed.ravel()

    return y_out


# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE & EVALUATE
# ═══════════════════════════════════════════════════════════════════

def get_feature_importance(model):
    try:
        clf = model.named_steps.get("classifier", model) if hasattr(model, "named_steps") else model
        if hasattr(clf, "estimators_"):
            for name, est in clf.estimators_:
                cand = est.steps[-1][1] if hasattr(est, "steps") else est
                if hasattr(cand, "feature_importances_"):
                    return cand.feature_importances_, name
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_, "model"
    except Exception:
        pass
    return None, None


def evaluate_on_test(name, model, X_te, y_te, stems_te, feature_names, val_threshold, out_dir,
                     apply_postprocess=True):
    print(f"\n{'═'*70}\n  [{name}] — ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP\n{'═'*70}")

    y_sc = model.predict_proba(X_te)[:, 1]

    # Prediction gốc
    y_pred_raw = (y_sc >= val_threshold).astype(int)

    # [FIX-7] Post-processing
    if apply_postprocess:
        y_pred = postprocess_mask_per_image(y_pred_raw, y_sc, stems_te)
        print(f"  [Post-process] Patches thay đổi sau closing: {int((y_pred != y_pred_raw).sum())}")
    else:
        y_pred = y_pred_raw

    print(f"  [Config] Threshold = {val_threshold:.4f}")
    print(classification_report(y_te, y_pred, target_names=["bg(0)", "shadow(1)"], digits=4))

    auc = roc_auc_score(y_te, y_sc)
    cm  = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    img    = image_mask_metrics(y_te, y_pred, stems_te)
    f1_val = f1_score(y_te, y_pred)

    # Cũng in kết quả raw (trước post-process) để so sánh
    img_raw = image_mask_metrics(y_te, y_pred_raw, stems_te)
    print(f"  Raw   (trước PP): ImgIoU={img_raw['mean_iou']:.4f} | ImgDice={img_raw['mean_dice']:.4f}")
    print(f"  Final (sau  PP): ImgIoU={img['mean_iou']:.4f}  | ImgDice={img['mean_dice']:.4f}  (n={img['n_images']} ảnh)")
    print(f"  Patch F1-Score = {f1_val:.4f} | ROC-AUC = {auc:.4f}")

    # Feature importance plot
    imp_arr, src_name = get_feature_importance(model)
    if imp_arr is not None and feature_names and HAS_MPL:
        # Nếu passthrough=True, số features trong meta-learner khác feature_names gốc
        # Chỉ vẽ nếu kích thước khớp
        if len(imp_arr) == len(feature_names):
            fig, ax = plt.subplots(figsize=(10, 6))
            sidx   = np.argsort(imp_arr)
            names  = [feature_names[i] for i in sidx]
            vals   = imp_arr[sidx]
            colors = ["#e15759" if n in GEO_FEATURE_NAMES else
                      "#59a14f" if n in {"absolute_dark", "vray_dark_score", "shadow_zone_score"} else
                      "#4e79a7"
                      for n in names]
            ax.barh(names, vals, color=colors)
            ax.set_title(f"Feature Importance ({name} / {src_name})\n"
                         f"đỏ=GEO, xanh lá=NEW, xanh dương=base", fontsize=10)
            ax.grid(alpha=0.2, axis="x")
            plt.tight_layout()
            path = os.path.join(out_dir, f"{name.lower().replace(' ', '_')}_importance.png")
            plt.savefig(path, dpi=130); plt.close()
            print(f"  [Plot] Feature importance → {path}")

    return {
        "model":            name,
        "val_threshold":    val_threshold,
        "accuracy":         (tp + tn) / len(y_te),
        "roc_auc":          round(float(auc), 4),
        "test_iou_image":   round(float(img["mean_iou"]), 4),
        "test_dice_image":  round(float(img["mean_dice"]), 4),
        "test_iou_raw":     round(float(img_raw["mean_iou"]), 4),
        "test_dice_raw":    round(float(img_raw["mean_dice"]), 4),
        "f1":               round(float(f1_val), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN RUN
# ═══════════════════════════════════════════════════════════════════

def run(dataset_path: str, model_choice: str = "stack", test_size: float = 0.2,
        do_cv: bool = False, use_fhl: bool = True, manual_thr: float = 0.55,
        meta_c: float = 0.5, thr_min: float = 0.30, thr_max: float = 0.65,
        apply_postprocess: bool = True):

    out_dir = str(Path(dataset_path).parent / "models")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, Y_soft, feature_names, ratio, stems = load_dataset(dataset_path)

    train_mask, test_mask, test_bases = image_level_split(stems, test_ratio=test_size)
    X_tr, y_tr = X[train_mask], Y[train_mask]
    ys_tr, stems_tr = Y_soft[train_mask], stems[train_mask]
    X_te, y_te, stems_te = X[test_mask], Y[test_mask], stems[test_mask]

    print(f"\n[Data] Train={len(X_tr):,} patches | Test={len(X_te):,} patches")
    print(f"       Shadow (train): {int(y_tr.sum()):,} / {len(y_tr):,} ({y_tr.mean()*100:.1f}%)")

    with open(os.path.join(out_dir, "test_split_images.json"), "w", encoding="utf-8") as f:
        json.dump({"test_images": test_bases}, f, indent=2)

    all_metrics = []

    # ─── Random Forest ────────────────────────────────────────────
    if model_choice in ("rf", "both"):
        print(f"\n{'='*55}\n[1] RANDOM FOREST\n{'='*55}")
        val_thr = manual_thr
        if do_cv:
            val_thr = cross_validate_and_find_threshold(
                build_rf, X_tr, y_tr, ys_tr, stems_tr, ratio,
                use_fhl=use_fhl, thr_min=thr_min, thr_max=thr_max
            )
        else:
            print(f"[Config] CV=TẮT. Threshold cố định = {val_thr:.4f}")

        print(f"\n[Final Train RF] {len(X_tr):,} mẫu ...")
        rf = build_rf(ratio)
        sw = compute_fhl_weights(ys_tr, y_tr) if use_fhl else None
        fit_with_weight(rf, X_tr, y_tr, sw)
        if hasattr(rf, "oob_score_"):
            print(f"  OOB Score = {rf.oob_score_:.4f}")

        m = evaluate_on_test("Random Forest", rf, X_te, y_te, stems_te, feature_names,
                             val_thr, out_dir, apply_postprocess=apply_postprocess)
        all_metrics.append(m)
        joblib.dump(rf, os.path.join(out_dir, "rf_model.pkl"))
        print(f"  [Saved] rf_model.pkl")

    # ─── Stacking ─────────────────────────────────────────────────
    if model_choice in ("stack", "both"):
        print(f"\n{'='*55}\n[2] STACKING (HGBT + GBT + RF)\n{'='*55}")
        print(f"[Config] Meta-LR C={meta_c} | passthrough=True")
        val_thr = manual_thr
        if do_cv:
            val_thr = cross_validate_and_find_threshold(
                build_stacking, X_tr, y_tr, ys_tr, stems_tr, ratio,
                use_fhl=use_fhl, meta_c=meta_c, thr_min=thr_min, thr_max=thr_max
            )
        else:
            print(f"[Config] CV=TẮT. Threshold cố định = {val_thr:.4f}")

        print(f"\n[Final Train Stacking] {len(X_tr):,} mẫu ...")
        t0    = time.time()
        stack = build_stacking(ratio, meta_c=meta_c)
        sw    = compute_fhl_weights(ys_tr, y_tr) if use_fhl else None
        fit_with_weight(stack, X_tr, y_tr, sw)
        print(f"  Hoàn tất ({time.time()-t0:.1f}s)")

        m = evaluate_on_test("Stacking Ensemble", stack, X_te, y_te, stems_te, feature_names,
                             val_thr, out_dir, apply_postprocess=apply_postprocess)
        all_metrics.append(m)
        joblib.dump(stack, os.path.join(out_dir, "stacking_model.pkl"))
        print(f"  [Saved] stacking_model.pkl")

    # ─── Tổng kết ─────────────────────────────────────────────────
    print(f"\n{'═'*90}")
    print(f"  TỔNG KẾT — BLIND TEST")
    print(f"{'═'*90}")
    header = f"  {'Model':<22} {'Thr':>6} {'IoU(raw)':>10} {'Dice(raw)':>11} {'IoU(PP)':>9} {'Dice(PP)':>10} {'F1':>8}"
    print(header)
    print(f"  {'─'*88}")
    for m in all_metrics:
        print(f"  {m['model']:<22} {m['val_threshold']:>6.3f} "
              f"{m['test_iou_raw']:>10.4f} {m['test_dice_raw']:>11.4f} "
              f"{m['test_iou_image']:>9.4f} {m['test_dice_image']:>10.4f} "
              f"{m['f1']:>8.4f}")
    print(f"{'═'*90}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] results.json → {out_dir}")


# ═══════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shadow — Stacking + Fixed Features")
    parser.add_argument("--dataset", default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed\dataset.npz")
    parser.add_argument("--model",   default="stack", choices=["rf", "stack", "both"])
    parser.add_argument("--test_size", default=0.2, type=float)

    # CV
    parser.add_argument("--cv",  action="store_true", help="Bật 5-Fold CV để dò threshold")
    parser.add_argument("--thr", default=0.55, type=float,
                        help="Threshold cố định nếu tắt CV (Mặc định: 0.55 ← hạ từ 0.7)")

    # [FIX-4] Dải quét rộng hơn, mặc định 0.30→0.65
    parser.add_argument("--thr_min", default=0.30, type=float,
                        help="Threshold quét nhỏ nhất (Mặc định: 0.30)")
    parser.add_argument("--thr_max", default=0.65, type=float,
                        help="Threshold quét lớn nhất (Mặc định: 0.65)")

    parser.add_argument("--meta_c",  default=0.5, type=float,
                        help="C cho Logistic Regression meta-learner (Mặc định: 0.5)")
    parser.add_argument("--no_fhl",  action="store_true", help="Tắt FHL weights")
    parser.add_argument("--no_pp",   action="store_true", help="Tắt post-processing morphological closing")
    args = parser.parse_args()

    print("=" * 80)
    print("  TRAIN — SHADOW DETECTION (DATN) — v2 FIXED")
    print("  Framework: HGBT + GBT + RF (Stacking Ensemble)")
    print(f"  Features : {len(SELECTED_FEATURES)} ({', '.join(SELECTED_FEATURES)})")
    if args.cv:
        print(f"  CV       : BẬT | Quét [{args.thr_min}, {args.thr_max}] bước 0.01")
    else:
        print(f"  CV       : TẮT | Threshold cố định = {args.thr}")
    print(f"  Meta C   : {args.meta_c}")
    print(f"  FHL      : {'TẮT' if args.no_fhl else 'BẬT'}")
    print(f"  Post-PP  : {'TẮT' if args.no_pp  else 'BẬT (morphological closing)'}")
    print("=" * 80 + "\n")

    run(
        dataset_path      = args.dataset,
        model_choice      = args.model,
        test_size         = args.test_size,
        do_cv             = args.cv,
        use_fhl           = not args.no_fhl,
        manual_thr        = args.thr,
        meta_c            = args.meta_c,
        thr_min           = args.thr_min,
        thr_max           = args.thr_max,
        apply_postprocess = not args.no_pp,
    )