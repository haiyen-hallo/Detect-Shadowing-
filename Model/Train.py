"""
Train.py — Tối ưu hóa: Khung Stacking (HGBT + GBT + RF)
Phiên bản: ĐÃ LOẠI BỎ GEOMETRIC + BỔ SUNG COLUMN-CONTEXT FEATURES

CHANGELOG so với phiên bản cũ:
  [FIX-BUG-1] load_dataset: Y_soft fallback → dùng "coverage" thay vì Y_binary
               → FHL weighting hoạt động đúng với soft label (coverage ratio)
  [FIX-BUG-2] SELECTED_FEATURES: bổ sung "below_bright" vốn đã có trong dataset
               nhưng bị bỏ sót, không đưa vào model
  [NEW-F1]    Thêm "above_max_mean"  — độ sáng cực đại phía trên cùng cột
  [NEW-F2]    Thêm "col_dark_ratio"  — mức tối tương đối so với trung bình cột
  [NEW-F3]    Thêm "mean_drop"       — độ sụt giảm mean đột ngột (boundary bóng)
  [NEW-F4]    Thêm "std"             — độ đồng đều (bóng cản rất đồng nhất, std thấp)
  [KEEP]      absolute_dark, col_dark_score (derived trong engineer_probe_features)
  [KEEP]      FIX-1 ~ FIX-7 từ phiên bản trước
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

# Ngưỡng tối tuyệt đối: patch phải có mean < ngưỡng này mới là bóng
ABSOLUTE_DARK_THRESHOLD = 0.38

# ─── SELECTED_FEATURES — ĐÃ CẬP NHẬT ────────────────────────────
# Nguyên tắc lựa chọn: mỗi feature phải trả lời 1 câu hỏi vật lý khác nhau
# về bóng cản siêu âm (acoustic shadow)
SELECTED_FEATURES = [
    # --- [1] VỊ TRÍ KHÔNG GIAN ---
    # Bóng cản thường xuất hiện ở vùng sâu hơn (row_ratio cao)
    # và lệch sang bên (col_ratio tương quan với góc tia)
    "row_ratio",
    "col_ratio",

    # --- [2] CƯỜNG ĐỘ TUYỆT ĐỐI ---
    # mean: feature quan trọng nhất, bóng cản PHẢI tối
    # std:  bóng cản rất đồng đều (std thấp) — phân biệt với speckle noise
    "mean",
    "std",

    # --- [3] KẾT CẤU GLCM ---
    # Bóng cản có kết cấu rất khác vùng mô bình thường:
    # - homogeneity cao (patch đồng đều)
    # - energy cao      (phân phối pixel tập trung)
    # - correlation cao (pixel lân cận tương tự nhau)
    # - entropy thấp    (ít thông tin = ít cấu trúc = chỉ là tối thuần)
    "contrast",
    "homogeneity",
    "energy",
    "correlation",
    "entropy",

    # --- [4] GATE TUYỆT ĐỐI (derived trong engineer_probe_features) ---
    # absolute_dark: chặn false positive — patch PHẢI tối dưới ngưỡng cứng 0.38
    # col_dark_score: darkness tương đối × depth_weight → tổng hợp mean + position
    "absolute_dark",
    "col_dark_score",

    # --- [5] TÍN HIỆU REFLECTOR (BUG FIX: below_bright đã có nhưng bị bỏ sót) ---
    # below_bright: tỉ lệ pixel SÁNG (>0.70) phía TRÊN patch hiện tại
    # → đây là tín hiệu quan trọng nhất: bóng cản PHẢI có reflector sáng bên trên
    # → feature này đã được tính sẵn trong Noise.py nhưng chưa được dùng!
    "below_bright",

    # --- [6] NGỮ CẢNH CỘT — MỚI HOÀN TOÀN (tính trong Noise.py) ---
    # above_max_mean: độ sáng tối đa của tất cả patch phía trên cùng cột
    #   → capture hyperechoic reflector: nếu có patch rất sáng trên → bóng cản bên dưới
    #   → mean đơn lẻ không làm được điều này
    "above_max_mean",

    # col_dark_ratio: patch này tối hơn trung bình cột bao nhiêu (0→1)
    #   → phân biệt cột bóng cản (tối đồng đều) với cột mô bình thường (có speckle)
    #   → lý do: speckle ngẫu nhiên nên một patch tối không làm cho cả cột tối
    "col_dark_ratio",

    # mean_drop: sụt giảm mean đột ngột từ patch liền trên → patch hiện tại
    #   → capture ranh giới khởi đầu bóng cản (entry point of shadow)
    #   → kết hợp với above_max_mean: sáng → tối đột ngột = bóng cản thật
    "mean_drop",
]

# Không còn geometric features
GEO_FEATURE_NAMES = set()

# Các đặc trưng phái sinh (tính trong engineer_probe_features, không có trong Noise.py)
DERIVED_FEATURE_NAMES = [
    "col_dark_score",   # darkness tương đối × depth_weight
    "absolute_dark",    # gate cứng theo ngưỡng tuyệt đối
]

# Nhóm màu cho feature importance chart
_COL_CONTEXT_FEATURES = {"above_max_mean", "col_dark_ratio", "mean_drop"}
_GLCM_FEATURES        = {"homogeneity", "energy", "correlation", "entropy", "contrast"}
_REFLECTOR_FEATURES   = {"below_bright", "absolute_dark", "col_dark_score"}

ALPHA_POS  = 1.5
NEG_WEIGHT = 0.5
N_JOBS_RF  = 14


# ═══════════════════════════════════════════════════════════════════
# SPLITTING
# ═══════════════════════════════════════════════════════════════════

def _base_stem(stem: str) -> str:
    for suf in ["_flip", "_tx+5", "_tx-5", "_tx+10", "_tx-10"]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
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
            np.where([b in vb     for b in base])[0],
        ))
    return splits


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def engineer_probe_features(X: np.ndarray, feature_names: list,
                             stems: np.ndarray = None) -> tuple:
    """
    Tính thêm 2 đặc trưng phái sinh không có trong Noise.py:
      • col_dark_score  = darkness tương đối × depth_weight
                          → kết hợp mean + vị trí theo chiều sâu
      • absolute_dark   = gate cứng: (ABSOLUTE_DARK_THRESHOLD - mean) / threshold
                          → bằng 0 nếu patch không đủ tối → triệt tiêu false positive
    Lý do tính ở đây (không phải Noise.py): cần adaptive_max_mean per image,
    phải tính sau khi có đủ toàn bộ patches của từng ảnh trong batch.
    """
    fn  = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6

    def _c(name):
        return (X[:, fn[name]].astype(np.float64)
                if name in fn else np.zeros(len(X), np.float64))

    mean      = _c("mean")
    row_ratio = _c("row_ratio")

    # Adaptive max mean: median của ảnh × 0.85 (ngưỡng "sáng bình thường")
    # Tính per image để thích ứng với từng ảnh siêu âm (khác nhau về gain/brightness)
    adaptive_max_mean = np.zeros_like(mean)
    if stems is not None:
        for stem in np.unique(stems):
            idx = (stems == stem)
            adaptive_max_mean[idx] = np.clip(np.median(mean[idx]) * 0.85, 0.15, 0.50)
    else:
        adaptive_max_mean[:] = np.clip(np.median(mean) * 0.85, 0.15, 0.50)

    # col_dark_score: darkness tương đối × hệ số chiều sâu
    # depth_weight tăng theo row_ratio: patch càng sâu → bóng cản càng có khả năng
    darkness  = np.clip((adaptive_max_mean - mean) / (adaptive_max_mean + eps), 0., 1.)
    col_dark  = darkness * np.clip(0.4 + 0.6 * row_ratio, 0.4, 1.0)

    # absolute_dark: gate cứng, bằng 0 nếu mean >= ABSOLUTE_DARK_THRESHOLD
    absolute_dark = np.clip(
        (ABSOLUTE_DARK_THRESHOLD - mean) / (ABSOLUTE_DARK_THRESHOLD + eps), 0., 1.)

    new = np.column_stack([col_dark, absolute_dark]).astype(np.float32)
    return (
        np.hstack([X, np.nan_to_num(new, nan=0., posinf=1., neginf=-1.)]),
        feature_names + DERIVED_FEATURE_NAMES,
    )


def select_features(X: np.ndarray, feature_names: list,
                    selected: list = None) -> tuple:
    if selected is None:
        selected = SELECTED_FEATURES
    fn_map  = {name: i for i, name in enumerate(feature_names)}
    missing = [s for s in selected if s not in fn_map]
    if missing:
        print(f"  [WARN] Features không tìm thấy, bỏ qua: {missing}")
        selected = [s for s in selected if s in fn_map]
    idx = [fn_map[s] for s in selected]
    return X[:, idx].astype(X.dtype, copy=False), list(selected)


def load_dataset(npz_path: str):
    """
    [BUG FIX] Thứ tự ưu tiên khi load Y_soft:
      1. Key "Y_soft" trong file (nếu Noise.py mới đã lưu)
      2. Key "coverage" (tỉ lệ pixel bóng trong patch — soft label tốt nhất)
      3. Y.astype(float32) — fallback cuối cùng (FHL weights bị mất tác dụng)

    Trước đây chỉ có (1) và (3) → Noise.py cũ không lưu Y_soft
    → luôn rơi vào (3) → FHL weights = binary → mất lợi ích soft label.
    """
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.int32)
    stems         = d["stems"] if "stems" in d else np.array([str(i) for i in range(len(Y))])
    feature_names = [str(s) for s in d["feature_names"]]

    # [BUG FIX] Ưu tiên coverage làm soft label
    if "Y_soft" in d:
        Y_soft = d["Y_soft"].astype(np.float32)
        print("  [Y_soft] Loaded from 'Y_soft' key ✓")
    elif "coverage" in d:
        Y_soft = d["coverage"].astype(np.float32)
        print("  [Y_soft] Fallback to 'coverage' key (chạy lại Noise.py để lưu Y_soft đúng cách)")
    else:
        Y_soft = Y.astype(np.float32)
        print("  [Y_soft] WARNING: Fallback to binary Y — FHL weights mất tác dụng!")

    if np.isnan(X).sum() + np.isinf(X).sum() > 0:
        X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)

    X, feature_names = engineer_probe_features(X, feature_names, stems)
    X, feature_names = select_features(X, feature_names, SELECTED_FEATURES)
    print(f"\n[Features] Sử dụng {len(feature_names)} đặc trưng:")
    for i, fn in enumerate(feature_names):
        print(f"  [{i+1:02d}] {fn}")

    n_pos = int(Y.sum())
    ratio = (len(Y) - n_pos) / max(n_pos, 1)
    return X, Y, Y_soft, feature_names, ratio, stems


# ═══════════════════════════════════════════════════════════════════
# MODELS & FHL WEIGHTS
# ═══════════════════════════════════════════════════════════════════

def compute_fhl_weights(Y_soft, Y_binary, alpha=ALPHA_POS, neg_weight=NEG_WEIGHT):
    """
    Fuzzy Hard Label (FHL) weights:
    - Positive patches: weight = coverage^alpha
        → patch có nhiều pixel bóng hơn được ưu tiên cao hơn (alpha=1.5)
        → patch chỉ có 1-2 pixel bóng (boundary noise) được giảm weight
    - Negative patches: weight nhỏ và đều nhau (neg_weight / n_neg)
        → tránh model bị dominated bởi số lượng negative quá nhiều

    Điều kiện hoạt động: Y_soft phải là coverage (0.0→1.0), KHÔNG phải binary.
    → Đây là lý do phải fix bug Y_soft ở load_dataset.
    """
    w       = np.zeros(len(Y_soft), dtype=np.float64)
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
    """
    Random Forest không dùng class_weight="balanced" vì:
    → Nếu dùng FHL sample_weight, class_weight sẽ double-compensate → overfit minority
    → Thay vào đó để FHL weights điều khiển hoàn toàn
    """
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=N_JOBS_RF,
        random_state=42,
        oob_score=True,
    )


def build_stacking(ratio, meta_c=0.5):
    """
    Stacking 3 base learners với meta Logistic Regression:

    HGBT: xử lý tốt imbalanced data qua class_weight="balanced"
          early_stopping tránh overfit, phù hợp dataset nhỏ-vừa

    GBT:  cây quyết định sâu hơn, bắt pattern phức tạp hơn
          subsample=0.8 giảm variance

    RF:   diversity cao (bagging), OOB score dùng để monitor
          class_weight="balanced" vì Stacking không truyền sample_weight xuống base

    Meta LR (passthrough=True): nhận cả xác suất từ 3 model + original features
          → meta learner học được "khi nào tin model nào + feature nào quan trọng"
          C=0.5 (L2 regularization vừa phải, tránh overfit meta level)
    """
    hgbt = HistGradientBoostingClassifier(
        max_iter=500,
        max_leaf_nodes=31,
        learning_rate=0.03,
        min_samples_leaf=15,
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    gbt = GradientBoostingClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        min_samples_leaf=8,
        subsample=0.8,
        random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=N_JOBS_RF,
        random_state=42,
        oob_score=True,
    )
    meta_lr = LogisticRegression(
        C=meta_c, class_weight="balanced", max_iter=2000, random_state=42)

    stacking = StackingClassifier(
        estimators=[("hgbt", hgbt), ("gbt", gbt), ("rf", rf)],
        final_estimator=meta_lr,
        cv=5,
        passthrough=True,   # meta learner nhận thêm raw features
        n_jobs=1,
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", stacking)])


def fit_with_weight(model, X_tr, y_tr, sw):
    """
    Truyền sample_weight xuống model:
    - Stacking: không thể truyền sample_weight xuống individual base estimators
      qua sklearn API → dùng class_weight nội bộ của từng base estimator thay thế
    - RF standalone: truyền trực tiếp FHL sample_weight
    - Pipeline có classifier: truyền qua classifier__sample_weight
    """
    if isinstance(model, Pipeline) and "classifier" in model.named_steps:
        if isinstance(model.named_steps["classifier"], StackingClassifier):
            print("  [Info] Stacking: dùng class_weight nội bộ (không truyền FHL sample_weight).")
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
# METRICS & THRESHOLD SEARCH
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
    """
    Tính IoU và Dice ở cấp độ ảnh (image-level), không phải patch-level.
    Lý do: patch-level F1 bị dominated bởi số lượng, không phản ánh chất lượng
    mask thực tế. Image-level IoU/Dice mới là metric đúng cho segmentation.
    """
    unique = np.unique(np.asarray(stems, str))
    ious, dices = [], []
    for s in unique:
        idx = np.where(np.asarray(stems, str) == s)[0]
        st  = _bin_stats(yt[idx], yp[idx])
        ious.append(st["iou"])
        dices.append(st["dice"])
    return {
        "mean_iou":  float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "n_images":  len(unique),
    }


def find_best_threshold(y_score, y_te, stems, thr_min=0.30, thr_max=0.65):
    """
    Dò threshold tối ưu theo metric kết hợp 0.6*Dice + 0.4*IoU.
    Dice được ưu tiên hơn IoU vì nhạy hơn với false negative
    (bỏ sót bóng cản quan trọng hơn là detect thêm vùng không phải bóng).
    """
    n_steps = int(round((thr_max - thr_min) / 0.01)) + 1
    grid    = np.linspace(thr_min, thr_max, n_steps)

    best_thr, best_score = thr_min, 0.0
    for thr in grid:
        yp       = (y_score >= thr).astype(np.int32)
        im       = image_mask_metrics(y_te, yp, stems)
        combined = 0.6 * im["mean_dice"] + 0.4 * im["mean_iou"]
        if combined > best_score:
            best_score, best_thr = combined, thr

    return best_thr


def cross_validate_and_find_threshold(
    model_builder, X_tv, y_tv, ys_tv, stems_tv, ratio,
    n_splits=5, use_fhl=True, meta_c=0.5, thr_min=0.30, thr_max=0.65
):
    """
    5-Fold CV ở cấp độ ảnh (image-level split):
    - Tránh data leakage: augmented patches của cùng 1 ảnh gốc
      (flip, translate) phải nằm cùng fold
    - Chốt median threshold qua các fold để tránh outlier
    """
    print(f"\n[Validation] {n_splits}-Fold CV | [{thr_min:.2f}, {thr_max:.2f}] bước 0.01 ...")
    splits            = image_level_kfold(stems_tv, n_splits=n_splits)
    thrs, ious, dices = [], [], []

    for k, (tr_idx, va_idx) in enumerate(splits):
        X_tr_, y_tr_, ys_tr_ = X_tv[tr_idx], y_tv[tr_idx], ys_tv[tr_idx]
        X_va_, y_va_, st_va_ = X_tv[va_idx], y_tv[va_idx], stems_tv[va_idx]

        sw = compute_fhl_weights(ys_tr_, y_tr_) if use_fhl else None

        m = (model_builder(ratio) if model_builder.__name__ == "build_rf"
             else model_builder(ratio, meta_c=meta_c))
        fit_with_weight(m, X_tr_, y_tr_, sw)

        y_prob   = m.predict_proba(X_va_)[:, 1]
        best_thr = find_best_threshold(y_prob, y_va_, st_va_, thr_min, thr_max)
        y_pred   = (y_prob >= best_thr).astype(np.int32)
        im       = image_mask_metrics(y_va_, y_pred, st_va_)

        thrs.append(best_thr)
        ious.append(im["mean_iou"])
        dices.append(im["mean_dice"])
        print(f"  Fold {k+1}/{n_splits} | Thr={best_thr:.3f} | "
              f"ImgIoU={im['mean_iou']:.4f} | ImgDice={im['mean_dice']:.4f}")

    median_thr = float(np.median(thrs))
    print(f"  => [CHỐT] Median Thr={median_thr:.4f} | "
          f"CV IoU={np.mean(ious):.4f} | CV Dice={np.mean(dices):.4f}")
    return median_thr


# ═══════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════

def postprocess_mask_per_image(y_pred, y_score, stems,
                                patch_h=16, patch_w=16, min_patch_area=4):
    """
    Morphological closing per image:
    1. Reshape patches về grid 2D
    2. Binary closing (lấp lỗ hổng nhỏ trong mask bóng)
    3. Loại component quá nhỏ (< min_patch_area patches) — noise
    Lý do: bóng cản thật có hình dạng liên tục theo chiều sâu,
    không phải các patch rời rạc ngẫu nhiên.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return y_pred

    y_out        = y_pred.copy()
    unique_stems = np.unique(np.asarray(stems, str))

    for s in unique_stems:
        idx = np.where(np.asarray(stems, str) == s)[0]
        n   = len(idx)
        if n == 0:
            continue

        best_shape = None
        for ratio_wh in [1.25, 1.33, 1.5, 1.6, 2.0]:
            nr = int(round((n / ratio_wh) ** 0.5))
            if nr < 1:
                continue
            nc = int(round(n / nr))
            if nr * nc == n:
                best_shape = (nr, nc)
                break
        if best_shape is None:
            for nr in range(int(n ** 0.5), 0, -1):
                if n % nr == 0:
                    best_shape = (nr, n // nr)
                    break
        if best_shape is None:
            continue

        n_rows, n_cols = best_shape
        grid = y_pred[idx].reshape(n_rows, n_cols)

        struct      = ndimage.generate_binary_structure(2, 1)
        grid_closed = ndimage.binary_closing(grid, structure=struct, iterations=2).astype(np.int32)

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
        clf = (model.named_steps.get("classifier", model)
               if hasattr(model, "named_steps") else model)
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


def evaluate_on_test(name, model, X_te, y_te, stems_te, feature_names,
                     val_threshold, out_dir, apply_postprocess=True):
    print(f"\n{'═'*70}\n  [{name}] — ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP\n{'═'*70}")

    y_sc       = model.predict_proba(X_te)[:, 1]
    y_pred_raw = (y_sc >= val_threshold).astype(int)

    if apply_postprocess:
        y_pred = postprocess_mask_per_image(y_pred_raw, y_sc, stems_te)
        print(f"  [Post-process] Patches thay đổi sau closing: "
              f"{int((y_pred != y_pred_raw).sum())}")
    else:
        y_pred = y_pred_raw

    print(f"  [Config] Threshold = {val_threshold:.4f}")
    print(classification_report(y_te, y_pred,
                                 target_names=["bg(0)", "shadow(1)"], digits=4))

    auc          = roc_auc_score(y_te, y_sc)
    cm           = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    img     = image_mask_metrics(y_te, y_pred, stems_te)
    img_raw = image_mask_metrics(y_te, y_pred_raw, stems_te)
    f1_val  = f1_score(y_te, y_pred)

    print(f"  Raw   (trước PP): ImgIoU={img_raw['mean_iou']:.4f} | "
          f"ImgDice={img_raw['mean_dice']:.4f}")
    print(f"  Final (sau  PP): ImgIoU={img['mean_iou']:.4f}  | "
          f"ImgDice={img['mean_dice']:.4f}  (n={img['n_images']} ảnh)")
    print(f"  Patch F1={f1_val:.4f} | ROC-AUC={auc:.4f}")

    # ─── Feature importance plot với màu theo nhóm ────────────────
    imp_arr, src_name = get_feature_importance(model)
    if imp_arr is not None and feature_names and HAS_MPL:
        if len(imp_arr) == len(feature_names):
            fig, ax = plt.subplots(figsize=(11, 7))
            sidx  = np.argsort(imp_arr)
            names = [feature_names[i] for i in sidx]
            vals  = imp_arr[sidx]

            # Màu theo nhóm chức năng:
            # cam   = column-context mới (above_max_mean, col_dark_ratio, mean_drop)
            # tím   = GLCM texture
            # xanh lá = gate / reflector (absolute_dark, below_bright, col_dark_score)
            # xanh dương = base intensity + position
            colors = []
            for n in names:
                if n in _COL_CONTEXT_FEATURES:
                    colors.append("#f28e2b")   # cam — column-context (NEW)
                elif n in _GLCM_FEATURES:
                    colors.append("#b07aa1")   # tím — GLCM texture
                elif n in _REFLECTOR_FEATURES:
                    colors.append("#59a14f")   # xanh lá — gate/reflector
                elif n in GEO_FEATURE_NAMES:
                    colors.append("#e15759")   # đỏ — geometric (nếu có)
                else:
                    colors.append("#4e79a7")   # xanh dương — base

            ax.barh(names, vals, color=colors)
            ax.set_title(
                f"Feature Importance ({name} / {src_name})\n"
                f"cam=COLUMN-CTX(NEW)  tím=GLCM  xanh lá=GATE/REFLECTOR  xanh=BASE",
                fontsize=9,
            )
            ax.grid(alpha=0.2, axis="x")
            plt.tight_layout()
            path = os.path.join(out_dir,
                                f"{name.lower().replace(' ', '_')}_importance.png")
            plt.savefig(path, dpi=130)
            plt.close()
            print(f"  [Plot] Feature importance → {path}")

    return {
        "model":           name,
        "val_threshold":   val_threshold,
        "accuracy":        (tp + tn) / len(y_te),
        "roc_auc":         round(float(auc), 4),
        "test_iou_image":  round(float(img["mean_iou"]), 4),
        "test_dice_image": round(float(img["mean_dice"]), 4),
        "test_iou_raw":    round(float(img_raw["mean_iou"]), 4),
        "test_dice_raw":   round(float(img_raw["mean_dice"]), 4),
        "f1":              round(float(f1_val), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN RUN
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# MODEL CONFIG — lưu/đọc để Predict_mask.py tự động sync features
# ═══════════════════════════════════════════════════════════════════

def _save_model_config(out_dir: str, feature_names: list, threshold: float, model_type: str):
    """
    Lưu model_config.json sau khi train xong.
    Predict_mask.py đọc file này để biết chính xác:
      - selected_features: danh sách features đúng thứ tự model expect
      - val_threshold    : threshold tối ưu từ CV hoặc manual
      - n_features       : số features model expect (để validate)
      - absolute_dark_threshold, derived_feature_names: để engineer đúng
    """
    config = {
        "selected_features":       list(feature_names),
        "derived_feature_names":   DERIVED_FEATURE_NAMES,
        "absolute_dark_threshold": ABSOLUTE_DARK_THRESHOLD,
        "val_threshold":           round(float(threshold), 4),
        "n_features":              len(feature_names),
        "model_type":              model_type,
        "col_context_features":    ["above_max_mean", "col_dark_ratio", "mean_drop"],
    }
    path = os.path.join(out_dir, "model_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [Saved] model_config.json → {path}")
    print(f"          features={len(feature_names)}: {feature_names}")

def run(dataset_path: str, model_choice: str = "stack", test_size: float = 0.2,
        do_cv: bool = False, use_fhl: bool = True, manual_thr: float = 0.55,
        meta_c: float = 0.5, thr_min: float = 0.30, thr_max: float = 0.65,
        apply_postprocess: bool = True):

    out_dir = str(Path(dataset_path).parent / "models")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, Y_soft, feature_names, ratio, stems = load_dataset(dataset_path)

    train_mask, test_mask, test_bases = image_level_split(stems, test_ratio=test_size)
    X_tr, y_tr           = X[train_mask], Y[train_mask]
    ys_tr, stems_tr      = Y_soft[train_mask], stems[train_mask]
    X_te, y_te, stems_te = X[test_mask], Y[test_mask], stems[test_mask]

    print(f"\n[Data] Train={len(X_tr):,} patches | Test={len(X_te):,} patches")
    print(f"       Shadow (train): {int(y_tr.sum()):,}/{len(y_tr):,} "
          f"({y_tr.mean()*100:.1f}%)")
    print(f"       Y_soft (train): mean={float(ys_tr.mean()):.4f} "
          f"| max={float(ys_tr.max()):.4f}")   # debug: kiểm tra Y_soft đúng hay binary

    with open(os.path.join(out_dir, "test_split_images.json"), "w", encoding="utf-8") as f:
        json.dump({"test_images": test_bases}, f, indent=2)

    all_metrics = []

    if model_choice in ("rf", "both"):
        print(f"\n{'='*55}\n[1] RANDOM FOREST\n{'='*55}")
        val_thr = manual_thr
        if do_cv:
            val_thr = cross_validate_and_find_threshold(
                build_rf, X_tr, y_tr, ys_tr, stems_tr, ratio,
                use_fhl=use_fhl, thr_min=thr_min, thr_max=thr_max,
            )
        else:
            print(f"[Config] CV=TẮT. Threshold cố định = {val_thr:.4f}")

        print(f"\n[Final Train RF] {len(X_tr):,} mẫu ...")
        rf = build_rf(ratio)
        sw = compute_fhl_weights(ys_tr, y_tr) if use_fhl else None
        fit_with_weight(rf, X_tr, y_tr, sw)
        if hasattr(rf, "oob_score_"):
            print(f"  OOB Score = {rf.oob_score_:.4f}")

        m = evaluate_on_test("Random Forest", rf, X_te, y_te, stems_te,
                             feature_names, val_thr, out_dir, apply_postprocess)
        all_metrics.append(m)
        joblib.dump(rf, os.path.join(out_dir, "rf_model.pkl"))
        print(f"  [Saved] rf_model.pkl")
        _save_model_config(out_dir, feature_names, val_thr, "rf")

    if model_choice in ("stack", "both"):
        print(f"\n{'='*55}\n[2] STACKING (HGBT + GBT + RF)\n{'='*55}")
        print(f"[Config] Meta-LR C={meta_c} | passthrough=True")
        val_thr = manual_thr
        if do_cv:
            val_thr = cross_validate_and_find_threshold(
                build_stacking, X_tr, y_tr, ys_tr, stems_tr, ratio,
                use_fhl=use_fhl, meta_c=meta_c, thr_min=thr_min, thr_max=thr_max,
            )
        else:
            print(f"[Config] CV=TẮT. Threshold cố định = {val_thr:.4f}")

        print(f"\n[Final Train Stacking] {len(X_tr):,} mẫu ...")
        t0    = time.time()
        stack = build_stacking(ratio, meta_c=meta_c)
        sw    = compute_fhl_weights(ys_tr, y_tr) if use_fhl else None
        fit_with_weight(stack, X_tr, y_tr, sw)
        print(f"  Hoàn tất ({time.time()-t0:.1f}s)")

        m = evaluate_on_test("Stacking Ensemble", stack, X_te, y_te, stems_te,
                             feature_names, val_thr, out_dir, apply_postprocess)
        all_metrics.append(m)
        joblib.dump(stack, os.path.join(out_dir, "stacking_model.pkl"))
        print(f"  [Saved] stacking_model.pkl")
        _save_model_config(out_dir, feature_names, val_thr, "stacking")

    print(f"\n{'═'*90}")
    print(f"  TỔNG KẾT — BLIND TEST")
    print(f"{'═'*90}")
    header = (f"  {'Model':<22} {'Thr':>6} {'IoU(raw)':>10} "
              f"{'Dice(raw)':>11} {'IoU(PP)':>9} {'Dice(PP)':>10} {'F1':>8}")
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
    parser = argparse.ArgumentParser(description="Train Shadow — Column-Context Features")
    parser.add_argument("--dataset",   default=r"C:\Users\ThinkPad\Graduation_project\Data\bongcan_processed\dataset.npz")
    parser.add_argument("--model",     default="stack", choices=["rf", "stack", "both"])
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--cv",        action="store_true", help="Bật 5-Fold CV")
    parser.add_argument("--thr",       default=0.55, type=float)
    parser.add_argument("--thr_min",   default=0.30, type=float)
    parser.add_argument("--thr_max",   default=0.65, type=float)
    parser.add_argument("--meta_c",    default=0.5,  type=float)
    parser.add_argument("--no_fhl",    action="store_true")
    parser.add_argument("--no_pp",     action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("  TRAIN — SHADOW DETECTION (DATN) — COLUMN-CONTEXT FEATURES")
    print("  Framework: HGBT + GBT + RF (Stacking Ensemble)")
    print(f"  Features : {len(SELECTED_FEATURES)}")
    for f in SELECTED_FEATURES:
        tag = ("[NEW-COL]"   if f in _COL_CONTEXT_FEATURES else
               "[REFLECTOR]" if f in _REFLECTOR_FEATURES   else
               "[GLCM]"      if f in _GLCM_FEATURES        else
               "[BASE]")
        print(f"    {tag:<12} {f}")
    print(f"  CV       : {'BẬT' if args.cv else 'TẮT'} | "
          f"Thr={'[auto]' if args.cv else args.thr}")
    print(f"  FHL      : {'TẮT' if args.no_fhl else 'BẬT (cần Y_soft = coverage)'}")
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