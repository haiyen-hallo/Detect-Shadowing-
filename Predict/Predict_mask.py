"""
Predict_mask.py — Đồng bộ với Train.py mới (15 features + column-context)

CHANGELOG:
  [SYNC-1]  SELECTED_FEATURES: 9 → 15 features (khớp Train.py mới)
  [SYNC-2]  Thêm below_bright vào SELECTED_FEATURES (đã có trong dataset, bị bỏ sót)
  [SYNC-3]  Thêm 3 column-context features: above_max_mean, col_dark_ratio, mean_drop
  [SYNC-4]  Predict pipeline dùng lại Noise.py.compute_patch_features
  [SYNC-5]  Loại bỏ pipeline hình học/ray cũ để khớp Train.py
  [FIX-PP]  Mặc định dùng raw prediction vì kết quả Train.py raw tốt hơn PP
"""

import os
import sys
import json
import argparse
import warnings
import traceback
import contextlib
import io
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "Processing"))
    from Noise import (
        compute_patch_features,
        normalize,
        PATCH_H, PATCH_W, GLCM_BINS,
        FEATURE_NAMES as RAW_FEATURE_NAMES,
    )
    print("[import] Noise.py feature pipeline loaded thanh cong")
except ImportError as e:
    print(f"[import] Khong import duoc Noise.py: {e}")
    sys.exit(1)

THR_FINAL = 0.55
N_THETA = 256
N_R_SAMPLES = 200
TE_LOW_PCT = 45
RF_MIN_PROB = 0.25
MAX_SHADOW_PCT = 0.15

TUNED_MODEL_DEFAULTS = {
    "rf": {
        "threshold": 0.46,
        "pp_mode": "vertical",
        "pp_close_iter": 1,
        "pp_min_area": 12,
        "pp_min_height": 10,
    },
    "extratrees": {
        "threshold": 0.45,
        "pp_mode": "vertical",
        "pp_close_iter": 2,
        "pp_min_area": 6,
        "pp_min_height": 4,
    },
    "stacking": {
        "threshold": 0.89,
        "pp_mode": "vertical",
        "pp_close_iter": 2,
        "pp_min_area": 12,
        "pp_min_height": 12,
    },
}


# ═══════════════════════════════════════════════════════════════════
# CONFIG — đọc từ model_config.json (được Train.py lưu sau khi train)
# ═══════════════════════════════════════════════════════════════════
# KHÔNG hardcode feature list ở đây nữa.
# Predict_mask.py sẽ load model_config.json từ thư mục models/ để biết:
#   - selected_features       : đúng thứ tự, đúng số lượng model expect
#   - val_threshold           : threshold tối ưu từ lần train
#   - absolute_dark_threshold : ngưỡng gate cứng
#   - derived_feature_names   : features được tính ở predict time
# Điều này đảm bảo Predict luôn khớp với bất kỳ phiên bản Train nào.

# Fallback nếu không tìm thấy model_config.json (model cũ chưa lưu config)
# Các list này sẽ được THAY THẾ bởi giá trị đọc từ file khi load_model_config() chạy
ABSOLUTE_DARK_THRESHOLD = 0.38
DERIVED_FEATURE_NAMES   = ["col_dark_score", "absolute_dark"]
COLUMN_CONTEXT_FEATURES = {"above_max_mean", "col_dark_ratio", "mean_drop"}

# Fallback feature list theo số features model expect
_FALLBACK_FEATURES = {
    # 9 features — phiên bản rất cũ
    9: ["row_ratio","col_ratio","mean","contrast",
        "homogeneity","energy","correlation","entropy","absolute_dark"],
    # 12 features — phiên bản cũ (12 raw Noise.py features, không có derived)
    12: ["contrast","homogeneity","correlation","energy",
         "mean","std","skewness","entropy",
         "row_ratio","col_ratio","local_contrast","below_bright"],
    # 15 features — phiên bản mới (Train.py hiện tại)
    15: ["row_ratio","col_ratio","mean","std",
         "contrast","homogeneity","energy","correlation","entropy",
         "absolute_dark","col_dark_score","below_bright",
         "above_max_mean","col_dark_ratio","mean_drop"],
}

# Sẽ được set bởi load_model_config() — KHÔNG sửa trực tiếp
SELECTED_FEATURES: list = []


def load_model_config(models_dir: str, model, model_key: str = None) -> dict:
    """
    Đọc model_config.json từ thư mục models/ (được Train.py lưu sau train).
    Nếu không có → auto-detect từ n_features_in_ của model + _FALLBACK_FEATURES.

    Returns dict với keys: selected_features, val_threshold,
                            absolute_dark_threshold, derived_feature_names,
                            col_context_features, n_features
    """
    global SELECTED_FEATURES, ABSOLUTE_DARK_THRESHOLD, DERIVED_FEATURE_NAMES

    config_names = []
    if model_key:
        config_names.append(f"{model_key}_model_config.json")
    config_names.append("model_config.json")

    # ── Ưu tiên 1: đọc từ model_config.json ──────────────────────
    for config_name in config_names:
        config_path = os.path.join(models_dir, config_name)
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            SELECTED_FEATURES       = cfg["selected_features"]
            ABSOLUTE_DARK_THRESHOLD = cfg.get("absolute_dark_threshold", 0.38)
            DERIVED_FEATURE_NAMES   = cfg.get("derived_feature_names", ["col_dark_score","absolute_dark"])
            print(f"[config] Loaded {config_name}")
            print(f"         features={cfg['n_features']}: {SELECTED_FEATURES}")
            print(f"         threshold={cfg['val_threshold']}  |  dark_thr={ABSOLUTE_DARK_THRESHOLD}")
            return cfg

    # ── Ưu tiên 2: auto-detect từ n_features_in_ của model ───────
    n_feat = getattr(model, "n_features_in_", None)
    # Với Pipeline/Stacking, lấy từ bước cuối
    if n_feat is None and hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier", None)
        n_feat = getattr(clf, "n_features_in_", None)

    if n_feat is not None and n_feat in _FALLBACK_FEATURES:
        SELECTED_FEATURES = _FALLBACK_FEATURES[n_feat]
        cfg = {
            "selected_features":       SELECTED_FEATURES,
            "derived_feature_names":   DERIVED_FEATURE_NAMES,
            "absolute_dark_threshold": ABSOLUTE_DARK_THRESHOLD,
            "val_threshold":           0.55,
            "n_features":              n_feat,
            "col_context_features":    list(COLUMN_CONTEXT_FEATURES),
        }
        print(f"[config] model_config.json không tìm thấy")
        print(f"         Auto-detect: model expect {n_feat} features")
        print(f"         Dùng fallback: {SELECTED_FEATURES}")
        print(f"         [LƯU Ý] Chạy lại Train.py để lưu model_config.json đúng")
        return cfg

    # ── Fallback cuối: dùng 15 features (phiên bản mới nhất) ─────
    SELECTED_FEATURES = _FALLBACK_FEATURES[15]
    cfg = {
        "selected_features":       SELECTED_FEATURES,
        "derived_feature_names":   DERIVED_FEATURE_NAMES,
        "absolute_dark_threshold": ABSOLUTE_DARK_THRESHOLD,
        "val_threshold":           0.55,
        "n_features":              15,
        "col_context_features":    list(COLUMN_CONTEXT_FEATURES),
    }
    print(f"[config] Không xác định được features từ model. Dùng fallback 15 features.")
    return cfg


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — PHẢI KHỚP HOÀN TOÀN VỚI Train.py
# ═══════════════════════════════════════════════════════════════════

def add_train_derived_features(X: np.ndarray, feature_names: list,
                               stems: np.ndarray = None) -> tuple:
    """
    Tinh 2 derived features dung nhu Train.py: col_dark_score, absolute_dark.
    Day khong phai cac feature hinh hoc/ray cua Predict.py cu.
    """
    fn  = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6

    def _c(name):
        return (X[:, fn[name]].astype(np.float64)
                if name in fn else np.zeros(len(X), np.float64))

    mean      = _c("mean")
    row_ratio = _c("row_ratio")

    adaptive_max = np.zeros_like(mean)
    if stems is not None:
        stems = np.asarray(stems, str)
        for stem in np.unique(stems):
            idx = stems == stem
            adaptive_max[idx] = np.clip(np.median(mean[idx]) * 0.85, 0.15, 0.50)
    else:
        adaptive_max[:] = float(np.clip(np.median(mean) * 0.85, 0.15, 0.50)) if len(mean) > 0 else 0.30

    darkness  = np.clip((adaptive_max - mean) / (adaptive_max + eps), 0., 1.)
    col_dark  = darkness * np.clip(0.4 + 0.6 * row_ratio, 0.4, 1.0)
    abs_dark  = np.clip((ABSOLUTE_DARK_THRESHOLD - mean) / (ABSOLUTE_DARK_THRESHOLD + eps), 0., 1.)

    new = np.column_stack([col_dark, abs_dark]).astype(np.float32)
    return (
        np.hstack([X, np.nan_to_num(new, nan=0., posinf=1., neginf=-1.)]),
        feature_names + DERIVED_FEATURE_NAMES,
    )


def select_features_predict(X: np.ndarray, feature_names: list,
                              selected: list = None) -> np.ndarray:
    if selected is None:
        selected = SELECTED_FEATURES
    fn_map  = {name: i for i, name in enumerate(feature_names)}
    missing = [s for s in selected if s not in fn_map]
    if missing:
        raise ValueError(
            f"[FeatureSync] Thiếu features: {missing}.\n"
            f"  → Hãy chạy lại Noise.py mới để sinh đủ features trong dataset.npz.\n"
            f"  → Hoặc kiểm tra SELECTED_FEATURES trong Train.py vs Predict_mask.py."
        )
    idx = [fn_map[s] for s in selected]
    return X[:, idx].astype(X.dtype, copy=False)


def prepare_features_for_model(X_raw: np.ndarray, raw_feature_names: list,
                               model, stem: str = None) -> np.ndarray:
    names = list(raw_feature_names)
    X = X_raw.astype(np.float32, copy=False)
    if np.isnan(X).sum() + np.isinf(X).sum() > 0:
        X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)

    needs_derived = any(f in DERIVED_FEATURE_NAMES and f not in names
                        for f in SELECTED_FEATURES)
    if needs_derived:
        stems = np.array([stem] * len(X), dtype=object) if stem is not None else None
        X, names = add_train_derived_features(X, names, stems)

    X_sel = select_features_predict(X, names, SELECTED_FEATURES)
    n_model = getattr(model, "n_features_in_", X_sel.shape[1])
    if n_model != X_sel.shape[1]:
        raise ValueError(
            f"[FeatureSync] selected={X_sel.shape[1]} features nhung model expect "
            f"{n_model}. Hay train lai model hoac cap nhat model_config.json."
        )
    return X_sel


def postprocess_patch_grid(pred_grid: np.ndarray, min_patch_area: int = 4,
                           mode: str = "legacy", close_iter: int = None,
                           min_patch_height: int = 1) -> np.ndarray:
    grid = pred_grid.astype(np.uint8)
    if mode == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        iterations = 1 if close_iter is None else close_iter
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        iterations = 2 if close_iter is None else close_iter
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(closed, dtype=np.uint8)
    for idx in range(1, n_lab):
        area = stats[idx, cv2.CC_STAT_AREA]
        height = stats[idx, cv2.CC_STAT_HEIGHT]
        if area >= min_patch_area and height >= min_patch_height:
            cleaned[labels == idx] = 1
    return cleaned


def predict_grid_from_image(norm_img: np.ndarray, model, stem: str,
                            threshold: float, apply_postprocess: bool,
                            pp_mode: str = "legacy",
                            pp_close_iter: int = None,
                            pp_min_area: int = 4,
                            pp_min_height: int = 1) -> dict:
    empty_mask = np.zeros(norm_img.shape, dtype=np.uint8)
    with contextlib.redirect_stdout(io.StringIO()):
        feats = compute_patch_features(norm_img, empty_mask, PATCH_H, PATCH_W, GLCM_BINS)
    X_raw = feats["patch_features"]
    X_sel = prepare_features_for_model(X_raw, RAW_FEATURE_NAMES, model, stem)

    scores = model.predict_proba(X_sel)[:, 1]
    n_rows = norm_img.shape[0] // PATCH_H
    n_cols = norm_img.shape[1] // PATCH_W
    score_grid = scores.reshape(n_rows, n_cols)
    pred_raw = (score_grid >= threshold).astype(np.uint8)
    pred_final = (
        postprocess_patch_grid(
            pred_raw,
            min_patch_area=pp_min_area,
            mode=pp_mode,
            close_iter=pp_close_iter,
            min_patch_height=pp_min_height,
        )
        if apply_postprocess else pred_raw
    )
    return {
        "score_grid": score_grid,
        "pred_raw": pred_raw,
        "pred_final": pred_final,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "H": norm_img.shape[0],
        "W": norm_img.shape[1],
        "n_final": int(pred_final.sum()),
    }


# ═══════════════════════════════════════════════════════════════════
# FILE UTILS & VISUALIZATION & METRICS
# ═══════════════════════════════════════════════════════════════════

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


def find_file(folder: str, stem: str):
    for ext in IMG_EXTS:
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path
    try:
        for fname in os.listdir(folder):
            if Path(fname).stem.lower() == stem.lower():
                return os.path.join(folder, fname)
    except Exception:
        pass
    return None


def load_mask_binary(mask_path: str, H: int, W: int) -> np.ndarray:
    img = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if img.shape != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return (img > 10).astype(np.uint8)


def pred_grid_to_pixel(pred_grid: np.ndarray, H: int, W: int) -> np.ndarray:
    n_rows, n_cols = pred_grid.shape
    pixel = np.zeros((H, W), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            if pred_grid[r, c] > 0:
                y0 = r * PATCH_H; y1 = min(y0 + PATCH_H, H)
                x0 = c * PATCH_W; x1 = min(x0 + PATCH_W, W)
                pixel[y0:y1, x0:x1] = 1
    return pixel


def pixel_to_patch_grid(mask: np.ndarray, n_rows: int, n_cols: int,
                         H: int, W: int, thr: float = 0.30) -> np.ndarray:
    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * PATCH_H; y1 = min(y0 + PATCH_H, H)
            x0 = c * PATCH_W; x1 = min(x0 + PATCH_W, W)
            patch = mask[y0:y1, x0:x1]
            if patch.size > 0 and float(patch.mean()) >= thr:
                grid[r, c] = 1
    return grid


def make_overlay_image(gray, gt_mask, pred_mask, iou, stem):
    H, W   = gray.shape
    base   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = base.copy()
    ALPHA  = 0.55

    gt_only   = (gt_mask == 1) & (pred_mask == 0)
    pred_only = (pred_mask == 1) & (gt_mask == 0)
    both      = (gt_mask == 1) & (pred_mask == 1)

    for region, color in [(gt_only, (255, 50, 50)), (pred_only, (30, 30, 220)), (both, (180, 30, 180))]:
        layer = np.zeros_like(canvas)
        layer[region] = color
        canvas = cv2.addWeighted(canvas, 1.0, layer, ALPHA, 0)

    for mask_u8, color in [((gt_mask * 255).astype(np.uint8), (255, 80, 80)),
                            ((pred_mask * 255).astype(np.uint8), (60, 60, 255))]:
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, color, 2)

    LEG_W  = 210
    legend = np.full((H, LEG_W, 3), 25, dtype=np.uint8)

    def txt(img, s, y, col, sc=0.40, th=1):
        cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    def box(img, col, y):
        cv2.rectangle(img, (8, y - 11), (22, y + 2), col, -1)

    gt_px  = int(gt_mask.sum());   pr_px = int(pred_mask.sum())
    bo_px  = int(both.sum());      un_px = int(((gt_mask == 1) | (pred_mask == 1)).sum())
    dice   = 2 * bo_px / max(gt_px + pr_px, 1)
    sens_m = bo_px / max(gt_px, 1)
    prec_m = bo_px / max(pr_px, 1)

    txt(legend, stem[:22],               18, (220, 220, 220), 0.36)
    txt(legend, f"IoU  : {iou*100:.1f}%", 36, (180, 255, 180), 0.42)
    txt(legend, f"Dice : {dice*100:.1f}%", 54, (180, 255, 180), 0.42)
    txt(legend, f"Sens : {sens_m*100:.1f}%", 72, (180, 255, 180), 0.42)
    txt(legend, f"Prec : {prec_m*100:.1f}%", 90, (180, 255, 180), 0.42)
    box(legend, (255, 80, 80),  118); txt(legend, f"Bac si : {gt_px:,}px",  120, (200, 180, 255))
    box(legend, (60, 60, 255),  138); txt(legend, f"Model  : {pr_px:,}px",  140, (180, 200, 255))
    box(legend, (180, 30, 180), 158); txt(legend, f"Trung  : {bo_px:,}px",  160, (255, 180, 255))
    txt(legend, f"Union  : {un_px:,}px", 178, (160, 160, 160))

    if H > 220:
        txt(legend, "DO/XANH = Bac si",   H - 56, (255, 120, 120), 0.38)
        txt(legend, "XANH DUONG = Model", H - 40, (120, 120, 255), 0.38)
        txt(legend, "TIM  = Trung nhau",  H - 24, (255, 120, 255), 0.38)

    sep = np.full((H, 4, 3), 60, dtype=np.uint8)
    return np.hstack([base, sep, canvas, sep, legend])


def compute_metrics(y_true, y_pred):
    acc  = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec  = tn / max(tn + fp, 1)
    inter = int(((y_pred == 1) & (y_true == 1)).sum())
    union = int(((y_pred == 1) | (y_true == 1)).sum())
    iou   = inter / max(union, 1)
    dice  = 2 * inter / max(int(y_pred.sum()) + int(y_true.sum()), 1)
    return {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall":   round(rec, 4), "f1":        round(f1, 4),
        "specificity": round(spec, 4), "iou": round(iou, 4), "dice": round(dice, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum()),
    }


def compute_overlap(gt, pred):
    inter   = int(((gt == 1) & (pred == 1)).sum())
    union   = int(((gt == 1) | (pred == 1)).sum())
    gt_sz   = int(gt.sum());   pr_sz = int(pred.sum())
    iou     = inter / max(union, 1)
    dice    = 2 * inter / max(gt_sz + pr_sz, 1)
    overlap = inter / max(min(gt_sz, pr_sz), 1)
    sens    = inter / max(gt_sz, 1)
    ppc     = inter / max(pr_sz, 1)
    return {
        "iou": round(iou, 4), "dice": round(dice, 4),
        "overlap_coeff": round(overlap, 4), "sensitivity": round(sens, 4),
        "precision": round(ppc, 4), "gt_pixels": gt_sz, "pred_pixels": pr_sz,
        "inter_pixels": inter, "union_pixels": union,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_evaluate(
    data_dir        : str,
    out_dir         : str   = None,
    thr_final       : float = None,
    n_theta         : int   = N_THETA,
    n_r             : int   = N_R_SAMPLES,
    te_low_pct      : int   = TE_LOW_PCT,
    rf_min          : float = RF_MIN_PROB,
    max_shadow_frac : float = MAX_SHADOW_PCT,
    mask_patch_thr  : float = 0.30,
    model_name      : str   = "auto",
    tuned_defaults  : bool  = False,
    pp_mode         : str   = None,
    pp_close_iter   : int   = None,
    pp_min_area     : int   = None,
    pp_min_height   : int   = None,
    save_outputs    : bool  = True,
    apply_postprocess: bool = True,
):
    data_dir    = os.path.abspath(data_dir)
    models_dir  = os.path.join(data_dir, "models")
    out_dir     = os.path.abspath(out_dir) if out_dir else os.path.join(data_dir, "eval_results")
    overlay_dir = os.path.join(out_dir, "overlays")
    if save_outputs:
        os.makedirs(overlay_dir, exist_ok=True)

    # ── 1. Load test split ─────────────────────────────────────────
    split_path = os.path.join(models_dir, "test_split_images.json")
    if not os.path.exists(split_path):
        print(f"[ERROR] Không tìm thấy: {split_path}"); sys.exit(1)

    with open(split_path, encoding="utf-8") as f:
        test_stems = json.load(f)["test_images"]
    print(f"\n[eval] {len(test_stems)} ảnh test từ test_split_images.json")

    # ── 2. Load model ──────────────────────────────────────────────
    model_key = (model_name or "auto").lower()
    model_candidates = {
        "auto":          ["extratrees_model.pkl", "stacking_model.pkl", "rf_model.pkl"],
        "extratrees":    ["extratrees_model.pkl"],
        "extra":         ["extratrees_model.pkl"],
        "et":            ["extratrees_model.pkl"],
        "stacking":      ["stacking_model.pkl"],
        "stack":         ["stacking_model.pkl"],
        "rf":            ["rf_model.pkl"],
        "random_forest": ["rf_model.pkl"],
        "random-forest": ["rf_model.pkl"],
    }
    canonical_models = {
        "auto": "auto",
        "extratrees": "extratrees",
        "extra": "extratrees",
        "et": "extratrees",
        "stacking": "stacking",
        "stack": "stacking",
        "rf": "rf",
        "random_forest": "rf",
        "random-forest": "rf",
    }
    canonical_model = canonical_models.get(model_key, model_key)
    wanted_files = model_candidates.get(model_key)
    if wanted_files is None:
        print(f"[ERROR] --model khong hop le: {model_name}"); sys.exit(1)

    raw_model = None
    loaded_model_key = canonical_model
    for fname in wanted_files:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            raw_model = joblib.load(p)
            if fname.startswith("extratrees"):
                loaded_model_key = "extratrees"
            elif fname.startswith("stacking"):
                loaded_model_key = "stacking"
            elif fname.startswith("rf"):
                loaded_model_key = "rf"
            print(f"[eval] Model: {fname} (--model {model_key})")
            break
    if raw_model is None:
        print(f"[ERROR] Khong tim thay model {', '.join(wanted_files)} trong {models_dir}"); sys.exit(1)

    # ── 3. Load feature names từ dataset.npz ──────────────────────
    npz_candidates = [
        os.path.join(data_dir, "dataset.npz"),
        os.path.join(os.path.dirname(data_dir), "dataset.npz"),
    ]
    raw_feature_names = None
    for npz_path in npz_candidates:
        if os.path.exists(npz_path):
            d = np.load(npz_path, allow_pickle=True)
            raw_feature_names = [str(s) for s in d["feature_names"]]
            print(f"[eval] Loaded {len(raw_feature_names)} raw features từ: {npz_path}")
            break

    if raw_feature_names is None:
        # [SYNC-5] Fallback cập nhật 15 raw features từ Noise.py mới
        raw_feature_names = [
            "contrast", "homogeneity", "correlation", "energy",
            "mean", "std", "skewness", "entropy",
            "row_ratio", "col_ratio", "local_contrast", "below_bright",
            "above_max_mean", "col_dark_ratio", "mean_drop",
        ]
        print(f"[eval] Fallback raw features ({len(raw_feature_names)} features)")

    # ── [KEY] Load model_config.json để biết chính xác features cần dùng ──
    # Thứ tự bắt buộc: load model → load config → tạo wrapper
    # model_config.json được Train.py lưu sau mỗi lần train
    cfg = load_model_config(models_dir, raw_model, loaded_model_key)
    val_thr_from_train = float(cfg.get("val_threshold", 0.55))
    tuned = TUNED_MODEL_DEFAULTS.get(loaded_model_key, {})
    if tuned_defaults and tuned:
        if thr_final is None:
            thr_final = tuned["threshold"]
        if pp_mode is None:
            pp_mode = tuned["pp_mode"]
        if pp_close_iter is None:
            pp_close_iter = tuned["pp_close_iter"]
        if pp_min_area is None:
            pp_min_area = tuned["pp_min_area"]
        if pp_min_height is None:
            pp_min_height = tuned["pp_min_height"]
    elif thr_final is None:
        thr_final = val_thr_from_train

    pp_mode = pp_mode or "legacy"
    pp_min_area = 4 if pp_min_area is None else int(pp_min_area)
    pp_min_height = 1 if pp_min_height is None else int(pp_min_height)

    print(f"\n[eval] Pipeline: Noise.py features ({len(RAW_FEATURE_NAMES)}) -> Train-derived -> model")
    print(f"       dataset schema co {len(raw_feature_names)} features")
    print(f"       old geometric/ray pipeline: OFF")
    print(f"       → select {len(SELECTED_FEATURES)} → model")
    print(f"  SELECTED ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")
    print(f"  Threshold dang dung: {thr_final} (Train saved: {val_thr_from_train})")
    print(f"  PP config: mode={pp_mode}, close_iter={pp_close_iter}, min_area={pp_min_area}, min_height={pp_min_height}")

    # ── 4. Tìm thư mục ảnh & mask ─────────────────────────────────
    img_dir = next(
        (os.path.join(data_dir, d) for d in ["images_gray", "images", "imgs", "bongcan"]
         if os.path.isdir(os.path.join(data_dir, d))), None
    )
    mask_dir = next(
        (os.path.join(data_dir, d) for d in ["masks", "mask", "labels", "annotations", "ground_truth"]
         if os.path.isdir(os.path.join(data_dir, d))), None
    )

    if img_dir is None:
        print(f"[ERROR] Không tìm thấy thư mục ảnh trong {data_dir}"); sys.exit(1)

    has_mask = mask_dir is not None
    print(f"\n[eval] Ảnh : {img_dir}")
    print(f"[eval] Mask: {mask_dir or '(không có → chỉ predict)'}")
    print(f"[eval] PP  : {'BẬT' if apply_postprocess else 'TẮT'}")
    print(f"{'─'*70}")

    # ── 5. Vòng lặp đánh giá ──────────────────────────────────────
    all_gt_patch = []; all_pred_patch = []
    all_gt_pixel = []; all_pred_pixel = []
    per_image    = []
    n_ok = n_skip = n_fail = 0

    for stem in test_stems:
        img_path  = find_file(img_dir, stem)
        if img_path is None:
            n_skip += 1; continue

        mask_path = find_file(mask_dir, stem) if has_mask else None
        if has_mask and mask_path is None:
            n_skip += 1; continue

        print(f"\n  → {stem}")
        try:
            img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
            gray   = cv2.medianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), ksize=3)
            H, W   = gray.shape

            gt_pixel = load_mask_binary(mask_path, H, W) if mask_path else np.zeros((H, W), dtype=np.uint8)

            norm = normalize(gray)
            res = predict_grid_from_image(
                norm, raw_model, stem, thr_final, apply_postprocess,
                pp_mode=pp_mode,
                pp_close_iter=pp_close_iter,
                pp_min_area=pp_min_area,
                pp_min_height=pp_min_height)
            pred_grid = (res["pred_final"] > 0).astype(np.uint8)
            pred_pixel_full = pred_grid_to_pixel(pred_grid, H, W)

            nr, nc = res["n_rows"], res["n_cols"]
            gt_patch = pixel_to_patch_grid(gt_pixel, nr, nc, H, W, mask_patch_thr)
            all_gt_patch.extend(gt_patch.ravel().tolist())
            all_pred_patch.extend(pred_grid.ravel().tolist())

            eval_mode = "pp" if apply_postprocess else "raw"
            print(f"    Grid {nr}×{nc}: {int(pred_grid.sum())} patches bóng ({eval_mode})")

            all_gt_pixel.extend(gt_pixel.ravel().tolist())
            all_pred_pixel.extend(pred_pixel_full.ravel().tolist())

            ovl = compute_overlap(gt_pixel, pred_pixel_full)
            print(f"    IoU={ovl['iou']:.3f} | Dice={ovl['dice']:.3f} | "
                  f"Recall={ovl['sensitivity']:.3f}")

            if save_outputs:
                ov_img = make_overlay_image(gray, gt_pixel, pred_pixel_full, ovl["iou"], stem)
                cv2.imwrite(os.path.join(overlay_dir, f"{stem}_eval.png"), ov_img)

            per_image.append({
                "stem": stem, "img_path": img_path, "mask_path": mask_path,
                "n_pred_patches": int(pred_grid.sum()),
                **ovl,
            })
            n_ok += 1

        except Exception as e:
            print(f"    Lỗi: {e}")
            traceback.print_exc()
            n_fail += 1

    if not all_gt_pixel:
        print("\n[ERROR] Không có dữ liệu để đánh giá."); sys.exit(1)

    y_gt_px   = np.array(all_gt_pixel,   dtype=np.uint8)
    y_pred_px = np.array(all_pred_pixel, dtype=np.uint8)
    y_gt_pa   = np.array(all_gt_patch,   dtype=np.uint8)
    y_pred_pa = np.array(all_pred_patch, dtype=np.uint8)

    m_px = compute_metrics(y_gt_px, y_pred_px)
    m_pa = compute_metrics(y_gt_pa, y_pred_pa) if len(all_gt_patch) > 0 else {}

    def _mean(key): return float(np.mean([r[key] for r in per_image]))
    mean_ovl = {k: round(_mean(k), 4)
                for k in ["iou", "dice", "overlap_coeff", "sensitivity", "precision"]
                } if per_image else {}

    SEP = "═" * 75
    print(f"\n{SEP}")
    print(f"  KẾT QUẢ — {n_ok} ảnh test | skip={n_skip} | fail={n_fail}")
    print(f"{SEP}")

    def show_metrics(title, m):
        if not m: return
        print(f"\n  {title}:")
        print(f"  {'Metric':<16} {'Value':>8}  Bar")
        print(f"  {'─'*50}")
        for k in ["accuracy", "precision", "recall", "f1", "specificity", "iou", "dice"]:
            v = m.get(k)
            if v is None: continue
            bar = "█" * int(v * 20)
            print(f"  {k:<16} {v:>8.4f}  {bar}")

    if mean_ovl:
        print(f"\n  Doctor-label Image-level (PRIMARY, mean over {n_ok} anh):")
        for k, v in mean_ovl.items():
            bar = "#" * int(v * 20)
            print(f"  {k:<16} {v:>8.4f}  {bar}")

    show_metrics("Pixel-level (global, secondary)", m_px)
    show_metrics("Patch-level (secondary)", m_pa)

    output = {
        "summary":       {"n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail},
        "pixel_metrics": m_px,
        "patch_metrics": m_pa,
        "mean_overlap":  mean_ovl,
        "per_image":     per_image,
    }
    if save_outputs:
        with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    if save_outputs:
        print(f"\n{SEP}")
        print(f"  Báo cáo → {out_dir}")
        print(f"{SEP}")
    return output


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict_mask — synced với Train.py 15-feature")
    parser.add_argument("--data_dir",       default=r"C:\Users\ThinkPad\Graduation_project\Data\bongcan_processed")
    parser.add_argument("--out_dir",        default=None,
                        help="Thu muc luu eval_results; mac dinh la data_dir/eval_results")
    parser.add_argument("--no_save",        action="store_true",
                        help="Chi in metrics, khong ghi overlay/json")
    parser.add_argument("--thr_final",      default=None,        type=float,
                        help="Threshold predict; mặc định dùng val_threshold trong model_config.json")
    parser.add_argument("--model",          default="auto",
                        choices=["auto", "extratrees", "extra", "et",
                                 "stacking", "stack", "rf", "random_forest", "random-forest"],
                        help="Chon model .pkl: auto uu tien extratrees, stacking, roi rf")
    parser.add_argument("--tuned",          action="store_true",
                        help="Dung threshold va vertical post-process da benchmark cho tung model")
    parser.add_argument("--pp_mode",        default=None, choices=["legacy", "vertical"],
                        help="Kieu post-process khi bat --pp/--tuned")
    parser.add_argument("--pp_close_iter",  default=None, type=int)
    parser.add_argument("--pp_min_area",    default=None, type=int)
    parser.add_argument("--pp_min_height",  default=None, type=int)
    parser.add_argument("--n_theta",        default=N_THETA,     type=int)
    parser.add_argument("--n_r",            default=N_R_SAMPLES, type=int)
    parser.add_argument("--te_low_pct",     default=TE_LOW_PCT,  type=int)
    parser.add_argument("--rf_min",         default=RF_MIN_PROB, type=float)
    parser.add_argument("--max_shadow_pct", default=15,          type=int)
    parser.add_argument("--mask_patch_thr", default=0.30,        type=float)
    parser.add_argument("--pp",             action="store_true",
                        help="Bật post-processing dạng grid giống bước evaluate trong Train.py")
    parser.add_argument("--no_pp",          action="store_true",
                        help="Giữ tương thích lệnh cũ; mặc định hiện đã tắt PP để lấy raw tốt nhất")
    args = parser.parse_args()
    apply_pp = (args.pp or args.tuned) and not args.no_pp

    print("=" * 80)
    print("  PREDICT_MASK — SHADOW DETECTION (DATN)")
    print("  Features    : load from model_config.json")
    print(f"  Model       : {args.model}")
    print(f"  Threshold   : {args.thr_final if args.thr_final is not None else ('[tuned]' if args.tuned else '[model_config]')}")
    print(f"  Post-process: {'BẬT' if apply_pp else 'TẮT'}")
    print("  Pipeline    : Noise.py feature extraction, no old geometric/ray pipeline")
    print("=" * 80 + "\n")

    run_evaluate(
        data_dir         = args.data_dir,
        out_dir          = args.out_dir,
        thr_final        = args.thr_final,
        n_theta          = args.n_theta,
        n_r              = args.n_r,
        te_low_pct       = args.te_low_pct,
        rf_min           = args.rf_min,
        max_shadow_frac  = args.max_shadow_pct / 100.0,
        mask_patch_thr   = args.mask_patch_thr,
        model_name       = args.model,
        tuned_defaults   = args.tuned,
        pp_mode          = args.pp_mode,
        pp_close_iter    = args.pp_close_iter,
        pp_min_area      = args.pp_min_area,
        pp_min_height    = args.pp_min_height,
        save_outputs     = not args.no_save,
        apply_postprocess= apply_pp,
    )
