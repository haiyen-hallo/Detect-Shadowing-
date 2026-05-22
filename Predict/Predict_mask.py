import os
import sys
import json
import argparse
import warnings
import traceback
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

# ── Import pipeline từ Predict.py (phải cùng thư mục) ─────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from Predict import (
        predict_panel, get_fan_mask, normalize,
        detect_and_split_dual_panel,
        PATCH_H, PATCH_W,
        THR_FINAL, N_THETA, N_R_SAMPLES,
        TE_LOW_PCT, RF_MIN_PROB, MAX_SHADOW_PCT,
    )
    print("[import] ✅ Predict.py loaded thành công")
except ImportError as e:
    print(f"[import] ❌ Không import được Predict.py: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# ★ ĐỒNG BỘ HOÀN TOÀN VỚI Train.py MỚI (9 features)
# ═══════════════════════════════════════════════════════════════════

# [SYNC] Ngưỡng tối tuyệt đối — phải khớp Train.py
ABSOLUTE_DARK_THRESHOLD = 0.38

# [SYNC] 9 features cốt lõi — phải khớp CHÍNH XÁC SELECTED_FEATURES trong Train.py
SELECTED_FEATURES = [
    # --- Positional ---
    "row_ratio", "col_ratio",
    # --- Core intensity signals ---
    "mean", "contrast",
    # --- Texture GLCM Features ---
    "homogeneity", "energy", "correlation", "entropy",
    # --- Absolute darkness gate ---
    "absolute_dark",
]

# [SYNC] Các đặc trưng phái sinh — phải khớp DERIVED_FEATURE_NAMES trong Train.py
DERIVED_FEATURE_NAMES = [
    "col_dark_score",
    "absolute_dark",
]


def engineer_probe_features_predict(X: np.ndarray, feature_names: list) -> tuple:
    """
    [SYNC] Phiên bản ĐÃ LOẠI BỎ GEOMETRIC.
    Phải giống hệt hàm engineer_probe_features() trong Train.py mới.
    """
    fn  = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6
    def _c(name):
        return X[:, fn[name]].astype(np.float64) if name in fn else np.zeros(len(X), np.float64)

    mean      = _c("mean")
    row_ratio = _c("row_ratio")

    # Adaptive max mean (tính trên toàn panel — giống lúc predict đơn panel)
    adaptive_max_mean = np.clip(np.median(mean) * 0.85, 0.15, 0.50) if len(mean) > 0 else 0.30

    # Tính toán đặc trưng phái sinh (thuần cường độ, không dùng hình học)
    darkness = np.clip((adaptive_max_mean - mean) / (adaptive_max_mean + eps), 0.0, 1.0)
    col_dark = darkness * np.clip(0.4 + 0.6 * row_ratio, 0.4, 1.0)
    
    # Gate chặn speckle sáng
    absolute_dark = np.clip((ABSOLUTE_DARK_THRESHOLD - mean) / (ABSOLUTE_DARK_THRESHOLD + eps), 0.0, 1.0)

    new = np.column_stack([
        col_dark,       
        absolute_dark,  
    ]).astype(np.float32)

    return np.hstack([X, np.nan_to_num(new, nan=0., posinf=1., neginf=-1.)]), feature_names + DERIVED_FEATURE_NAMES


def select_features_predict(X: np.ndarray, feature_names: list, selected: list = None) -> np.ndarray:
    if selected is None:
        selected = SELECTED_FEATURES
    fn_map  = {name: i for i, name in enumerate(feature_names)}
    missing = [s for s in selected if s not in fn_map]
    if missing:
        raise ValueError(f"[FeatureSync] Thiếu features: {missing}. "
                         f"Hãy đảm bảo dataset.npz hoặc luồng Predict.py đã trích xuất đủ các đặc trưng GLCM.")
    idx = [fn_map[s] for s in selected]
    return X[:, idx].astype(X.dtype, copy=False)


# ═══════════════════════════════════════════════════════════════════
# WRAPPER — tự động pipeline features trước khi gọi model
# ═══════════════════════════════════════════════════════════════════

class FeatureAdaptedModelWrapper:
    """
    Bọc model để tự động:
      raw features → engineer → select (9 features) → model.predict_proba
    Đảm bảo khớp 100% với pipeline Train.py mới.
    """
    def __init__(self, raw_model, raw_feature_names):
        self.model             = raw_model
        self.raw_feature_names = list(raw_feature_names)

    def predict_proba(self, X):
        if np.isnan(X).sum() + np.isinf(X).sum() > 0:
            X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)

        X_eng, full_names = engineer_probe_features_predict(X, list(self.raw_feature_names))
        X_sel = select_features_predict(X_eng, full_names, SELECTED_FEATURES)
        return self.model.predict_proba(X_sel)


# ═══════════════════════════════════════════════════════════════════
# FILE UTILS & VISUALIZATION & METRICS
# (Giữ nguyên toàn bộ logic đánh giá, xử lý ảnh của bạn)
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

    gt_only   = (gt_mask == 1)   & (pred_mask == 0)
    pred_only = (pred_mask == 1) & (gt_mask == 0)
    both      = (gt_mask == 1)   & (pred_mask == 1)

    for region, color in [(gt_only, (255, 50, 50)), (pred_only, (30, 30, 220)), (both, (180, 30, 180))]:
        layer = np.zeros_like(canvas)
        layer[region] = color
        canvas = cv2.addWeighted(canvas, 1.0, layer, ALPHA, 0)

    for mask_u8, color in [((gt_mask * 255).astype(np.uint8),   (255, 80, 80)),
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
    sens_m = bo_px / max(gt_px, 1); prec_m = bo_px / max(pr_px, 1)

    txt(legend, stem[:22],               18,  (220, 220, 220), 0.36)
    txt(legend, f"IoU  : {iou*100:.1f}%", 36,  (180, 255, 180), 0.42)
    txt(legend, f"Dice : {dice*100:.1f}%", 54, (180, 255, 180), 0.42)
    txt(legend, f"Sens : {sens_m*100:.1f}%", 72, (180, 255, 180), 0.42)
    txt(legend, f"Prec : {prec_m*100:.1f}%", 90, (180, 255, 180), 0.42)
    box(legend, (255, 80, 80),  118); txt(legend, f"Bac si : {gt_px:,}px",  120, (200,180,255))
    box(legend, (60, 60, 255),  138); txt(legend, f"Model  : {pr_px:,}px",  140, (180,200,255))
    box(legend, (180, 30, 180), 158); txt(legend, f"Trung  : {bo_px:,}px",  160, (255,180,255))
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
        "recall":   round(rec, 4), "f1":        round(f1,   4),
        "specificity": round(spec, 4), "iou": round(iou, 4), "dice": round(dice, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum()),
    }

def compute_overlap(gt, pred):
    inter   = int(((gt == 1) & (pred == 1)).sum())
    union   = int(((gt == 1) | (pred == 1)).sum())
    gt_sz   = int(gt.sum()); pr_sz = int(pred.sum())
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
    thr_final       : float = 0.55,
    n_theta         : int   = N_THETA,
    n_r             : int   = N_R_SAMPLES,
    te_low_pct      : int   = TE_LOW_PCT,
    rf_min          : float = RF_MIN_PROB,
    max_shadow_frac : float = MAX_SHADOW_PCT,
    mask_patch_thr  : float = 0.30,
):
    data_dir    = os.path.abspath(data_dir)
    models_dir  = os.path.join(data_dir, "models")
    out_dir     = os.path.join(data_dir, "eval_results")
    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    # ── 1. Load test split ─────────────────────────────────────────
    split_path = os.path.join(models_dir, "test_split_images.json")
    if not os.path.exists(split_path):
        print(f"[ERROR] Không tìm thấy: {split_path}")
        sys.exit(1)

    with open(split_path, encoding="utf-8") as f:
        test_stems = json.load(f)["test_images"]
    print(f"\n[eval] {len(test_stems)} ảnh test từ test_split_images.json")

    # ── 2. Load model ──────────────────────────────────────────────
    raw_model = None
    for fname in ["stacking_model.pkl", "rf_model.pkl"]:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            raw_model = joblib.load(p)
            print(f"[eval] Model: {fname}")
            break
    if raw_model is None:
        print(f"[ERROR] Không tìm thấy .pkl trong {models_dir}")
        sys.exit(1)

    # ── 3. Load feature names từ dataset.npz ─────────────────────
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
        # Tên chuẩn fallback (chú ý: phải có 4 GLCM features nếu Noise.py đã sinh ra)
        raw_feature_names = [
            "mean", "std", "contrast", "homogeneity", "correlation", "energy", "entropy",
            "row_ratio", "col_ratio", "local_contrast", "below_bright"
        ]
        print(f"[eval] ⚠ Không tìm thấy dataset.npz → dùng fallback raw features")

    expected_n = len(SELECTED_FEATURES)
    print(f"\n[eval] Pipeline: {len(raw_feature_names)} raw → engineer → {expected_n} selected → model")
    
    model = FeatureAdaptedModelWrapper(raw_model, raw_feature_names)

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
        print(f"[ERROR] Không tìm thấy thư mục ảnh trong {data_dir}")
        sys.exit(1)

    has_mask = mask_dir is not None
    print(f"\n[eval] Ảnh : {img_dir}")
    print(f"[eval] Mask: {mask_dir or '(không có → chỉ predict)'}")
    print(f"[eval] Out : {out_dir}")
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

            panels            = detect_and_split_dual_panel(gray)
            pred_pixel_full   = np.zeros((H, W), dtype=np.uint8)
            panel_res_list    = []

            for panel_gray, x_off in panels:
                ph, pw = panel_gray.shape
                norm   = normalize(panel_gray)
                fan, fan_raw = get_fan_mask(norm, thresh=0.05)

                res = predict_panel(
                    norm, fan, fan_raw, model,
                    thr_final, n_theta, n_r,
                    te_low_pct, rf_min, max_shadow_frac,
                )
                panel_pred = pred_grid_to_pixel(res["pred_final"], ph, pw)
                y1 = min(ph, H)
                pred_pixel_full[:y1, x_off:x_off + pw] = panel_pred[:y1, :pw]
                panel_res_list.append(res)
                print(f"    Panel {pw}×{ph}: {int(res['pred_final'].sum())} patches bóng")

            if panel_res_list:
                res0   = panel_res_list[0]
                nr, nc = res0["n_rows"], res0["n_cols"]
                ph0, pw0 = res0["H"], res0["W"]
                gt_patch   = pixel_to_patch_grid(gt_pixel[:ph0, :pw0], nr, nc, ph0, pw0, mask_patch_thr)
                pred_patch = (res0["pred_final"] > 0).astype(np.uint8)
                all_gt_patch.extend(gt_patch.ravel().tolist())
                all_pred_patch.extend(pred_patch.ravel().tolist())

            all_gt_pixel.extend(gt_pixel.ravel().tolist())
            all_pred_pixel.extend(pred_pixel_full.ravel().tolist())

            ovl = compute_overlap(gt_pixel, pred_pixel_full)
            print(f"    IoU={ovl['iou']:.3f} | Dice={ovl['dice']:.3f} | Recall={ovl['sensitivity']:.3f}")

            ov_img = make_overlay_image(gray, gt_pixel, pred_pixel_full, ovl["iou"], stem)
            cv2.imwrite(os.path.join(overlay_dir, f"{stem}_eval.png"), ov_img)

            per_image.append({
                "stem": stem, "img_path": img_path, "mask_path": mask_path,
                "n_pred_patches": int(sum(r["n_final"] for r in panel_res_list)),
                **ovl,
            })
            n_ok += 1

        except Exception as e:
            print(f"    ✗ Lỗi: {e}")
            traceback.print_exc()
            n_fail += 1

    if not all_gt_pixel:
        print("\n[ERROR] Không có dữ liệu để đánh giá.")
        sys.exit(1)

    y_gt_px   = np.array(all_gt_pixel,   dtype=np.uint8)
    y_pred_px = np.array(all_pred_pixel, dtype=np.uint8)
    y_gt_pa   = np.array(all_gt_patch,   dtype=np.uint8)
    y_pred_pa = np.array(all_pred_patch, dtype=np.uint8)

    m_px = compute_metrics(y_gt_px, y_pred_px)
    m_pa = compute_metrics(y_gt_pa, y_pred_pa) if len(all_gt_patch) > 0 else {}

    def _mean(key): return float(np.mean([r[key] for r in per_image]))
    mean_ovl = {k: round(_mean(k), 4) for k in ["iou", "dice", "overlap_coeff", "sensitivity", "precision"]} if per_image else {}

    SEP = "═" * 75
    print(f"\n{SEP}")
    print(f"  KẾT QUẢ — {n_ok} ảnh test | skip={n_skip} | fail={n_fail}")
    print(f"{SEP}")

    def show_metrics(title, m):
        if not m: return
        print(f"\n  📊 {title}:")
        print(f"  {'Metric':<16} {'Value':>8}  Bar")
        print(f"  {'─'*50}")
        for k in ["accuracy", "precision", "recall", "f1", "specificity", "iou", "dice"]:
            v = m.get(k)
            if v is None: continue
            bar = "█" * int(v * 20)
            print(f"  {k:<16} {v:>8.4f}  {bar}")

    show_metrics("Pixel-level", m_px)
    show_metrics("Patch-level", m_pa)

    if mean_ovl:
        print(f"\n  📊 Image-level (mean over {n_ok} ảnh):")
        for k, v in mean_ovl.items():
            bar = "█" * int(v * 20)
            print(f"  {k:<16} {v:>8.4f}  {bar}")

    output = {
        "summary":       {"n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail},
        "pixel_metrics": m_px,
        "patch_metrics": m_pa,
        "mean_overlap":  mean_ovl,
        "per_image":     per_image,
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}")
    print(f"  ✅ Báo cáo → {out_dir}")
    print(f"{SEP}")
    return output


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict_mask — synced với Train.py GLCM (No Geometric)")
    parser.add_argument("--data_dir",       default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed")
    parser.add_argument("--thr_final",      default=0.55,        type=float,
                        help="Threshold predict (đồng bộ với Train.py, mặc định 0.55)")
    parser.add_argument("--n_theta",        default=N_THETA,     type=int)
    parser.add_argument("--n_r",            default=N_R_SAMPLES, type=int)
    parser.add_argument("--te_low_pct",     default=TE_LOW_PCT,  type=int)
    parser.add_argument("--rf_min",         default=RF_MIN_PROB, type=float)
    parser.add_argument("--max_shadow_pct", default=15,          type=int)
    parser.add_argument("--mask_patch_thr", default=0.30,        type=float)
    args = parser.parse_args()

    run_evaluate(
        data_dir        = args.data_dir,
        thr_final       = args.thr_final,
        n_theta         = args.n_theta,
        n_r             = args.n_r,
        te_low_pct      = args.te_low_pct,
        rf_min          = args.rf_min,
        max_shadow_frac = args.max_shadow_pct / 100.0,
        mask_patch_thr  = args.mask_patch_thr,
    )