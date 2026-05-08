

import os, sys, json, argparse, warnings, traceback
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, confusion_matrix, average_precision_score,
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
    print("         Đảm bảo Evaluate.py cùng thư mục với Predict.py")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# FILE UTILS
# ═══════════════════════════════════════════════════════════════════

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp",
            ".JPG", ".JPEG", ".PNG", ".BMP"]


def find_file(folder: str, stem: str):
    """Tìm file theo stem trong folder, thử tất cả extension ảnh."""
    for ext in IMG_EXTS:
        path = os.path.join(folder, stem + ext)
        if os.path.exists(path):
            return path
    # Tìm kiếm không phân biệt hoa thường
    try:
        for fname in os.listdir(folder):
            if Path(fname).stem.lower() == stem.lower():
                return os.path.join(folder, fname)
    except Exception:
        pass
    return None


def load_mask_binary(mask_path: str, H: int, W: int) -> np.ndarray:
    """
    Load mask bác sĩ → binary ndarray (H, W) uint8 {0, 1}.
    Resize về (H, W) nếu cần.
    """
    img = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    if img.shape != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return (img > 10).astype(np.uint8)


def pred_grid_to_pixel(pred_grid: np.ndarray, H: int, W: int) -> np.ndarray:
    """Chuyển pred_final (n_rows x n_cols) → pixel mask (H x W)."""
    n_rows, n_cols = pred_grid.shape
    pixel = np.zeros((H, W), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            if pred_grid[r, c] > 0:
                y0 = r * PATCH_H;  y1 = min(y0 + PATCH_H, H)
                x0 = c * PATCH_W;  x1 = min(x0 + PATCH_W, W)
                pixel[y0:y1, x0:x1] = 1
    return pixel


def pixel_to_patch_grid(mask: np.ndarray, n_rows: int, n_cols: int,
                         H: int, W: int, thr: float = 0.30) -> np.ndarray:
    """
    Chuyển pixel mask → patch grid (n_rows x n_cols).
    Patch = 1 nếu tỉ lệ pixel shadow trong patch >= thr.
    """
    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * PATCH_H;  y1 = min(y0 + PATCH_H, H)
            x0 = c * PATCH_W;  x1 = min(x0 + PATCH_W, W)
            patch = mask[y0:y1, x0:x1]
            if patch.size > 0 and float(patch.mean()) >= thr:
                grid[r, c] = 1
    return grid


# ═══════════════════════════════════════════════════════════════════
# OVERLAY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def make_overlay_image(gray: np.ndarray,
                       gt_mask: np.ndarray,
                       pred_mask: np.ndarray,
                       iou: float,
                       stem: str) -> np.ndarray:
    """
    Tạo ảnh overlay với 3 lớp màu:
      XANH (Blue)    = bác sĩ khoanh  (GT only)
      ĐỎ   (Red)     = model detect   (Pred only)
      TÍM  (Magenta) = trùng nhau     (GT ∩ Pred)

    Layout: [ảnh gốc | ảnh overlay | legend]
    """
    H, W  = gray.shape
    base  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    canvas = base.copy()

    gt_only   = (gt_mask   == 1) & (pred_mask == 0)
    pred_only = (pred_mask == 1) & (gt_mask   == 0)
    both      = (gt_mask   == 1) & (pred_mask == 1)

    ALPHA = 0.55

    # XANH dương = bác sĩ
    layer = np.zeros_like(canvas)
    layer[gt_only] = (255, 50, 50)         # BGR
    canvas = cv2.addWeighted(canvas, 1.0, layer, ALPHA, 0)

    # ĐỎ = model
    layer = np.zeros_like(canvas)
    layer[pred_only] = (30, 30, 220)       # BGR
    canvas = cv2.addWeighted(canvas, 1.0, layer, ALPHA, 0)

    # TÍM/MAGENTA = overlap
    layer = np.zeros_like(canvas)
    layer[both] = (180, 30, 180)           # BGR
    canvas = cv2.addWeighted(canvas, 1.0, layer, ALPHA, 0)

    # Contour bác sĩ — xanh dương đậm
    gt_u8 = (gt_mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, (255, 80, 80), 2)

    # Contour model — đỏ đậm
    pr_u8 = (pred_mask * 255).astype(np.uint8)
    cnts, _ = cv2.findContours(pr_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, (60, 60, 255), 2)

    # ── Legend ─────────────────────────────────────────────────────
    LEG_W = 210
    legend = np.full((H, LEG_W, 3), 25, dtype=np.uint8)

    def txt(img, s, y, col, sc=0.40, th=1):
        cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, th, cv2.LINE_AA)

    def box(img, col, y):
        cv2.rectangle(img, (8, y - 11), (22, y + 2), col, -1)
        cv2.rectangle(img, (8, y - 11), (22, y + 2), (120,120,120), 1)

    gt_px   = int(gt_mask.sum())
    pr_px   = int(pred_mask.sum())
    bo_px   = int(both.sum())
    un_px   = int(((gt_mask == 1) | (pred_mask == 1)).sum())
    dice    = 2 * bo_px / max(gt_px + pr_px, 1)
    sens_m  = bo_px / max(gt_px, 1)
    prec_m  = bo_px / max(pr_px, 1)

    txt(legend, stem[:22],         18, (220, 220, 220), 0.36)
    txt(legend, f"IoU  : {iou*100:.1f}%",   36, (180, 255, 180), 0.42)
    txt(legend, f"Dice : {dice*100:.1f}%",  54, (180, 255, 180), 0.42)
    txt(legend, f"Sens : {sens_m*100:.1f}%", 72, (180, 255, 180), 0.42)
    txt(legend, f"Prec : {prec_m*100:.1f}%", 90, (180, 255, 180), 0.42)

    box(legend, (255, 80,  80),  118); txt(legend, f"Bac si : {gt_px:,}px",  120, (200,180,255))
    box(legend, (60,  60, 255),  138); txt(legend, f"Model  : {pr_px:,}px",  140, (180,200,255))
    box(legend, (180, 30, 180),  158); txt(legend, f"Trung  : {bo_px:,}px",  160, (255,180,255))
    txt(legend, f"Union  : {un_px:,}px", 178, (160, 160, 160))

    if H > 220:
        txt(legend, "XANH = Bac si",    H - 56, (255, 120, 120), 0.38)
        txt(legend, "DO   = Model",      H - 40, (120, 120, 255), 0.38)
        txt(legend, "TIM  = Trung nhau", H - 24, (255, 120, 255), 0.38)

    sep = np.full((H, 4, 3), 60, dtype=np.uint8)
    return np.hstack([base, sep, canvas, sep, legend])


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray = None) -> dict:
    acc  = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    spec = tn / max(tn + fp, 1)

    inter = int(((y_pred == 1) & (y_true == 1)).sum())
    union = int(((y_pred == 1) | (y_true == 1)).sum())
    iou   = inter / max(union, 1)
    dice  = 2 * inter / max(int(y_pred.sum()) + int(y_true.sum()), 1)

    auc = ap = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_score))
            ap  = float(average_precision_score(y_true, y_score))
        except Exception:
            pass

    return {
        "accuracy"   : round(acc,  4),
        "precision"  : round(prec, 4),
        "recall"     : round(rec,  4),
        "f1"         : round(f1,   4),
        "specificity": round(spec, 4),
        "iou"        : round(iou,  4),
        "dice"       : round(dice, 4),
        "auc"        : round(auc,  4) if auc is not None else None,
        "avg_prec"   : round(ap,   4) if ap  is not None else None,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_pos": int(y_true.sum()), "n_neg": int((y_true == 0).sum()),
    }


def compute_overlap(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Tính overlap metrics giữa 2 pixel mask."""
    inter   = int(((gt == 1) & (pred == 1)).sum())
    union   = int(((gt == 1) | (pred == 1)).sum())
    gt_sz   = int(gt.sum())
    pr_sz   = int(pred.sum())
    iou     = inter / max(union, 1)
    dice    = 2 * inter / max(gt_sz + pr_sz, 1)
    overlap = inter / max(min(gt_sz, pr_sz), 1)   # Overlap coefficient
    sens    = inter / max(gt_sz, 1)                # recall mask-level
    ppc     = inter / max(pr_sz, 1)               # precision mask-level
    return {
        "iou"           : round(iou,     4),
        "dice"          : round(dice,    4),
        "overlap_coeff" : round(overlap, 4),
        "sensitivity"   : round(sens,    4),
        "precision"     : round(ppc,     4),
        "gt_pixels"     : gt_sz,
        "pred_pixels"   : pr_sz,
        "inter_pixels"  : inter,
        "union_pixels"  : union,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run_evaluate(
    data_dir        : str,
    thr_final       : float = THR_FINAL,
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
        print("        Hãy chạy Train.py trước để sinh file này.")
        sys.exit(1)

    with open(split_path, encoding="utf-8") as f:
        test_stems = json.load(f)["test_images"]
    print(f"\n[eval] {len(test_stems)} ảnh test từ test_split_images.json")

    # ── 2. Load model ──────────────────────────────────────────────
    model = None
    for fname in ["stacking_model.pkl", "rf_model.pkl"]:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            model = joblib.load(p)
            print(f"[eval] Model: {fname}")
            break
    if model is None:
        print(f"[ERROR] Không tìm thấy model trong {models_dir}")
        sys.exit(1)

    # ── 3. Tìm thư mục ảnh & mask ─────────────────────────────────
    img_dir = next(
        (os.path.join(data_dir, d) for d in
         ["images_gray", "images", "imgs", "bongcan"]
         if os.path.isdir(os.path.join(data_dir, d))),
        None,
    )
    mask_dir = next(
        (os.path.join(data_dir, d) for d in
         ["masks", "mask", "labels", "annotations", "ground_truth"]
         if os.path.isdir(os.path.join(data_dir, d))),
        None,
    )

    if img_dir is None:
        print(f"[ERROR] Không tìm thấy thư mục ảnh trong {data_dir}")
        sys.exit(1)

    has_mask = mask_dir is not None
    print(f"[eval] Ảnh  : {img_dir}")
    print(f"[eval] Mask : {mask_dir or '(không có → chỉ chạy predict)'}")
    print(f"[eval] Out  : {out_dir}")
    print(f"{'─'*70}")

    # ── 4. Vòng lặp đánh giá ──────────────────────────────────────
    all_gt_patch   = []
    all_pred_patch = []
    all_gt_pixel   = []
    all_pred_pixel = []
    per_image      = []
    n_ok = n_skip = n_fail = 0

    for stem in test_stems:
        img_path = find_file(img_dir, stem)
        if img_path is None:
            print(f"  ⚠ Không thấy ảnh: {stem}")
            n_skip += 1
            continue

        mask_path = find_file(mask_dir, stem) if has_mask else None
        if has_mask and mask_path is None:
            print(f"  ⚠ Không thấy mask: {stem}")
            n_skip += 1
            continue

        print(f"\n  → {stem}")
        try:
            # Load ảnh
            img_np = np.array(
                Image.open(img_path).convert("RGB"), dtype=np.uint8
            )
            gray = cv2.medianBlur(
                cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), ksize=3
            )
            H, W = gray.shape

            # Load mask bác sĩ
            gt_pixel = np.zeros((H, W), dtype=np.uint8)
            if mask_path:
                gt_pixel = load_mask_binary(mask_path, H, W)

            # Predict (xử lý dual panel)
            panels = detect_and_split_dual_panel(gray)
            pred_pixel_full = np.zeros((H, W), dtype=np.uint8)
            panel_res_list  = []

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
                print(f"    Panel {pw}×{ph}: {int(res['pred_final'].sum())} patches")

            # Patch-level (panel đầu tiên)
            if panel_res_list:
                res0   = panel_res_list[0]
                nr, nc = res0["n_rows"], res0["n_cols"]
                ph0, pw0 = res0["H"], res0["W"]

                gt_patch   = pixel_to_patch_grid(
                    gt_pixel[:ph0, :pw0], nr, nc, ph0, pw0, mask_patch_thr
                )
                pred_patch = (res0["pred_final"] > 0).astype(np.uint8)

                all_gt_patch.extend(gt_patch.ravel().tolist())
                all_pred_patch.extend(pred_patch.ravel().tolist())

            # Pixel-level
            all_gt_pixel.extend(gt_pixel.ravel().tolist())
            all_pred_pixel.extend(pred_pixel_full.ravel().tolist())

            # Overlap per image
            ovl = compute_overlap(gt_pixel, pred_pixel_full)
            print(f"    IoU={ovl['iou']:.3f}  Dice={ovl['dice']:.3f}  "
                  f"Recall={ovl['sensitivity']:.3f}  Prec={ovl['precision']:.3f}")

            # Overlay image
            ov_img = make_overlay_image(
                gray, gt_pixel, pred_pixel_full, ovl["iou"], stem
            )
            cv2.imwrite(
                os.path.join(overlay_dir, f"{stem}_eval.png"), ov_img
            )

            per_image.append({
                "stem"         : stem,
                "img_path"     : img_path,
                "mask_path"    : mask_path,
                "n_pred_patches": int(
                    sum(r["n_final"] for r in panel_res_list)
                ),
                **ovl,
            })
            n_ok += 1

        except Exception as e:
            print(f"    ✗ {e}")
            traceback.print_exc()
            n_fail += 1

    if not all_gt_pixel:
        print("\n[ERROR] Không có dữ liệu để tính metrics.")
        sys.exit(1)

    # ── 5. Aggregate metrics ───────────────────────────────────────
    y_gt_px   = np.array(all_gt_pixel,   dtype=np.uint8)
    y_pred_px = np.array(all_pred_pixel, dtype=np.uint8)
    y_gt_pa   = np.array(all_gt_patch,   dtype=np.uint8)
    y_pred_pa = np.array(all_pred_patch, dtype=np.uint8)

    m_px = compute_metrics(y_gt_px,   y_pred_px)
    m_pa = compute_metrics(y_gt_pa,   y_pred_pa) if len(all_gt_patch) > 0 else {}

    # Mean per-image overlap
    def _mean(key): return float(np.mean([r[key] for r in per_image]))
    mean_ovl = {
        "iou"           : round(_mean("iou"),          4),
        "dice"          : round(_mean("dice"),         4),
        "overlap_coeff" : round(_mean("overlap_coeff"),4),
        "sensitivity"   : round(_mean("sensitivity"),  4),
        "precision"     : round(_mean("precision"),    4),
    } if per_image else {}

    # ── 6. In kết quả ─────────────────────────────────────────────
    SEP = "═" * 70

    print(f"\n{SEP}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ — {n_ok} ảnh test")
    print(SEP)

    def show_metrics(title, m):
        if not m: return
        print(f"\n  📊 {title}:")
        print(f"  {'Metric':<16} {'Value':>8}  Bar")
        print(f"  {'─'*45}")
        targets = ["accuracy","precision","recall","f1","specificity",
                   "iou","dice","auc","avg_prec"]
        for k in targets:
            v = m.get(k)
            if v is None: continue
            bar = "█" * int(v * 20)
            ok  = " ✅" if (
                (k == "recall" and v >= 0.80) or
                (k == "f1"     and v >= 0.75) or
                (k == "iou"    and v >= 0.50) or
                (k == "dice"   and v >= 0.60)
            ) else ""
            print(f"  {k:<16} {v:>8.4f}  {bar}{ok}")

    show_metrics("Pixel-level metrics (toàn tập)", m_px)
    show_metrics("Patch-level metrics (toàn tập)", m_pa)

    if mean_ovl:
        print(f"\n  📐 Trùng khớp mask (trung bình per-image):")
        print(f"  {'─'*45}")
        labels = {
            "iou"           : "IoU (Jaccard)",
            "dice"          : "Dice coefficient",
            "overlap_coeff" : "Overlap coefficient",
            "sensitivity"   : "Sensitivity (Recall)",
            "precision"     : "Precision",
        }
        for k, lbl in labels.items():
            v = mean_ovl[k]
            ok = " ✅" if (k == "sensitivity" and v >= 0.80) else ""
            print(f"  {lbl:<22}: {v*100:>6.1f}%{ok}")

    tp = m_px["tp"]; fp = m_px["fp"]
    fn = m_px["fn"]; tn = m_px["tn"]
    print(f"\n  Confusion Matrix (pixel-level):")
    print(f"  {'─'*40}")
    print(f"  {'':>14} Pred Neg    Pred Pos")
    print(f"  GT Negative  {tn:>10,}  {fp:>10,}   spec={tn/max(tn+fp,1):.3f}")
    print(f"  GT Positive  {fn:>10,}  {tp:>10,}   sens={tp/max(tp+fn,1):.3f}")

    n_det = sum(1 for r in per_image if r["n_pred_patches"] > 0)
    print(f"\n  Tổng: {n_ok} ảnh OK | {n_skip} bỏ qua | {n_fail} lỗi")
    print(f"  Model phát hiện shadow: {n_det}/{n_ok} ảnh")

    # ── 7. Lưu JSON ────────────────────────────────────────────────
    output = {
        "summary": {
            "n_ok": n_ok, "n_skip": n_skip, "n_fail": n_fail,
            "n_detected": n_det,
        },
        "pixel_metrics" : m_px,
        "patch_metrics" : m_pa,
        "mean_overlap"  : mean_ovl,
        "per_image"     : per_image,
    }
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}")
    print(f"  ✅ Hoàn thành!")
    print(f"  📁 Overlays  → {overlay_dir}")
    print(f"  📄 JSON      → {json_path}")
    print(SEP)

    return output


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate shadow detection trên 20% test split"
    )
    parser.add_argument(
        "--data_dir",
        default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed",
        help="Thư mục bongcan_processed chứa images_gray/, masks/, models/",
    )
    parser.add_argument("--thr_final",      default=THR_FINAL,      type=float)
    parser.add_argument("--n_theta",        default=N_THETA,        type=int)
    parser.add_argument("--n_r",            default=N_R_SAMPLES,    type=int)
    parser.add_argument("--te_low_pct",     default=TE_LOW_PCT,     type=int)
    parser.add_argument("--rf_min",         default=RF_MIN_PROB,    type=float)
    parser.add_argument("--max_shadow_pct", default=15,             type=int)
    parser.add_argument("--mask_patch_thr", default=0.30,           type=float,
        help="Tỉ lệ pixel shadow tối thiểu để patch = positive (default: 0.30)")
    args = parser.parse_args()

    print("=" * 70)
    print("  EVALUATE — Shadow Detection on 20% Test Split")
    print("  XANH = Bác sĩ khoanh  |  ĐỎ = Model  |  TÍM = Trùng nhau")
    print("=" * 70)
    print(f"  data_dir     : {args.data_dir}")
    print(f"  thr_final    : {args.thr_final}")
    print(f"  mask_thr     : {args.mask_patch_thr}")
    print("=" * 70 + "\n")

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