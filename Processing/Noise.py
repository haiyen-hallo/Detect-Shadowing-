import os, sys, json, glob, base64, argparse, io
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt

TARGET_LABEL     = "bc"
PATCH_H          = 16
PATCH_W          = 16
GLCM_BINS        = 32
MASK_PATCH_THR   = 0.30

# 12 đặc trưng cơ bản + 3 đặc trưng ngữ cảnh cột (column-context)
FEATURE_NAMES = [
    "contrast","homogeneity","correlation","energy",
    "mean","std","skewness","entropy",
    "row_ratio","col_ratio","local_contrast","below_bright",
    # [NEW] Column-context: phản ánh nguyên lý vật lý bóng cản
    "above_max_mean",   # độ sáng cực đại phía trên cùng cột (reflector siêu âm)
    "col_dark_ratio",   # mức tối tương đối so với trung bình cột
    "mean_drop",        # độ sụt giảm mean đột ngột từ patch phía trên
]


# ═══════════════════════════════════════════════════════════════════
# HÀM VẼ BÓNG CẢN BẮT BUỘC & TRỰC QUAN HÓA
# ═══════════════════════════════════════════════════════════════════

def draw_labelme_json_on_image(json_path:str, image_path: str = None, show: bool = True):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if data.get('imageData') is not None:
        image_data = base64.b64decode(data['imageData'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = np.array(image)
    elif image_path is not None:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image data found in JSON and no image path provided.")
        
    for shape in data['shapes']:
        label = shape['label']
        if label =='bc':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
            x, y = points[0]
            cv2.putText(image, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title('image')
        plt.show()
        
    return image


# ═══════════════════════════════════════════════════════════════════
# TÌM FILES
# ═══════════════════════════════════════════════════════════════════

def find_json_files(data_root):
    files  = glob.glob(os.path.join(data_root, "*.json"))
    files += glob.glob(os.path.join(data_root, "**", "*.json"), recursive=True)
    files  = [f for f in files if ".backup" not in f]
    files  = sorted(set(files))
    print(f"[Tìm file] {data_root}  →  {len(files)} JSON")
    return files


# ═══════════════════════════════════════════════════════════════════
# PREPROCESSING CƠ BẢN
# ═══════════════════════════════════════════════════════════════════

def to_grayscale(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def normalize(gray):
    img = gray.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img-lo)/(hi-lo) if hi-lo > 1e-6 else np.zeros_like(img)


# ═══════════════════════════════════════════════════════════════════
# SHADOW MASK 
# ═══════════════════════════════════════════════════════════════════

def make_shadow_mask(shapes, orig_h, orig_w,
                     crop_x, crop_y, crop_w, crop_h,
                     panel_offset_x, meta, out_h, out_w):
    """
    Vẽ polygon shadow lên mask theo đúng annotation JSON.
    """
    canvas = Image.new("L", (out_w, out_h), 0)
    draw   = ImageDraw.Draw(canvas)
    count  = 0

    scale  = meta["scale"]
    pad_x  = meta["pad_x"]
    pad_y  = meta["pad_y"]

    for shape in shapes:
        if shape.get("label") != TARGET_LABEL:
            continue
        if shape.get("shape_type") != "polygon":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue

        pts_t = []
        for x, y in pts:
            x1 = max(0., min(x - panel_offset_x - crop_x, float(crop_w)))
            y1 = max(0., min(y - crop_y, float(crop_h)))
            pts_t.append((min(x1*scale + pad_x, float(out_w-1)),
                           min(y1*scale + pad_y, float(out_h-1))))
        if len(pts_t) >= 3:
            draw.polygon(pts_t, fill=255)
            count += 1

    mask_np = np.array(canvas, dtype=np.uint8)
    return mask_np, count

# ═══════════════════════════════════════════════════════════════════
# GLCM BUILDER & FEATURE EXTRACTION 
# ═══════════════════════════════════════════════════════════════════

def _build_glcm(patch_q, bins):
    glcm = np.zeros((bins, bins), dtype=np.float32)
    ph, pw = patch_q.shape
    for dy, dx in [(0,1),(1,1),(1,0),(1,-1)]:
        if   dy==0: rp,cp = patch_q[:,:pw-1], patch_q[:,1:]
        elif dx==0: rp,cp = patch_q[:ph-1,:], patch_q[1:,:]
        elif dx>0:  rp,cp = patch_q[:ph-1,:pw-1], patch_q[1:,1:]
        else:       rp,cp = patch_q[:ph-1,1:], patch_q[1:,:pw-1]
        rv, cv_ = rp.ravel(), cp.ravel()
        np.add.at(glcm, (rv,cv_), 1)
        np.add.at(glcm, (cv_,rv), 1)
    s = float(glcm.sum())
    if s > 0: glcm /= s
    return glcm


def compute_patch_features(norm_img, mask, patch_h, patch_w, bins,
                           mask_patch_thr=MASK_PATCH_THR):
    # ĐÃ XÓA parameter fan_mask
    H, W   = norm_img.shape
    n_rows = H // patch_h
    n_cols = W // patch_w
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"patch {patch_h}×{patch_w} quá lớn vs {H}×{W}")

    q       = np.clip((norm_img*(bins-1)).astype(np.int32), 0, bins-1)
    idx     = np.arange(bins, dtype=np.float32)
    II, JJ  = np.meshgrid(idx, idx, indexing="ij")
    diff_sq = (II-JJ)**2
    denom_h = 1.0 + diff_sq
    ks      = max(3, (patch_h*3)|1)
    nbr     = cv2.GaussianBlur(norm_img, (ks,ks), sigmaX=ks/3.0)
    bright  = (norm_img > 0.70).astype(np.float32)

    # Tất cả feature grids (kể cả 3 feature cột mới — sẽ được điền sau vòng lặp)
    f = {k: np.zeros((n_rows, n_cols), dtype=np.float64) for k in FEATURE_NAMES}
    label_p = np.zeros((n_rows, n_cols), dtype=np.int32)
    coverage_p = np.zeros((n_rows, n_cols), dtype=np.float32)

    n_sh_total = 0

    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r*patch_h, (r+1)*patch_h
            x0, x1 = c*patch_w, (c+1)*patch_w

            pf  = norm_img[y0:y1, x0:x1]
            pq  = q[y0:y1, x0:x1]
            pm  = mask[y0:y1, x0:x1]

            cov = float(pm.sum()) / (255.0 * pm.size)
            coverage_p[r, c] = cov
            label_p[r, c]    = 1 if cov >= mask_patch_thr else 0
            if cov > 0: n_sh_total += 1

            g = _build_glcm(pq, bins)
            f["contrast"][r,c]  = float((g*diff_sq).sum())
            f["homogeneity"][r,c] = float((g/denom_h).sum())
            mx=float((II*g).sum()); my=float((JJ*g).sum())
            sx=float(np.sqrt(max(float((g*(II-mx)**2).sum()), 1e-12)))
            sy=float(np.sqrt(max(float((g*(JJ-my)**2).sum()), 1e-12)))
            f["correlation"][r,c] = float(((II-mx)*(JJ-my)*g).sum()/(sx*sy))
            f["energy"][r,c]     = float((g**2).sum())
            pnz = g[g > 0]
            f["entropy"][r,c]    = float(-np.sum(pnz*np.log2(pnz+1e-12)))

            vals  = pf.ravel().astype(np.float64)
            m_val = float(vals.mean())
            s_val = float(vals.std())
            f["mean"][r,c]  = m_val
            f["std"][r,c]   = s_val
            if s_val > 1e-6:
                f["skewness"][r,c] = float(np.clip(np.mean((vals-m_val)**3)/(s_val**3), -10, 10))

            f["row_ratio"][r,c]      = r / max(n_rows-1, 1)
            f["col_ratio"][r,c]      = c / max(n_cols-1, 1)
            nbr_m = float(nbr[y0:y1, x0:x1].mean())
            f["local_contrast"][r,c] = m_val / (nbr_m + 1e-6)
            ab = bright[:r*patch_h, x0:x1] if r > 0 else np.zeros((1,1))
            f["below_bright"][r,c]   = float(ab.mean()) if ab.size else 0.0

    # ─── POST-LOOP: 3 đặc trưng ngữ cảnh cột ─────────────────────────────────
    # Tính sau vòng lặp chính vì phụ thuộc vào f["mean"] đã hoàn chỉnh
    mean_grid = f["mean"]                             # (n_rows, n_cols)
    col_mean  = mean_grid.mean(axis=0)                # (n_cols,)  trung bình cột

    for c in range(n_cols):
        cm = col_mean[c]
        for r in range(n_rows):
            cur = mean_grid[r, c]

            # above_max_mean: độ sáng tối đa của các patch phía trên trong cột
            # → capture hyperechoic reflector; bóng cản thật luôn có reflector sáng phía trên
            if r > 0:
                f["above_max_mean"][r, c] = float(mean_grid[:r, c].max())
            # r==0 giữ nguyên 0.0

            # col_dark_ratio: patch này tối hơn trung bình cột bao nhiêu (0→1)
            # → phân biệt bóng cản (tối liên tục dọc cột) với speckle ngẫu nhiên
            f["col_dark_ratio"][r, c] = float(np.clip((cm - cur) / (cm + 1e-6), 0., 1.))

            # mean_drop: sụt giảm mean đột ngột từ patch liền trên → patch hiện tại
            # → dấu hiệu khởi đầu bóng cản (chuyển từ sáng sang tối)
            if r > 0:
                f["mean_drop"][r, c] = float(np.clip(mean_grid[r-1, c] - cur, 0., 1.))
            # r==0 giữ nguyên 0.0

    patch_features = np.stack([f[k].ravel() for k in FEATURE_NAMES], axis=1).astype(np.float32)
    patch_labels   = label_p.ravel().astype(np.int32)
    patch_coverage = coverage_p.ravel().astype(np.float32)
    patch_features = np.nan_to_num(patch_features, nan=0., posinf=1., neginf=-1.)

    n_sh  = int(patch_labels.sum())
    n_tot = len(patch_labels)
    print(f"    [patches] grid={n_rows}×{n_cols}  shadow={n_sh}/{n_tot} ({n_sh/max(n_tot,1)*100:.1f}%)")

    def _u8(a):
        lo, hi = float(a.min()), float(a.max())
        if hi-lo < 1e-10: return np.zeros_like(a, dtype=np.uint8)
        return ((a-lo)/(hi-lo)*255).clip(0,255).astype(np.uint8)

    def _up(a):
        return cv2.resize(a, (W,H), interpolation=cv2.INTER_NEAREST)

    maps = {k: _up(_u8(f[k])) for k in FEATURE_NAMES}
    maps["patch_label"]   = _up((label_p*255).astype(np.uint8))

    return {
        **maps,
        "patch_features"  : patch_features,
        "patch_labels"    : patch_labels,
        "patch_coverage"  : patch_coverage,
        "n_shadow"        : n_sh,
        "n_total"         : n_tot,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE 
# ═══════════════════════════════════════════════════════════════════

def run(data_root, output_root, patch_h=PATCH_H, patch_w=PATCH_W,
        mask_patch_thr=MASK_PATCH_THR, debug=False):

    img_dir     = os.path.join(output_root, "images_gray")
    feat_dir    = os.path.join(output_root, "feature_maps")
    plabel_dir  = os.path.join(output_root, "patch_labels")
    mask_dir    = os.path.join(output_root, "masks")
    compare_dir = os.path.join(output_root, "compare")
    for d in [img_dir, feat_dir, plabel_dir, mask_dir, compare_dir]:
        os.makedirs(d, exist_ok=True)

    json_files = find_json_files(data_root)
    if not json_files:
        print("Không tìm thấy JSON!"); sys.exit(1)

    print(f"\n{'─'*65}")
    print(f"  {len(json_files)} file  patch={patch_h}×{patch_w}  bins={GLCM_BINS}")
    print(f"  Features: {len(FEATURE_NAMES)} (Base Features)")
    print(f"  crop_fan : ĐÃ LOẠI BỎ (Dùng ảnh kích thước gốc 100%)")
    print(f"  Label rule: coverage >= {mask_patch_thr:.2f}")
    print(f"{'─'*65}")

    ok = 0; skip = 0
    all_features = []; all_labels = []; all_coverage = []; all_stems = []
    total_shadow = 0

    for jf in json_files:
        stem = Path(jf).stem
        json_dir = os.path.dirname(jf)
        
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            shapes = data.get("shapes", [])
            bc_shapes = [s for s in shapes if s.get("label") == TARGET_LABEL]
            if not bc_shapes:
                print(f"  [SKIP] {stem} — không có polygon '{TARGET_LABEL}'")
                skip += 1; continue

            # --- 1. TÌM ẢNH ĐỒNG BỘ TUYỆT ĐỐI VỚI HÀM VẼ OVERLAY ---
            img_np = None
            img_fallback_path = None
            
            # Ưu tiên 1: Đọc từ Base64
            if data.get('imageData') is not None:
                try:
                    image_data = base64.b64decode(data['imageData'])
                    img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
                    img_np = np.array(img_pil)
                except Exception:
                    pass
            
            # Ưu tiên 2: Tìm ảnh file vật lý trùng tên
            if img_np is None:
                valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
                for ext in valid_exts:
                    p_check = os.path.join(json_dir, stem + ext)
                    if os.path.exists(p_check):
                        img_fallback_path = p_check
                        img_cv = cv2.imread(p_check)
                        img_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                        break
                        
            # Ưu tiên 3: Tìm theo imagePath
            if img_np is None:
                rel = data.get("imagePath", "")
                if rel:
                    p_check = os.path.join(json_dir, rel)
                    if os.path.exists(p_check):
                        img_fallback_path = p_check
                        img_cv = cv2.imread(p_check)
                        img_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            if img_np is None:
                print(f"  [SKIP] {stem} — không tìm thấy ảnh")
                skip += 1; continue

            # --- 2. VẼ OVERLAY BẰNG HÀM CỦA BẠN ---
            try:
                overlay_img = draw_labelme_json_on_image(json_path=jf, image_path=img_fallback_path, show=False)
                overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(compare_dir, stem + "_overlay.png"), overlay_bgr)
            except Exception as e_overlay:
                print(f"  [WARN-OVERLAY] Không thể tạo ảnh overlay cho {stem}: {e_overlay}")

            # --- 3. PIPELINE XỬ LÝ (Trên cùng 1 hệ tọa độ ảnh gốc) ---
            orig_h, orig_w = img_np.shape[:2]

            gray_orig    = to_grayscale(img_np)
            gray_clean   = cv2.medianBlur(gray_orig, ksize=3)
            H_img, W_img = gray_clean.shape
            norm_img     = normalize(gray_clean)

            meta_d = {"scale": 1., "pad_x": 0, "pad_y": 0, "new_w": W_img, "new_h": H_img}

            # Tạo Mask - KHÔNG truyền crop_x, crop_y -> Khớp 100% tọa độ
            mask, bc_count = make_shadow_mask(
                shapes, orig_h, orig_w, 
                0, 0, W_img, H_img, 
                0, meta_d, H_img, W_img
            )

            if mask.sum() == 0:
                print(f"  [SKIP] {stem} — mask rỗng")
                continue

            print(f"  → {stem}  ({W_img}×{H_img}) {bc_count} poly:")
            feats = compute_patch_features(norm_img, mask, patch_h, patch_w, GLCM_BINS, mask_patch_thr)

            all_features.append(feats["patch_features"])
            all_labels.append(feats["patch_labels"])
            all_coverage.append(feats["patch_coverage"])
            all_stems.extend([stem] * len(feats["patch_labels"]))
            total_shadow += feats["n_shadow"]

            # Lưu ảnh kích thước chuẩn
            Image.fromarray(gray_clean).save(os.path.join(img_dir, stem + ".png"))
            Image.fromarray(feats["patch_label"]).save(os.path.join(plabel_dir, stem + ".png"))
            Image.fromarray(mask).save(os.path.join(mask_dir, stem + ".png"))

            # Tạm thời chỉ lưu 3 loại feature map chính để kiểm tra
            for fn in ["contrast", "mean", "local_contrast"]:
                if fn in feats and isinstance(feats[fn], np.ndarray):
                    Image.fromarray(feats[fn]).save(os.path.join(feat_dir, f"{stem}_{fn}.png"))

            plv = np.zeros((H_img, W_img, 3), dtype=np.uint8)
            plv[feats["patch_label"] > 0] = (30, 30, 160)
            
            for ri in range(0, H_img, patch_h): cv2.line(plv, (0,ri), (W_img-1,ri), (40,40,40), 1)
            for ci in range(0, W_img, patch_w): cv2.line(plv, (ci,0), (ci,H_img-1), (40,40,40), 1)
            
            cnts_m, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(plv, cnts_m, -1, (0,220,80), 2)
            # ĐÃ XÓA vẽ tâm origin

            gray_bgr = cv2.cvtColor(gray_clean, cv2.COLOR_GRAY2BGR)
            cmp = np.hstack([
                gray_bgr,
                cv2.applyColorMap(feats["contrast"],    cv2.COLORMAP_HOT),
                cv2.applyColorMap(feats["mean"],        cv2.COLORMAP_BONE),
                plv,
            ])
            for i, lbl in enumerate(["ORIGINAL","CONTRAST","MEAN","LABEL & MASK"]):
                cv2.putText(cmp, lbl, (W_img*i+4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,200), 1)
            cv2.imwrite(os.path.join(compare_dir, stem+".png"), cmp)
            print(f"  [OK] {stem}")

            ok += 1

        except Exception as e:
            print(f"  [ERR] {stem}: {e}")
            skip += 1

    if not all_features:
        print("\n[WARN] Không có data!"); return {}

    X   = np.vstack(all_features).astype(np.float32)
    Y   = np.concatenate(all_labels).astype(np.int32)
    COV = np.concatenate(all_coverage).astype(np.float32)

    dataset_path = os.path.join(output_root, "dataset.npz")
    np.savez(
        dataset_path,
        X=X,
        Y=Y,
        Y_soft=COV,
        coverage=COV,
        stems=np.array(all_stems),
        feature_names=np.array(FEATURE_NAMES),
        mask_patch_thr=np.float32(mask_patch_thr),
    )

    n_pos = int(Y.sum()); n_neg = len(Y) - n_pos
    ratio = n_neg / max(n_pos, 1)

    print(f"\n{'='*65}")
    print(f"  OK={ok}  Skip={skip}")
    print(f"  Dataset  : {dataset_path}")
    print(f"  X={X.shape}  shadow={n_pos:,} ({n_pos/len(Y)*100:.1f}%)")
    print(f"  Ratio 1:{ratio:.1f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise.py — 12 Base Features + Polygon Shadow Mask")
    parser.add_argument("--data", default=r"C:\Users\ThinkPad\Graduation project\Data\bongcan")
    parser.add_argument("--output", default=r"C:\Users\ThinkPad\Graduation project\Data\bongcan_processed")
    parser.add_argument("--patch_h", default=PATCH_H, type=int)
    parser.add_argument("--patch_w", default=PATCH_W, type=int)
    parser.add_argument("--mask_patch_thr", default=MASK_PATCH_THR, type=float)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run(args.data, args.output, args.patch_h, args.patch_w, args.mask_patch_thr, args.debug)
