
import os, sys, json, glob, base64, argparse
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw
import cv2

TARGET_LABEL     = "bc"
PATCH_H          = 16
PATCH_W          = 16
GLCM_BINS        = 32

# Mở rộng shadow mask xuống dưới để bắt reverberation artifact
# Bác sĩ khoanh "vùng ảnh hưởng", không chỉ vùng đen thuần túy
SHADOW_EXTEND_PX = 24   # ~1.5 patch xuống dưới

# Tên features
FEATURE_NAMES_BASE = [
    "contrast","homogeneity","correlation","energy",
    "mean","std","skewness","entropy",
    "row_ratio","col_ratio","local_contrast","below_bright",
]
FEATURE_NAMES_GEO = [
    "dist_origin_norm",   # khoảng cách probe→patch, normalized
    "angle_axis_norm",    # góc lệch so với trục giữa fan [-1, 1]
    "ray_mean_above",     # mean intensity phía trên patch theo tia
    "lateral_drop",       # mức độ tối hơn lân cận ngang
    "vert_continuity",    # liên tục tối theo cột dưới patch
    "reverb_score",       # tỉ lệ pixel sáng trong cột tối (reverberation)
]
FEATURE_NAMES = FEATURE_NAMES_BASE + FEATURE_NAMES_GEO


# ═══════════════════════════════════════════════════════════════════
# LOAD — ưu tiên _L/_R (đã tách bởi split_json.py)
# ═══════════════════════════════════════════════════════════════════

def find_json_files(data_root):
    files  = glob.glob(os.path.join(data_root, "*.json"))
    files += glob.glob(os.path.join(data_root, "**", "*.json"), recursive=True)
    # Bỏ qua file .backup
    files  = [f for f in files if ".backup" not in f]
    files  = sorted(set(files))
    print(f"[Tìm file] {data_root}  →  {len(files)} JSON")
    return files


def load_images_for_json(json_data, json_path):
    """
    Trả về list of (PIL.Image, sample_id).
    Ưu tiên ảnh đã cắt rời (_L/_left, _R/_right).
    Nếu không có → ảnh gốc / imagePath / base64.
    """
    json_dir   = os.path.dirname(json_path)
    stem       = Path(json_path).stem
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp",
                  ".JPG", ".JPEG", ".PNG", ".BMP"]
    images_found = []

    # --- Ưu tiên 1: ảnh đã cắt rời (_L/_left và _R/_right) ---
    for ext in valid_exts:
        for suffix_L, suffix_R in [("_L", "_R"), ("_left", "_right")]:
            path_L = os.path.join(json_dir, stem + suffix_L + ext)
            path_R = os.path.join(json_dir, stem + suffix_R + ext)
            if os.path.exists(path_L):
                images_found.append(
                    (Image.open(path_L).convert("RGB"), stem + suffix_L))
            if os.path.exists(path_R):
                images_found.append(
                    (Image.open(path_R).convert("RGB"), stem + suffix_R))
    if images_found:
        return images_found

    # --- Ưu tiên 2: ảnh cùng tên JSON ---
    for ext in valid_exts:
        p = os.path.join(json_dir, stem + ext)
        if os.path.exists(p):
            return [(Image.open(p).convert("RGB"), stem)]

    # --- Ưu tiên 3: imagePath trong JSON ---
    rel = json_data.get("imagePath", "")
    if rel:
        p = os.path.join(json_dir, rel)
        if os.path.exists(p):
            return [(Image.open(p).convert("RGB"), stem)]

    # --- Ưu tiên 4: Base64 ---
    b64 = json_data.get("imageData", "")
    if b64:
        try:
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            return [(img, stem)]
        except Exception:
            pass
    return []


# ═══════════════════════════════════════════════════════════════════
# PREPROCESSING — GIỐNG FILE GỐC (không Otsu, không dual panel)
# ═══════════════════════════════════════════════════════════════════

def crop_fan_region(gray):
    """
    Crop bounding box fan bằng threshold=10 (KHÔNG dùng Otsu).
    Otsu adaptive sẽ thay đổi theo từng ảnh → features không nhất quán.
    """
    _, bw = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = gray.shape
        return gray, 0, 0, w, h
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    m = 4; x = max(0, x-m); y = max(0, y-m)
    w = min(gray.shape[1]-x, w+2*m)
    h = min(gray.shape[0]-y, h+2*m)
    return gray[y:y+h, x:x+w], x, y, w, h


def to_grayscale(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def normalize(gray):
    img = gray.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img-lo)/(hi-lo) if hi-lo > 1e-6 else np.zeros_like(img)


def get_fan_mask_norm(norm_img, thresh=0.05):
    """Fan mask từ normalized image, thresh=0.05 (giống file gốc và Predict.py)."""
    bw = ((norm_img > thresh) * 255).astype(np.uint8)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PATCH_H, PATCH_H))
    return cv2.erode(mask, ek, iterations=1)


def detect_probe_origin(fan_mask):
    """
    Tâm đầu dò = trung điểm 2 đỉnh trên cùng fan mask.
    Dùng cho geometric features.
    """
    H, W   = fan_mask.shape
    ys, xs = np.where(fan_mask > 0)
    if len(ys) == 0:
        return (W // 2, 0)
    y_min  = int(ys.min())
    top_xs = xs[ys <= y_min + 10]
    if len(top_xs) > 0:
        ox = (int(np.min(top_xs)) + int(np.max(top_xs))) // 2
        return (ox, y_min)
    return (W // 2, 0)


# ═══════════════════════════════════════════════════════════════════
# SHADOW MASK — mở rộng xuống dưới (Reverberation Artifact)
# ═══════════════════════════════════════════════════════════════════

def make_mask_with_reverberation(shapes, orig_h, orig_w,
                                  crop_x, crop_y, crop_w, crop_h,
                                  panel_offset_x, meta, out_h, out_w,
                                  extend_px=SHADOW_EXTEND_PX):
    """
    Vẽ polygon shadow lên mask, sau đó MỞ RỘNG XUỐNG DƯỚI thêm extend_px pixel.

    Lý do mở rộng (theo giải thích bác sĩ):
    - Bóng cản trong siêu âm không chỉ là vùng đen hoàn toàn
    - Có reverberation artifact: tia phản xạ qua lại tạo dải sáng giả
      bên trong "vùng bóng cản" — bác sĩ vẫn khoanh vùng đó
    - Mở rộng mask để model học được cả vùng ảnh hưởng này
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

    # Mở rộng xuống dưới bằng dilation theo chiều dọc
    # Kernel dọc: chỉ mở rộng theo chiều Y (xuống dưới theo hướng tia)
    if extend_px > 0 and mask_np.sum() > 0:
        kernel = np.ones((extend_px, 1), dtype=np.uint8)
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
        # Giữ trong fan (không mở rộng ra ngoài ảnh)
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)

    return mask_np, count


# ═══════════════════════════════════════════════════════════════════
# GLCM BUILDER (giống file gốc)
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


def _sample_ray_vals(norm_img, ox, oy, px, py, t0, t1, n=20):
    """Lấy n mẫu intensity dọc tia từ probe (t=0) qua patch center (t=1)."""
    H, W = norm_img.shape
    dx = px - ox; dy = py - oy
    vals = []
    for i in range(n):
        t  = t0 + (t1-t0)*i/max(n-1, 1)
        sx = int(round(ox + dx*t))
        sy = int(round(oy + dy*t))
        if 0 <= sx < W and 0 <= sy < H:
            vals.append(float(norm_img[sy, sx]))
    return np.array(vals, dtype=np.float32) if vals else np.zeros(1)


# ═══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 18 FEATURES
# ═══════════════════════════════════════════════════════════════════

def compute_patch_features(norm_img, fan_mask, mask, patch_h, patch_w, bins):
    """
    18 features per patch = 12 gốc + 6 geometric.

    LABEL: coverage > 0 (bất kỳ pixel nào trong mask mở rộng → label=1)
    Bao gồm cả vùng reverberation artifact mà bác sĩ khoanh.
    """
    H, W   = norm_img.shape
    n_rows = H // patch_h
    n_cols = W // patch_w
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"patch {patch_h}×{patch_w} quá lớn vs {H}×{W}")

    # Pre-compute dùng chung
    q       = np.clip((norm_img*(bins-1)).astype(np.int32), 0, bins-1)
    idx     = np.arange(bins, dtype=np.float32)
    II, JJ  = np.meshgrid(idx, idx, indexing="ij")
    diff_sq = (II-JJ)**2
    denom_h = 1.0 + diff_sq
    ks      = max(3, (patch_h*3)|1)
    nbr     = cv2.GaussianBlur(norm_img, (ks,ks), sigmaX=ks/3.0)
    bright  = (norm_img > 0.70).astype(np.float32)

    # Probe origin cho geometric features
    origin  = detect_probe_origin(fan_mask)
    ox, oy  = float(origin[0]), float(origin[1])

    # Fan geometry
    ys_fan, xs_fan = np.where(fan_mask > 0)
    if len(xs_fan) > 0:
        fan_diag = float(np.sqrt(
            ((xs_fan - ox)**2 + (ys_fan - oy)**2).max()))
        fan_cx   = float(xs_fan.mean())
        fan_half_w = max(float(xs_fan.max() - xs_fan.min()) / 2.0, 1.0)
    else:
        fan_diag   = float(max(H, W))
        fan_cx     = W / 2.0
        fan_half_w = W / 2.0

    # Mảng lưu features
    f         = {k: np.zeros((n_rows, n_cols), dtype=np.float64)
                  for k in FEATURE_NAMES}
    label_p   = np.zeros((n_rows, n_cols), dtype=np.int32)
    coverage_p = np.zeros((n_rows, n_cols), dtype=np.float32)

    n_sh_total = 0
    n_in_fan   = 0

    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r*patch_h, (r+1)*patch_h
            x0, x1 = c*patch_w, (c+1)*patch_w

            pf  = norm_img[y0:y1, x0:x1]
            pq  = q[y0:y1, x0:x1]
            pm  = mask[y0:y1, x0:x1]

            # LABEL (coverage > 0, đúng paper)
            cov = float(pm.sum()) / (255.0 * pm.size)
            coverage_p[r, c] = cov
            label_p[r, c]    = 1 if cov > 0 else 0
            if cov > 0: n_sh_total += 1

            # ── 12 features gốc ────────────────────────────────────
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
                f["skewness"][r,c] = float(
                    np.clip(np.mean((vals-m_val)**3)/(s_val**3), -10, 10))

            f["row_ratio"][r,c]      = r / max(n_rows-1, 1)
            f["col_ratio"][r,c]      = c / max(n_cols-1, 1)
            nbr_m = float(nbr[y0:y1, x0:x1].mean())
            f["local_contrast"][r,c] = m_val / (nbr_m + 1e-6)
            ab = bright[:r*patch_h, x0:x1] if r > 0 else np.zeros((1,1))
            f["below_bright"][r,c]   = float(ab.mean()) if ab.size else 0.0

            # ── 6 Geometric features (vật lý siêu âm) ──────────────
            py_c = (r + 0.5) * patch_h
            px_c = (c + 0.5) * patch_w
            dx   = px_c - ox; dy = py_c - oy
            dist = float(np.sqrt(dx**2 + dy**2))

            # 1. Khoảng cách probe → patch (normalized 0..1)
            #    Bóng cản thường ở xa đầu dò hơn tissue thông thường
            f["dist_origin_norm"][r,c] = dist / (fan_diag + 1e-6)

            # 2. Góc lệch so với trục giữa fan (normalized -1..1)
            #    Bóng cản xuất phát từ bờ/trung tâm khối u → có vị trí góc đặc trưng
            f["angle_axis_norm"][r,c] = (px_c - fan_cx) / (fan_half_w + 1e-6)

            if dist > patch_h:
                # 3. Mean intensity phía trên patch theo hướng tia (t: 0.1 → 0.85)
                #    Shadow thật: phía trên phải CÓ reflector (sáng)
                #    Không phải shadow: phía trên cũng tối (mô bình thường)
                above_v = _sample_ray_vals(
                    norm_img, ox, oy, px_c, py_c, 0.10, 0.85, n=20)
                f["ray_mean_above"][r,c] = float(above_v.mean())

                # 4. Lateral drop: patch tối hơn lân cận ngang bao nhiêu
                #    Bóng cản: tối hơn rõ rệt so với 2 bên
                #    Tissue: không có sự chênh lệch rõ
                nbr_lat = []
                for dc in [-2, -1, 1, 2]:
                    nc = c + dc
                    if 0 <= nc < n_cols:
                        nbr_lat.append(float(
                            norm_img[y0:y1, nc*patch_w:(nc+1)*patch_w].mean()))
                if nbr_lat:
                    f["lateral_drop"][r,c] = float(np.mean(nbr_lat)) - m_val
                else:
                    f["lateral_drop"][r,c] = 0.0

                # 5. Vertical continuity: mức độ liên tục tối theo cột dưới patch
                #    Bóng cản: tiếp tục tối theo hướng tia xuống sâu
                #    Noise/tissue: không có continuity
                below_v = _sample_ray_vals(
                    norm_img, ox, oy, px_c, py_c, 1.05, 1.60, n=20)
                # Tỉ lệ pixel tối trong vùng dưới
                f["vert_continuity"][r,c] = float(
                    (below_v < 0.40).sum()) / max(len(below_v), 1)

                # 6. Reverberation score: tỉ lệ pixel sáng trong cột tối
                #    Bóng cản thật: có thể có dải sáng giả (reverberation)
                #    bên trong vùng tối → bác sĩ vẫn khoanh
                #    Feature này giúp model phân biệt "bóng có reverb" vs tissue
                full_col_v = _sample_ray_vals(
                    norm_img, ox, oy, px_c, py_c, 0.80, 1.60, n=30)
                col_dark   = float((full_col_v < 0.40).mean())
                col_bright = float((full_col_v > 0.60).mean())
                # Reverb = cột phần lớn tối nhưng có điểm sáng rải rác
                if col_dark > 0.5:
                    f["reverb_score"][r,c] = col_bright
                else:
                    f["reverb_score"][r,c] = 0.0
            else:
                # Patch quá gần probe, không có tia hợp lệ
                f["ray_mean_above"][r,c]  = m_val
                f["lateral_drop"][r,c]    = 0.0
                f["vert_continuity"][r,c] = 0.0
                f["reverb_score"][r,c]    = 0.0

    n_total = n_rows * n_cols
    n_in_fan_count = 0  # đếm sau

    patch_features = np.stack(
        [f[k].ravel() for k in FEATURE_NAMES], axis=1).astype(np.float32)
    patch_labels   = label_p.ravel().astype(np.int32)
    patch_coverage = coverage_p.ravel().astype(np.float32)
    patch_features = np.nan_to_num(patch_features, nan=0., posinf=1., neginf=-1.)

    n_sh  = int(patch_labels.sum())
    n_tot = len(patch_labels)
    print(f"    [patches] grid={n_rows}×{n_cols}  "
          f"shadow={n_sh}/{n_tot} ({n_sh/max(n_tot,1)*100:.1f}%)")

    # Visualization maps
    def _u8(a):
        lo, hi = float(a.min()), float(a.max())
        if hi-lo < 1e-10: return np.zeros_like(a, dtype=np.uint8)
        return ((a-lo)/(hi-lo)*255).clip(0,255).astype(np.uint8)

    def _up(a):
        return cv2.resize(a, (W,H), interpolation=cv2.INTER_NEAREST)

    maps = {k: _up(_u8(f[k])) for k in FEATURE_NAMES}
    maps["patch_label"]   = _up((label_p*255).astype(np.uint8))
    maps["origin"]        = origin

    return {
        **maps,
        "patch_features"  : patch_features,
        "patch_labels"    : patch_labels,
        "patch_coverage"  : patch_coverage,
        "n_shadow"        : n_sh,
        "n_total"         : n_tot,
    }


# ═══════════════════════════════════════════════════════════════════
# STATISTICS
# ═══════════════════════════════════════════════════════════════════

def compute_mean_std(img_dir):
    files = glob.glob(os.path.join(img_dir, "*.png"))
    if not files: return 0.5, 0.25
    s, sq, n = 0., 0., 0
    for fn in files:
        arr = np.array(Image.open(fn)).astype(np.float64) / 255.
        s += arr.sum(); sq += (arr**2).sum(); n += arr.size
    mean = s / n
    std  = float(np.sqrt(max(sq/n - mean**2, 1e-8)))
    print(f"[stats] mean={mean:.4f}  std={std:.4f}  ({len(files)} ảnh)")
    return float(mean), float(std)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run(data_root, output_root, patch_h=PATCH_H, patch_w=PATCH_W,
        extend_px=SHADOW_EXTEND_PX, debug=False):

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
    print(f"  Features: {len(FEATURE_NAMES)} "
          f"(12 gốc + {len(FEATURE_NAMES_GEO)} geometric)")
    print(f"  Geometric: {FEATURE_NAMES_GEO}")
    print(f"  crop_fan : threshold=10 (KHÔNG Otsu)")
    print(f"  fan_mask : normalized thresh=0.05")
    print(f"  Shadow extend: {extend_px}px xuống dưới (reverberation)")
    print(f"  Label rule: coverage > 0 (bất kỳ pixel shadow → label=1)")
    print(f"{'─'*65}")

    ok = 0; skip = 0
    all_features = []; all_labels = []; all_coverage = []; all_stems = []
    total_shadow = 0

    for jf in json_files:
        stem = Path(jf).stem
        try:
            with open(jf, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # Load ảnh (ưu tiên _L/_R)
            images = load_images_for_json(data, jf)
            if not images:
                print(f"  [SKIP] {stem} — không tìm thấy ảnh")
                skip += 1; continue

            shapes = data.get("shapes", [])
            bc_shapes = [s for s in shapes
                         if s.get("label") == TARGET_LABEL]
            if not bc_shapes:
                print(f"  [SKIP] {stem} — không có polygon '{TARGET_LABEL}'")
                skip += 1; continue

            json_w = data.get("imageWidth", 0)

            for img_pil, sample_id in images:
                img_np   = np.array(img_pil.convert("RGB"), dtype=np.uint8)
                orig_h, orig_w = img_np.shape[:2]

                # Pipeline chuẩn (KHÔNG dual panel, KHÔNG Otsu)
                gray_orig    = to_grayscale(img_np)
                gray_cropped, cx, cy, cw, ch = crop_fan_region(gray_orig)
                gray_clean   = cv2.medianBlur(gray_cropped, ksize=3)
                H_img, W_img = gray_clean.shape
                norm_img     = normalize(gray_clean)

                # Fan mask từ normalized
                fan_mask = get_fan_mask_norm(norm_img, thresh=0.05)

                meta_d = {"scale": 1., "pad_x": 0, "pad_y": 0,
                           "new_w": W_img, "new_h": H_img}

                # Shadow mask với mở rộng reverberation
                mask, bc_count = make_mask_with_reverberation(
                    shapes, orig_h, orig_w, cx, cy, cw, ch,
                    0,          # panel_offset_x = 0 (ảnh đã tách riêng)
                    meta_d, H_img, W_img,
                    extend_px=extend_px)

                if mask.sum() == 0:
                    print(f"  [SKIP] {sample_id} — mask rỗng")
                    continue

                print(f"  → {sample_id}  ({W_img}×{H_img}) {bc_count} poly:")
                feats = compute_patch_features(
                    norm_img, fan_mask, mask, patch_h, patch_w, GLCM_BINS)

                all_features.append(feats["patch_features"])
                all_labels.append(feats["patch_labels"])
                all_coverage.append(feats["patch_coverage"])
                all_stems.extend([sample_id] * len(feats["patch_labels"]))
                total_shadow += feats["n_shadow"]

                # Lưu ảnh
                Image.fromarray(gray_clean).save(
                    os.path.join(img_dir, sample_id + ".png"))
                Image.fromarray(feats["patch_label"]).save(
                    os.path.join(plabel_dir, sample_id + ".png"))
                Image.fromarray(mask).save(
                    os.path.join(mask_dir, sample_id + ".png"))

                for fn in ["contrast", "mean", "local_contrast",
                            "lateral_drop", "vert_continuity", "reverb_score"]:
                    if fn in feats and isinstance(feats[fn], np.ndarray):
                        Image.fromarray(feats[fn]).save(
                            os.path.join(feat_dir, f"{sample_id}_{fn}.png"))

                # Compare visualization
                origin = feats.get("origin", (W_img//2, 0))
                plv = np.zeros((H_img, W_img, 3), dtype=np.uint8)
                plv[feats["patch_label"] > 0] = (30, 30, 160)
                # Vẽ lưới patch
                for ri in range(0, H_img, patch_h):
                    cv2.line(plv, (0,ri), (W_img-1,ri), (40,40,40), 1)
                for ci in range(0, W_img, patch_w):
                    cv2.line(plv, (ci,0), (ci,H_img-1), (40,40,40), 1)
                # Vẽ shadow mask (outline)
                cnts_m, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(plv, cnts_m, -1, (0,220,80), 2)
                # Probe origin
                cv2.circle(plv, (int(origin[0]), int(origin[1])), 6,
                           (0,220,220), -1)

                gray_bgr = cv2.cvtColor(gray_clean, cv2.COLOR_GRAY2BGR)
                cmp = np.hstack([
                    gray_bgr,
                    cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2BGR),
                    cv2.applyColorMap(feats["contrast"],    cv2.COLORMAP_HOT),
                    cv2.applyColorMap(feats["mean"],        cv2.COLORMAP_BONE),
                    cv2.applyColorMap(feats["reverb_score"],cv2.COLORMAP_JET),
                    plv,
                ])
                for i, lbl in enumerate(["CLEAN","CROP","CONTRAST","MEAN","REVERB","LABEL"]):
                    cv2.putText(cmp, lbl, (W_img*i+4, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,200), 1)
                cv2.imwrite(os.path.join(compare_dir, sample_id+".png"), cmp)
                print(f"  [OK] {sample_id}")

            ok += 1

        except Exception as e:
            print(f"  [ERR] {stem}: {e}")
            import traceback; traceback.print_exc()
            skip += 1

    if not all_features:
        print("\n[WARN] Không có data!"); return {}

    X   = np.vstack(all_features).astype(np.float32)
    Y   = np.concatenate(all_labels).astype(np.int32)
    COV = np.concatenate(all_coverage).astype(np.float32)

    dataset_path = os.path.join(output_root, "dataset.npz")
    np.savez(dataset_path,
             X=X, Y=Y, coverage=COV,
             stems=np.array(all_stems),
             feature_names=np.array(FEATURE_NAMES))

    n_pos = int(Y.sum()); n_neg = len(Y) - n_pos
    ratio = n_neg / max(n_pos, 1)

    print(f"\n{'='*65}")
    print(f"  OK={ok}  Skip={skip}")
    print(f"  Dataset  : {dataset_path}")
    print(f"  X={X.shape}  shadow={n_pos:,} ({n_pos/len(Y)*100:.1f}%)")
    print(f"  Ratio 1:{ratio:.1f}")
    print(f"  Features : {len(FEATURE_NAMES)}  (12 gốc + 6 geometric)")
    print(f"  Shadow patches (incl. reverb extension): {total_shadow:,}")
    print(f"{'='*65}")

    mean, std = compute_mean_std(img_dir)
    stats = {
        "mean": round(mean, 6), "std": round(std, 6),
        "patch_h": patch_h, "patch_w": patch_w,
        "shadow_extend_px": extend_px,
        "n_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "geometric_features": FEATURE_NAMES_GEO,
        "label_rule": "coverage > 0 (paper rule, mask includes reverberation zone)",
        "crop_fan": "threshold=10 (NOT Otsu)",
        "fan_mask": "normalized thresh=0.05",
        "n_images": ok,
        "n_shadow_patches": total_shadow,
    }
    with open(os.path.join(output_root, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Noise.py — 18 Features + Reverberation Mask")
    parser.add_argument("--data",
        default=r"C:\Users\ThinkPad\DATN\Data\bongcan")
    parser.add_argument("--output",
        default=r"C:\Users\ThinkPad\DATN\Data\bongcan_processed")
    parser.add_argument("--patch_h",   default=PATCH_H,          type=int)
    parser.add_argument("--patch_w",   default=PATCH_W,          type=int)
    parser.add_argument("--extend_px", default=SHADOW_EXTEND_PX, type=int,
        help="Số pixel mở rộng mask xuống dưới (reverberation)")
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args()

    print("=" * 65)
    print("  NOISE.PY — 18 Features (12 gốc + 6 Geometric)")
    print("  Shadow mask mở rộng để bắt Reverberation Artifact")
    print("  crop_fan: threshold=10 (KHÔNG Otsu)")
    print("  Mỗi ảnh = 1 JSON riêng (đã chạy split_json.py)")
    print("=" * 65)
    print(f"  Data      : {args.data}")
    print(f"  Output    : {args.output}")
    print(f"  Patch     : {args.patch_h}×{args.patch_w}px")
    print(f"  Extend    : {args.extend_px}px (reverberation)")
    print(f"  Geometric : {FEATURE_NAMES_GEO}")
    print("=" * 65 + "\n")

    run(args.data, args.output, args.patch_h, args.patch_w,
        args.extend_px, args.debug)