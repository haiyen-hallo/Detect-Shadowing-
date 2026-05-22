import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# ROUGH MASK (Dùng để so sánh/fallback)
# ─────────────────────────────────────────────────────────────────────────────

def rough_fan_mask(norm_img, thresh=0.05):
    bw = ((norm_img > thresh) * 255).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k_open)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: tìm transition pixel theo chiều dọc
# ─────────────────────────────────────────────────────────────────────────────

def _find_top_transition(img_gray, x, thresh=15):
    col = img_gray[:, int(x)]
    for y in range(len(col) - 1):
        if col[y] <= thresh and col[y + 1] > thresh:
            return y + 1
    bright = np.where(col > thresh)[0]
    return int(bright[0]) if len(bright) else 0

def _find_bottom_transition(img_gray, x, thresh=15):
    col = img_gray[:, int(x)]
    for y in range(len(col) - 1, 0, -1):
        if col[y] <= thresh and col[y - 1] > thresh:
            return y - 1
    bright = np.where(col > thresh)[0]
    return int(bright[-1]) if len(bright) else len(col) - 1

def _trace_ray(img_gray, cx, cy, angle_from_y, thresh=15):
    H, W = img_gray.shape
    sin_a = np.sin(angle_from_y)
    cos_a = np.cos(angle_from_y)
    max_r = int(np.sqrt(H**2 + W**2)) + 2

    for r in range(1, max_r):
        px = int(round(cx + r * sin_a))
        py = int(round(cy + r * cos_a))
        if px < 0 or px >= W or py < 0 or py >= H:
            return float(r)
        if img_gray[py, px] <= thresh:
            return float(r)
    return float(max_r)

# ─────────────────────────────────────────────────────────────────────────────
# HÀM CHÍNH
# ─────────────────────────────────────────────────────────────────────────────

def detect_probe_origin_geometric(img_gray, patch_h=32, bright_thresh=15):
    H, W = img_gray.shape

    # Bước 1: Hai điểm trên cung trên
    xa = int(0.35* W)
    xb = int(0.2 * W)
    ya = _find_top_transition(img_gray, xa, bright_thresh)
    yb = _find_top_transition(img_gray, xb, bright_thresh)

    # Bước 2: Giao điểm 2 tia
    dx_ab = float(xb - xa)
    dy_ab = float(yb - ya)

    if abs(dx_ab) < 1e-3:
        cx = W / 2.0
        cy = float(ya)
    else:
        m  = dy_ab / dx_ab
        cx = W / 2.0
        cy = ya + m * (cx - xa)

    # Bước 3: Góc theta
    theta = abs(float(np.arctan2(xa - cx, ya - cy)))

    # Bước 7 (sớm): Phát hiện zoom lớn
    bright_ratio = float(np.mean(img_gray > bright_thresh))
    if bright_ratio > 0.95:
        cy -= 2.0 * patch_h

    cy = float(min(cy, float(ya) - 1))

    # Bước 4: Kéo dài tia -> R_ray
    R_left  = _trace_ray(img_gray, cx, cy, -theta, bright_thresh)
    R_right = _trace_ray(img_gray, cx, cy, +theta, bright_thresh)
    R_ray   = (R_left + R_right) / 2.0

    # Bước 5: Bottom arc -> R_arc
    arc_dists = []
    step = max(1, W // 80)
    for x in range(int(0.02 * W), int(0.98 * W), step):
        yb_pt = _find_bottom_transition(img_gray, x, bright_thresh)
        if yb_pt >= H - 3:
            continue
        dist = float(np.sqrt((x - cx)**2 + (yb_pt - cy)**2))
        arc_dists.append(dist)

    if len(arc_dists) >= 5:
        R_arc = float(np.percentile(arc_dists, 90))
    else:
        R_arc = R_ray

    # Bước 6: R cuối cùng
    R = max(R_ray, R_arc)

    # Sanity checks
    theta = float(np.clip(theta, np.radians(8),  np.radians(88)))
    R     = float(np.clip(R,     float(patch_h),  np.sqrt(H**2 + W**2) * 1.1))

    return float(cx), float(cy), R, -theta, theta

# ─────────────────────────────────────────────────────────────────────────────
# BUILD MASK
# ─────────────────────────────────────────────────────────────────────────────

def build_geometric_mask(H, W, cx, cy, R, phi_left, phi_right, patch_h=16, R_inner=0):
    ys_g, xs_g = np.mgrid[0:H, 0:W].astype(float)
    dx    = xs_g - cx
    dy    = ys_g - cy
    dist  = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dx, dy)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[
        (dist  >= R_inner)    &
        (dist  <= R)          &
        (angle >= phi_left)   &
        (angle <= phi_right)  &
        (dy    >= -patch_h * 0.5) 
    ] = 255
    return mask

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TỔNG HỢP 
# ─────────────────────────────────────────────────────────────────────────────

def process_ultrasound(img_bgr, patch_h=32, bright_thresh=15):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) \
               if img_bgr.ndim == 3 else img_bgr.copy()
    H, W = img_gray.shape

    # 1. Rough mask 
    rough = rough_fan_mask(img_gray.astype(float) / 255.0)

    # 2. Detect geometry 
    cx, cy, R, phi_l, phi_r = detect_probe_origin_geometric(
        img_gray, patch_h=patch_h, bright_thresh=bright_thresh
    )

    # 3. Build mask hoàn hảo (Đã bổ sung truyền patch_h để tránh lỗi)
    geo_mask = build_geometric_mask(H, W, cx, cy, R, phi_l, phi_r, patch_h=patch_h)

    return {
        "cx": cx, "cy": cy, "R": R,
        "phi_l": phi_l, "phi_r": phi_r,
        "theta_deg": np.degrees(phi_r),
        "rough_mask": rough,
        "geo_mask": geo_mask
    }

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION TRỰC QUAN (MODULE MỚI)
# ─────────────────────────────────────────────────────────────────────────────

def view_ultrasound_processing(image_path):
    # 1. Đọc ảnh
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Lỗi: Không thể đọc ảnh tại '{image_path}'")
        return

    # 2. Chạy pipeline phân tích
    results = process_ultrasound(img_bgr)
    cx = results["cx"]
    cy = results["cy"]
    R = results["R"]
    phi_r = results["phi_r"]
    geo_mask = results["geo_mask"]
    rough_mask = results["rough_mask"]

    # 3. Vẽ đè (Overlay) lên ảnh gốc
    display_img = img_bgr.copy()
    
    # Vẽ viền Geometric Mask
    contours, _ = cv2.findContours(geo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(display_img, contours, -1, (0, 0, 255), 2)
        
    # Vẽ Tâm (cx, cy)
    cv2.circle(display_img, (int(cx), int(cy)), 8, (0, 255, 0), -1)
    text_y = int(cy) - 15 if cy > 20 else 30
    cv2.putText(display_img, f"Center({cx:.0f}, {cy:.0f})", (int(cx) - 80, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Vẽ đường thẳng chỉ Bán kính R
    edge_x = int(cx + R * np.sin(phi_r))
    edge_y = int(cy + R * np.cos(phi_r))
    cv2.line(display_img, (int(cx), int(cy)), (edge_x, edge_y), (0, 255, 255), 2)
    
    mid_x = int(cx + (R / 2) * np.sin(phi_r)) - 30
    mid_y = int(cy + (R / 2) * np.cos(phi_r))
    cv2.putText(display_img, f"R = {R:.0f}", (mid_x, mid_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 4. Vẽ Plot
    plt.figure(figsize=(18, 6))

    # Plot 1: Kết quả Fit Hình học
    plt.subplot(1, 3, 1)
    plt.title(f"Ray Tracing Fit\nTâm (cx={cx:.1f}, cy={cy:.1f}), R={R:.1f}")
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Plot 2: Rough Mask
    plt.subplot(1, 3, 2)
    plt.title("Rough Mask (Dữ liệu nền)")
    plt.imshow(rough_mask, cmap='gray')
    plt.axis('off')

    # Plot 3: Geo Mask
    plt.subplot(1, 3, 3)
    plt.title(f"Geometric Mask (Quạt siêu âm)\nGóc mở: {results['theta_deg'] * 2:.1f}°")
    plt.imshow(geo_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



# ─────────────────────────────────────────────────────────────────────────────
# THỰC THI CHƯƠNG TRÌNH
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Đường dẫn ảnh được nạp sẵn
    YOUR_IMAGE_PATH = r"C:\Users\ThinkPad\DATN\Data\bongcan\2400018899_004\2400018899_004_right.jpg"
    view_ultrasound_processing(YOUR_IMAGE_PATH)
