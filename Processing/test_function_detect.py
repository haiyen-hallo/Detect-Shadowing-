r""""""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def rough_fan_mask(norm_img, thresh=0.05):
    bw = ((norm_img > thresh) * 255).astype(np.uint8)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask

def detect_probe_origin(fan_mask, PATCH_H=32):
    H, W   = fan_mask.shape
    ys, xs = np.where(fan_mask > 0)

    def _fallback():
        counts   = np.bincount(ys,minlength=H) if len(ys)>0 else np.zeros(H)
        vrows    = np.where(counts>=8)[0]
        y_top    = int(vrows[0]) if len(vrows) else 0
        cx_f     = W/2.0
        fan_w_f  = float(np.percentile(xs,98)-np.percentile(xs,2)) if len(xs)>0 else W*0.8
        cy_f     = y_top - fan_w_f*0.25    
        R_f      = fan_w_f*0.55
        return cx_f, cy_f, R_f, -np.radians(65), np.radians(65)

    if len(ys) < 50: return _fallback()

    top_pts = []
    for col in range(W):
        col_rows = ys[xs == col]
        if len(col_rows) > 0:
            top_pts.append((float(col), float(col_rows.min())))
    if len(top_pts) < 10: return _fallback()

    raw_x = np.array([p[0] for p in top_pts])
    raw_y = np.array([p[1] for p in top_pts])
    k_sm  = max(1, len(raw_y)//25)
    raw_y = np.convolve(raw_y, np.ones(k_sm)/k_sm, mode='same')

    n = len(raw_x); m = max(2, int(n*0.08))
    px = raw_x[m:n-m]; py = raw_y[m:n-m]
    if len(px) < 6: return _fallback()

    A = np.column_stack([2*px, 2*py, np.ones(len(px))])
    b = px**2 + py**2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = float(res[0])
        cy = float(res[1])
        R  = float(np.sqrt(max(res[2]+cx**2+cy**2, 1.0)))
        cx = float(np.clip(cx, W*0.05, W*0.95))
        cy_min_arc = float(py.min())
        fan_width  = float(px.max()-px.min())
        if cy > cy_min_arc:
            cy = cy_min_arc - fan_width*0.25
            R  = float(np.sqrt(((px-cx)**2+(py-cy)**2).mean()))
        if R < PATCH_H or R > max(H,W)*5: return _fallback()
    except Exception:
        return _fallback()

    phi_left  = float(np.arctan2(px[0]-cx,  py[0]-cy))
    phi_right = float(np.arctan2(px[-1]-cx, py[-1]-cy))
    if phi_left > phi_right: phi_left, phi_right = phi_right, phi_left
    return cx, cy, R, phi_left, phi_right

# ==========================================
# 2. HÀM BẮN TIA (RAY CASTING)
# ==========================================
def _sample_ray_vals(norm_img, ox, oy, px, py, t0, t1, n=20):
    H, W = norm_img.shape
    dx = px - ox; dy = py - oy
    vals, pts = [], [] 
    for i in range(n):
        t  = t0 + (t1-t0)*i/max(n-1, 1)
        sx = int(round(ox + dx*t))
        sy = int(round(oy + dy*t))
        if 0 <= sx < W and 0 <= sy < H:
            vals.append(float(norm_img[sy, sx]))
            pts.append((sx, sy))
    return np.array(vals, dtype=np.float32), pts


# ==========================================
# 3. THỰC THI VÀ TRỰC QUAN HÓA (ANNOTATION)
# ==========================================
IMAGE_PATH = r"C:\Users\ThinkPad\DATN1\Data\bongcan\240049829_003\240049829_003_left.jpg"
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is not None:
    norm_img = img / 255.0
    H, W = norm_img.shape
    PATCH_H = 32
    
    # --- BƯỚC A: TỰ ĐỘNG TÌM 5 THAM SỐ HÌNH HỌC ---
    mask = rough_fan_mask(norm_img)
    cx, cy, R, phi_left, phi_right = detect_probe_origin(mask, PATCH_H)
    
    # Chuẩn bị ảnh màu để vẽ
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Vì cy có thể bị âm (nằm ngoài màn hình), ta tạo một khoảng margin phía trên để dễ nhìn nếu cần
    # Nhưng để đơn giản, ta cứ vẽ lên ảnh hiện tại, những phần lòi ra ngoài CV2 sẽ tự cắt.
    
    # --- BƯỚC B: VẼ CÁC TIA (RAYS) ---
    num_rays = 20
    ray_length = max(H, W) * 1.5 # Cho tia đủ dài để chạm đáy màn hình
    angles_rad = np.linspace(phi_left, phi_right, num_rays)
    
    for angle_rad in angles_rad:
        px = int(cx + ray_length * math.sin(angle_rad))
        py = int(cy + ray_length * math.cos(angle_rad))
        
        # Bắn tia lấy mẫu từ ngay dưới bán kính trong (t0=0.1) đến cuối ảnh
        vals, pts = _sample_ray_vals(norm_img, cx, cy, px, py, t0=0.1, t1=1.0, n=50)
        
        # Vẽ tia mờ (xanh lam) và điểm lấy mẫu (vàng)
        cv2.line(display_img, (int(cx), int(cy)), (px, py), (255, 0, 0), 1)
        for sx, sy in pts:
            cv2.circle(display_img, (sx, sy), 1, (0, 255, 255), -1)

    # --- BƯỚC C: ANNOTATE (CHÚ THÍCH) 5 THAM SỐ HÌNH HỌC ---
    
    # 1. Chú thích (cx, cy) - Chấm đỏ to
    cv2.circle(display_img, (int(cx), int(cy)), 8, (0, 0, 255), -1)
    # Căn chữ xuôi xuống 1 chút nếu cy bị âm để chữ không văng khỏi màn hình
    text_y = max(20, int(cy) - 10) 
    cv2.putText(display_img, f"(cx, cy)", (int(cx) - 30, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 2. Chú thích phi_left và phi_right - Hai tia giới hạn màu Xanh Lá dày
    px_left = int(cx + ray_length * math.sin(phi_left))
    py_left = int(cy + ray_length * math.cos(phi_left))
    cv2.line(display_img, (int(cx), int(cy)), (px_left, py_left), (0, 255, 0), 3)
    cv2.putText(display_img, f"phi_left", (px_left + 10, py_left - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    px_right = int(cx + ray_length * math.sin(phi_right))
    py_right = int(cy + ray_length * math.cos(phi_right))
    cv2.line(display_img, (int(cx), int(cy)), (px_right, py_right), (0, 255, 0), 3)
    cv2.putText(display_img, f"phi_right", (px_right - 100, py_right - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 3. Chú thích R (Bán kính trong) - Cung tròn màu Hồng (Magenta)
    start_angle = np.degrees(phi_left)
    end_angle = np.degrees(phi_right)
    cv2.ellipse(display_img, (int(cx), int(cy)), (int(R), int(R)), 90, start_angle, end_angle, (255, 0, 255), 3)
    
    # Viết chữ "R" ở giữa cung tròn
    mid_angle = (phi_left + phi_right) / 2
    rx = int(cx + (R - 20) * math.sin(mid_angle))
    ry = int(cy + (R - 20) * math.cos(mid_angle))
    cv2.putText(display_img, f"R", (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # --- BƯỚC D: HIỂN THỊ ---
    plt.figure(figsize=(10, 10))
    plt.title("Trực quan hóa 5 tham số Hình học và Ray Casting", fontsize=14)
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

else:
    print(f"Vui lòng kiểm tra lại đường dẫn ảnh: {IMAGE_PATH}")







# Hàm khác 

def _sample_ray_vals(norm_img, ox, oy, px, py, t0, t1, n=20):
    H, W = norm_img.shape
    dx = px - ox; dy = py - oy
    vals, pts = [], [] # Thêm pts để vẽ
    for i in range(n):
        t  = t0 + (t1-t0)*i/max(n-1, 1)
        sx = int(round(ox + dx*t))
        sy = int(round(oy + dy*t))
        if 0 <= sx < W and 0 <= sy < H:
            vals.append(float(norm_img[sy, sx]))
            pts.append((sx, sy))
    return np.array(vals, dtype=np.float32), pts

# ==========================================
# PHẦN MỚI: BẮN NHIỀU TIA THEO HÌNH QUẠT
# ==========================================
IMAGE_PATH = r"C:\Users\ThinkPad\DATN1\Data\bongcan\240049829_003\240049829_003_left.jpg"
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is not None:
    norm_img = img / 255.0
    H, W = norm_img.shape
    
    # 1. Cài đặt Tâm đầu dò (Giữa màn hình, mép trên)
    ox, oy = W // 2, 0
    
    # 2. Cài đặt thông số quét 
    num_rays = 20           # Số lượng tia muốn bắn (20 tia)
    ray_length = H - 50     # Chiều dài của tia (bắn gần tới đáy ảnh)
    angle_start = -45       # Góc bên trái (độ)
    angle_end = 45          # Góc bên phải (độ)
    
    # Chuẩn bị ảnh để vẽ màu lên
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 3. Vòng lặp bắn 20 tia
    # Chia đều khoảng góc từ -45 đến 45 thành 20 phần
    angles = np.linspace(angle_start, angle_end, num_rays)
    
    for angle_deg in angles:
        # Chuyển độ sang radian để tính sin, cos
        angle_rad = math.radians(angle_deg)
        
        # Tính tọa độ đích (px, py) bằng lượng giác
        # Trục Y hướng xuống dưới nên oy + (cos * length)
        px = int(ox + ray_length * math.sin(angle_rad))
        py = int(oy + ray_length * math.cos(angle_rad))
        
        # Gọi hàm lấy mẫu cho tia hiện tại (Lấy 50 điểm trên mỗi tia cho nét)
        vals, pts = _sample_ray_vals(norm_img, ox, oy, px, py, t0=0.1, t1=1.0, n=50)
        
        # 4. Trực quan hóa: Vẽ tia lên ảnh
        # Vẽ đường thẳng mờ màu xanh
        cv2.line(display_img, (ox, oy), (px, py), (255, 0, 0), 1)
        
        # Chấm các điểm lấy mẫu màu vàng nhỏ li ti trên tia đó
        for sx, sy in pts:
            cv2.circle(display_img, (sx, sy), 1, (0, 255, 255), -1)

    # Hiển thị kết quả
    plt.figure(figsize=(8, 8))
    plt.title(f"Quét {num_rays} tia (Tỏa ra từ {angle_start}° đến {angle_end}°)")
    plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Vui lòng kiểm tra lại đường dẫn ảnh!") 
 















import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. HÀM TẠO MASK NHÁP (Cũ)
# ==========================================
def rough_fan_mask(norm_img, thresh=0.05):
    bw = ((norm_img > thresh) * 255).astype(np.uint8)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(mask, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return mask

# ==========================================
# 2. HÀM TÌM TÂM BẰNG TOÁN HỌC (Của bạn)
# ==========================================
def detect_probe_origin(fan_mask, PATCH_H=32):
    H, W   = fan_mask.shape
    ys, xs = np.where(fan_mask > 0)

    def _fallback():
        counts   = np.bincount(ys,minlength=H) if len(ys)>0 else np.zeros(H)
        vrows    = np.where(counts>=8)[0]
        y_top    = int(vrows[0]) if len(vrows) else 0
        cx_f     = W/2.0
        fan_w_f  = float(np.percentile(xs,98)-np.percentile(xs,2)) if len(xs)>0 else W*0.8
        cy_f     = y_top - fan_w_f*0.25    
        R_f      = fan_w_f*0.55
        return cx_f, cy_f, R_f, -np.radians(65), np.radians(65)

    if len(ys) < 50: return _fallback()

    top_pts = []
    for col in range(W):
        col_rows = ys[xs == col]
        if len(col_rows) > 0:
            top_pts.append((float(col), float(col_rows.min())))
    if len(top_pts) < 10: return _fallback()

    raw_x = np.array([p[0] for p in top_pts])
    raw_y = np.array([p[1] for p in top_pts])
    k_sm  = max(1, len(raw_y)//25)
    raw_y = np.convolve(raw_y, np.ones(k_sm)/k_sm, mode='same')

    n = len(raw_x); m = max(2, int(n*0.08))
    px = raw_x[m:n-m]; py = raw_y[m:n-m]
    if len(px) < 6: return _fallback()

    A = np.column_stack([2*px, 2*py, np.ones(len(px))])
    b = px**2 + py**2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = float(res[0])
        cy = float(res[1])
        R  = float(np.sqrt(max(res[2]+cx**2+cy**2, 1.0)))
        cx = float(np.clip(cx, W*0.05, W*0.95))
        cy_min_arc = float(py.min())
        fan_width  = float(px.max()-px.min())
        if cy > cy_min_arc:
            cy = cy_min_arc - fan_width*0.25
            R  = float(np.sqrt(((px-cx)**2+(py-cy)**2).mean()))
        if R < PATCH_H or R > max(H,W)*5: return _fallback()
    except Exception:
        return _fallback()

    phi_left  = float(np.arctan2(px[0]-cx,  py[0]-cy))
    phi_right = float(np.arctan2(px[-1]-cx, py[-1]-cy))
    if phi_left > phi_right: phi_left, phi_right = phi_right, phi_left
    return cx, cy, R, phi_left, phi_right

# ==========================================
# 3. HÀM VẼ MẶT NẠ HÌNH HỌC HOÀN HẢO
# ==========================================
def draw_perfect_geometric_mask(rough_mask, cx, cy, inner_R, phi_left, phi_right, PATCH_H=32):
    H, W = rough_mask.shape
    geom_mask = np.zeros((H, W), dtype=np.uint8)

    # 3.1 Tìm bán kính ngoài (Outer Radius) dựa vào phần tử xa nhất của rough mask
    ys, xs = np.where(rough_mask > 0)
    if len(ys) == 0: return geom_mask
    distances = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    outer_R = np.max(distances)

    # 3.2 Đổi góc từ Radian sang Độ cho OpenCV
    start_angle = np.degrees(phi_left)
    end_angle = np.degrees(phi_right)

    # 3.3 Vẽ toàn bộ hình quạt lớn (màu trắng)
    # Trục 90 độ của OpenCV hướng thẳng xuống dưới (trục Y)
    cv2.ellipse(geom_mask, (int(cx), int(cy)), (int(outer_R), int(outer_R)), 
                90, start_angle, end_angle, 255, -1)

    # 3.4 Cắt bỏ phần cong ở trên cùng đầu dò (tô màu đen)
    # Trừ hao +-5 độ để đảm bảo cắt sạch mép
    cv2.ellipse(geom_mask, (int(cx), int(cy)), (int(inner_R), int(inner_R)), 
                90, start_angle - 5, end_angle + 5, 0, -1)

    # 3.5 Gọt viền an toàn (Erode) giống hệt file Predict.py
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PATCH_H, PATCH_H))
    final_mask = cv2.erode(geom_mask, ek, iterations=1)

    return final_mask

# ==========================================
# 4. CHẠY THỬ TRÊN ẢNH CỦA BẠN
# ==========================================
IMAGE_PATH = r"C:\Users\ThinkPad\DATN1\Data\bongcan\2300160559_017\2300160559_017_right.jpg"  # <-- THAY ĐƯỜNG DẪN ẢNH VÀO ĐÂY
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"Không tìm thấy ảnh tại: {IMAGE_PATH}")
else:
    norm_img = img / 255.0
    PATCH_H = 32

    # Bước A: Lấy mask nháp
    rough_mask = rough_fan_mask(norm_img)

    # Bước B: Tính toán hình học
    cx, cy, R, phi_left, phi_right = detect_probe_origin(rough_mask, PATCH_H)

    # Bước C: Vẽ mask hoàn hảo
    perfect_mask = draw_perfect_geometric_mask(rough_mask, cx, cy, R, phi_left, phi_right, PATCH_H)

    # Bước D: Áp mask lên ảnh gốc
    overlay_img = img.copy()
    overlay_img[perfect_mask == 0] = 0

    # Hiển thị
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.title("1. Ảnh gốc")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.title("2. Rough Mask (bị lẹm góc)")
    plt.imshow(rough_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("3. Geometric Mask (Hoàn hảo)")
    plt.imshow(perfect_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.title("4. Ảnh Overlay Cuối Cùng")
    plt.imshow(overlay_img, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show() 


