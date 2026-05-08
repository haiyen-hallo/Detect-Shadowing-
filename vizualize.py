import cv2
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────
# 1. TẠO ẢNH GIẢ LẬP
# ────────────────────────────────────────────────────────
img = np.zeros((500, 600), dtype=np.uint8)

cx, cy = 300, -50   # Tâm hội tụ ảo (virtual apex)
R = 150             # Bán kính đầu dò

# Vẽ field of view
cv2.ellipse(img, (cx, cy), (500, 500), 0, 50, 130, 255, -1)
cv2.ellipse(img, (cx, cy), (R, R), 0, 50, 130, 0, -1)

# ────────────────────────────────────────────────────────
# 2. TẠO CHẤN TỬ TRÊN CUNG TRÒN
# ────────────────────────────────────────────────────────
theta_left = np.radians(-40)
theta_right = np.radians(40)
n_elements = 60

thetas = np.linspace(theta_left, theta_right, n_elements)

arc_x = cx + R * np.sin(thetas)
arc_y = cy + R * np.cos(thetas)

# ────────────────────────────────────────────────────────
# 3. VẼ
# ────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
plt.imshow(img, cmap='gray')

# Vẽ tâm
plt.scatter(cx, cy, c='orange', marker='*', s=300, edgecolors='black', label='Tâm hội tụ ảo')

# Vẽ chấn tử
plt.scatter(arc_x, arc_y, c='red', s=40, zorder=5, label='Bề mặt đầu dò')

# ────────────────────────────────────────────────────────
# 4. PHÓNG TIA (CÁCH ĐÚNG)
# ────────────────────────────────────────────────────────
ray_length = 350

for i in range(n_elements):
    # 👉 Vector pháp tuyến CHUẨN (từ tâm → điểm trên cung)
    dir_x = arc_x[i] - cx
    dir_y = arc_y[i] - cy

    # Chuẩn hóa
    norm = np.sqrt(dir_x**2 + dir_y**2)
    dir_x /= norm
    dir_y /= norm

    # Phóng tia xuống dưới
    end_x = arc_x[i] + dir_x * ray_length
    end_y = arc_y[i] + dir_y * ray_length

    plt.plot([arc_x[i], end_x], [arc_y[i], end_y], 'c-', alpha=0.8, linewidth=1.5)

    # (DEBUG) vẽ tia kéo ngược để kiểm tra hội tụ
    back_x = arc_x[i] - dir_x * 200
    back_y = arc_y[i] - dir_y * 200
    plt.plot([arc_x[i], back_x], [arc_y[i], back_y], 'y--', alpha=0.3)

# ────────────────────────────────────────────────────────
# 5. HIỂN THỊ
# ────────────────────────────────────────────────────────
plt.ylim(img.shape[0], cy - 50)
plt.xlim(0, img.shape[1])

plt.title("Fan Beam CHUẨN: Tất cả tia hội tụ về 1 tâm", fontsize=14, fontweight='bold')
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()