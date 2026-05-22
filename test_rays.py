import cv2
import numpy as np
import matplotlib.pyplot as plt

def auto_detect_ultrasound_roi_fixed(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Nhị phân hóa cơ bản
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # 2. MORPH_OPEN (Cực kỳ quan trọng): Xóa sổ các chi tiết nhỏ lơ lửng 
    # (như chữ Voluson E6, thước đo bên trái) trước khi chúng dính vào quạt
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    # 3. MORPH_CLOSE: Lấp các lỗ đen nhỏ bên trong quạt
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # 4. Tìm contour lớn nhất
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return img, (0,0,img.shape[1],img.shape[0]), img, np.ones_like(gray)
    
    max_contour = max(contours, key=cv2.contourArea)

    # 5. Tạo Mask thô (KHÔNG DÙNG CONVEX HULL NỮA)
    raw_mask = np.zeros_like(gray)
    cv2.drawContours(raw_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    # 6. TRICK LÀM MƯỢT: Blur mask thật mạnh rồi threshold lại 
    # Sẽ làm bo tròn mọi góc răng cưa mà vẫn giữ nguyên chỗ lõm ở đỉnh
    mask_blurred = cv2.GaussianBlur(raw_mask, (31, 31), 0)
    _, smooth_mask = cv2.threshold(mask_blurred, 128, 255, cv2.THRESH_BINARY)

    # 7. Cuối cùng, bóp nhỏ mask lại 15 pixel để an toàn tuyệt đối, né viền
    kernel_erode = np.ones((15, 15), np.uint8)
    safe_mask = cv2.erode(smooth_mask, kernel_erode, iterations=1)

    # Lấy Khung Bounding Box từ mask an toàn
    x, y, w, h = cv2.boundingRect(safe_mask)
    
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = safe_mask[y:y+h, x:x+w]

    # --- TẠO ẢNH DEBUG ---
    debug_img = img.copy()
    final_contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if final_contours:
        cv2.drawContours(debug_img, final_contours, -1, (0, 255, 0), 2)
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return cropped_img, (x, y, w, h), debug_img, cropped_mask

def evaluate_image(image_path):
    print(f"Đang xử lý: {image_path}...")
    cropped, bbox, debug, cropped_mask = auto_detect_ultrasound_roi_fixed(image_path)
    
    if debug is not None:
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.title("Ảnh gốc + Viền chuẩn (Xanh) & Khung (Đỏ)")
        plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Mask an toàn (Giữ được độ lõm)")
        plt.imshow(cropped_mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Kết quả Crop ({bbox[2]}x{bbox[3]})")
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    TEST_IMAGE_PATH = r"C:\Users\ThinkPad\DATN1\Data\bongcan\2300160559_002\2300160559_002_left.jpg"
    evaluate_image(TEST_IMAGE_PATH)