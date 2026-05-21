import os
import json
import base64
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def draw_labelme_json_on_image(json_path: str, image_path: str = None, show: bool = True):
    with open(json_path, 'r', encoding='utf-8') as f:
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
        # Lưu ý: Hàm của bạn đang lọc cố định nhãn 'bc'. 
        # Nếu muốn vẽ TẤT CẢ các nhãn, bạn có thể bỏ dòng 'if label == "bc":' này đi nhé.
        if label == 'bc':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=2)
            x, y = points[0]
            cv2.putText(image, label, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.title(os.path.basename(json_path))
        plt.show()
        
    return image

# ==================== ĐOẠN CODE CHẠY CHO THƯ MỤC CỦA BẠN ====================

# Đường dẫn đến thư mục chứa dữ liệu của bạn
folder_path = r"C:\Users\ThinkPad\DATN1\Data\bongcan\2400086315_002"

# Tìm tất cả các file .json trong thư mục đó
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

if not json_files:
    print(f"Không tìm thấy file JSON nào trong thư mục: {folder_path}")
else:
    print(f"Tìm thấy {len(json_files)} file JSON. Tiến hành vẽ...")
    
    for json_file in json_files:
        full_json_path = os.path.join(folder_path, json_file)
        
        # Đọc thử file JSON để lấy tên file ảnh tương ứng (nếu JSON không chứa imageData)
        with open(full_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Lấy tên file ảnh được lưu trong JSON (ví dụ: "image.jpg")
        image_name = data.get('imagePath')
        full_image_path = os.path.join(folder_path, image_name) if image_name else None
        
        # Nếu không tìm thấy file ảnh theo imagePath của JSON, ta thử tìm file ảnh trùng tên với file JSON
        if full_image_path is not None and not os.path.exists(full_image_path):
            # Thử các đuôi ảnh phổ biến như .jpg, .png, .jpeg
            base_name = os.path.splitext(full_json_path)[0]
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                if os.path.exists(base_name + ext):
                    full_image_path = base_name + ext
                    break

        # Gọi hàm vẽ
        try:
            draw_labelme_json_on_image(json_path=full_json_path, image_path=full_image_path, show=True)
        except Exception as e:
            print(f"Lỗi khi xử lý file {json_file}: {e}")