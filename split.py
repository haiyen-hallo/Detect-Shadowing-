
import os
import json
import glob
import re
from pathlib import Path

def remove_duplicate_jsons(data_dir):
    """
    Bước 1: Tìm các file copy (vd: file(1).json). 
    Mở ra so sánh nội dung với file gốc. Nếu giống hệt -> XÓA FILE COPY.
    """
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    duplicates_removed = 0
    
    for jf in json_files:
        if "_left.json" in jf or "_right.json" in jf or "backup" in jf:
            continue
            
        # Dùng Regex bắt các file có đuôi kiểu (1), (2), hoặc - Copy
        match = re.search(r'^(.*?)\s*(?:\(\d+\)|-\s*Copy)\.json$', jf, flags=re.IGNORECASE)
        
        if match:
            # Tìm lại tên file gốc. VD: "anh_01(1).json" -> gốc là "anh_01.json"
            base_jf = match.group(1) + ".json"
            
            if os.path.exists(base_jf):
                try:
                    with open(base_jf, "r", encoding="utf-8") as f1, open(jf, "r", encoding="utf-8") as f2:
                        data1 = json.load(f1)
                        data2 = json.load(f2)
                    
                    # So sánh danh sách tọa độ (shapes)
                    if data1.get("shapes") == data2.get("shapes"):
                        os.remove(jf)
                        print(f"[DỌN RÁC] Đã xóa file copy trùng lặp: {Path(jf).name}")
                        duplicates_removed += 1
                except Exception as e:
                    print(f"[ERR] Lỗi khi dọn rác file {jf}: {e}")
                    
    return duplicates_removed

def split_all_jsons(data_dir):
    # ==========================================
    # BƯỚC 1: Dọn dẹp file copy (1) trùng lặp
    # ==========================================
    print("Đang quét để dọn dẹp file copy thừa...")
    removed = remove_duplicate_jsons(data_dir)
    if removed > 0:
        print(f"-> Đã dọn dẹp {removed} file JSON copy!\n")
        
    # ==========================================
    # BƯỚC 2: Tiến hành chia tài sản JSON
    # ==========================================
    print("Đang quét để phân chia JSON...")
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    count_success = 0
    count_skip = 0
    
    for jf in json_files:
        # Bỏ qua file ko tồn tại (vừa bị xóa ở bước 1) hoặc file đã là left/right
        if not os.path.exists(jf): continue
        if "_left.json" in jf or "_right.json" in jf or "backup" in jf:
            continue
            
        stem = Path(jf).stem
        json_dir = os.path.dirname(jf)
        
        # 1. KIỂM TRA XEM CÓ ẢNH CẮT RỜI TRONG FOLDER HAY KHÔNG
        has_split_images = False
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        
        for ext in valid_exts:
            if (os.path.exists(os.path.join(json_dir, f"{stem}_left{ext}")) or 
                os.path.exists(os.path.join(json_dir, f"{stem}_right{ext}")) or
                os.path.exists(os.path.join(json_dir, f"{stem}_L{ext}")) or 
                os.path.exists(os.path.join(json_dir, f"{stem}_R{ext}"))):
                has_split_images = True
                break
        
        # NẾU CHỈ CÓ 1 ẢNH ĐƠN -> BỎ QUA AN TOÀN
        if not has_split_images:
            print(f"  [SKIP] {stem}.json -> Chỉ có ảnh đơn, giữ nguyên.")
            count_skip += 1
            continue
            
        # NẾU CÓ ẢNH RỜI -> TIẾN HÀNH CẮT JSON
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            total_w = data.get("imageWidth", 1136)
            split_x = total_w / 2.0
            
            left_data = data.copy()
            right_data = data.copy()
            
            # Cập nhật tên ảnh tham chiếu và chiều rộng cho file mới
            left_data["imagePath"] = f"{stem}_left.jpg"
            right_data["imagePath"] = f"{stem}_right.jpg"
            left_data["imageWidth"] = int(split_x)
            right_data["imageWidth"] = int(split_x)
            left_data["imageData"] = None
            right_data["imageData"] = None
            
            left_shapes = []
            right_shapes = []
            
            # Phân loại và trừ tọa độ
            for shape in data.get("shapes", []):
                pts = shape.get("points", [])
                if not pts: continue
                
                mean_x = sum(p[0] for p in pts) / len(pts)
                
                if mean_x < split_x:
                    left_shapes.append(shape)
                else:
                    new_shape = shape.copy()
                    # Ảnh Phải: Trừ đi tọa độ bù (568)
                    new_shape["points"] = [[p[0] - split_x, p[1]] for p in pts]
                    right_shapes.append(new_shape)
                    
            left_data["shapes"] = left_shapes
            right_data["shapes"] = right_shapes
            
            # LƯU 2 FILE JSON MỚI VÀO CÙNG FOLDER
            left_json_path = os.path.join(json_dir, f"{stem}_left.json")
            right_json_path = os.path.join(json_dir, f"{stem}_right.json")
            
            with open(left_json_path, "w", encoding="utf-8") as f:
                json.dump(left_data, f, indent=2, ensure_ascii=False)
            with open(right_json_path, "w", encoding="utf-8") as f:
                json.dump(right_data, f, indent=2, ensure_ascii=False)
                
            # ĐỔI ĐUÔI FILE GỐC THÀNH .backup
            backup_path = jf + ".backup"
            if os.path.exists(backup_path): os.remove(backup_path)
            os.rename(jf, backup_path)
                
            print(f"  [OK] {stem}.json -> Đã tách thành left/right và backup gốc.")
            count_success += 1
            
        except Exception as e:
            print(f"  [ERR] Lỗi khi cắt {stem}.json: {e}")

    print(f"\n{'='*55}")
    print(f"HOÀN THÀNH QUY TRÌNH CHUẨN BỊ DỮ LIỆU!")
    print(f" - Đã xóa rác file (1) trùng  : {removed} files")
    print(f" - Đã tách & backup gốc       : {count_success} files (Ảnh đôi)")
    print(f" - Đã bỏ qua an toàn          : {count_skip} files (Ảnh đơn)")
    print(f"{'='*55}")

if __name__ == "__main__":
    # Đường dẫn data của bạn
    DATA_DIR = r"C:\Users\ThinkPad\DATN\Data\bongcan"
    
    print(f"Đang xử lý thư mục: {DATA_DIR}\n")
    split_all_jsons(DATA_DIR)