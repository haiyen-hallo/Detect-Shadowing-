import numpy as np

data = np.load(r"C:\Users\ThinkPad\DATN\Data\bongcan_processed\dataset.npz")
print("Các biến có trong file:", data.files)

if "stems" in data.files:
    print("Mẫu tên 5 patch đầu tiên:", data["stems"][:5])
else:
    print("CẢNH BÁO: File npz của bạn CHƯA lưu biến 'stems'!")