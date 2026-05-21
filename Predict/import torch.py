import torch

print("===== PyTorch GPU Test =====")

# 1. Kiểm tra version PyTorch
print("PyTorch version:", torch.__version__)

# 2. Kiểm tra PyTorch đang build với CUDA nào
print("CUDA version in PyTorch:", torch.version.cuda)

# 3. Kiểm tra có GPU CUDA không
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Number of GPUs:", torch.cuda.device_count())

    # 4. Tạo tensor trên GPU
    device = torch.device("cuda")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)

    z = x @ y

    print("Tensor x device:", x.device)
    print("Result z:")
    print(z)

    print("=> PyTorch đang chạy bằng GPU")
else:
    print("=> PyTorch chưa dùng được GPU, đang chạy CPU")

    # Test chạy CPU
    device = torch.device("cpu")
    x = torch.randn(3, 3).to(device)
    y = torch.randn(3, 3).to(device)

    z = x @ y

    print("Tensor x device:", x.device)
    print("Result z:")
    print(z)