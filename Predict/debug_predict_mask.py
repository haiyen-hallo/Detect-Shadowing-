"""
debug_feature_mismatch.py
─────────────────────────────────────────────────────────────────────
Mục đích: Xác định chính xác xem Predict.py đang truyền features
          đúng thứ tự / đúng giá trị vào model hay không.

Chạy:
  python debug_feature_mismatch.py --data_dir C:\...\bongcan_processed
                                   --img_dir  C:\...\bongcan_processed\images_gray

Kết quả: In ra so sánh giữa features từ NPZ (Noise.py) vs features
          từ Predict.py cho cùng một ảnh test.
"""

import sys, os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Import từ Predict.py ───────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from Predict_mask import (
        predict_panel, get_fan_mask, normalize,
        detect_and_split_dual_panel,
        PATCH_H, PATCH_W,
    )
    # Thử import precompute_all_patches để lấy raw X và feature_names
    from Predict_mask import precompute_all_patches
    HAS_PRECOMPUTE = True
except ImportError as e:
    HAS_PRECOMPUTE = False
    print(f"[WARN] Không import được precompute_all_patches: {e}")

from PIL import Image
import cv2


def run_diagnostic(data_dir: str, img_dir: str, n_images: int = 3):
    data_dir = os.path.abspath(data_dir)
    npz_path = os.path.join(data_dir, "dataset.npz")

    # ── 1. Load NPZ ────────────────────────────────────────────────
    print("=" * 70)
    print("  BƯỚC 1: Kiểm tra dataset.npz (features từ Noise.py)")
    print("=" * 70)

    d = np.load(npz_path, allow_pickle=True)
    npz_features = [str(s) for s in d["feature_names"]]
    X_npz   = d["X"]
    Y_npz   = d["Y"]
    stems   = d["stems"]

    print(f"NPZ feature_names ({len(npz_features)}): {npz_features}")
    print(f"X_npz.shape = {X_npz.shape}")
    print(f"Số ảnh unique: {len(np.unique(stems))}")

    # Thống kê giá trị feature
    print("\n  Thống kê giá trị features từ NPZ (mean ± std):")
    print(f"  {'Feature':<22} {'Min':>8} {'Mean':>8} {'Max':>8} {'Std':>8}")
    print(f"  {'─'*58}")
    for i, name in enumerate(npz_features):
        col = X_npz[:, i]
        print(f"  {name:<22} {col.min():>8.3f} {col.mean():>8.3f} {col.max():>8.3f} {col.std():>8.3f}")

    # ── 2. Kiểm tra Predict.py pipeline ────────────────────────────
    print("\n" + "=" * 70)
    print("  BƯỚC 2: Kiểm tra features từ Predict.py (live computation)")
    print("=" * 70)

    if not HAS_PRECOMPUTE:
        print("[ERROR] Không import được precompute_all_patches từ Predict.py")
        print("        → Predict.py cần export hàm này để kiểm tra")
        _suggest_predict_fix()
        return

    # Lấy một số ảnh test
    split_path = os.path.join(data_dir, "models", "test_split_images.json")
    import json
    with open(split_path) as f:
        test_stems = json.load(f)["test_images"][:n_images]

    IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]

    for stem in test_stems:
        # Tìm ảnh
        img_path = None
        for ext in IMG_EXTS:
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                img_path = p; break
        if img_path is None:
            print(f"  [SKIP] {stem} — không tìm thấy ảnh")
            continue

        print(f"\n  → Ảnh: {stem}")

        img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gray   = cv2.medianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), ksize=3)
        norm   = normalize(gray)
        fan, fan_raw = get_fan_mask(norm, thresh=0.05)

        # Lấy raw features từ Predict.py
        try:
            # Dùng dummy model để chặn predict, chỉ lấy features
            class FeatureCapture:
                def __init__(self): self.X_captured = None; self.called = False
                def predict_proba(self, X):
                    self.X_captured = X.copy()
                    self.called = True
                    return np.zeros((len(X), 2))

            cap = FeatureCapture()
            try:
                precompute_all_patches(norm, fan, fan_raw, cap)
            except Exception:
                pass  # Có thể raise lỗi do prob=0, nhưng X đã được capture

            if cap.called and cap.X_captured is not None:
                X_pred = cap.X_captured
                print(f"  Predict.py feature matrix: shape={X_pred.shape}")
                print(f"  NPZ feature matrix:        n_features={len(npz_features)}")

                if X_pred.shape[1] != len(npz_features):
                    print(f"\n  ⚠ MISMATCH: Predict.py truyền {X_pred.shape[1]} features")
                    print(f"             NPZ (Noise.py) có {len(npz_features)} features")
                    print(f"  → ĐÂY LÀ NGUYÊN NHÂN CHÍNH gây train-test gap!")
                else:
                    print(f"  ✅ Số features khớp: {X_pred.shape[1]}")

                    # So sánh phân phối
                    print(f"\n  So sánh phân phối features (NPZ vs Predict):")
                    # Lọc NPZ theo stem này
                    stem_mask = (np.asarray(stems, str) == stem)
                    if stem_mask.sum() == 0:
                        stem_mask_base = np.array([s.split("_")[0] == stem.split("_")[0]
                                                   for s in np.asarray(stems, str)])
                        stem_mask = stem_mask_base

                    if stem_mask.sum() > 0:
                        X_npz_stem = X_npz[stem_mask]
                        print(f"  {'Feature':<22} {'NPZ_mean':>10} {'Pred_mean':>10} {'Diff':>10} {'OK?':>5}")
                        print(f"  {'─'*60}")
                        for i, name in enumerate(npz_features):
                            if i < X_pred.shape[1]:
                                npz_m  = float(X_npz_stem[:, i].mean())
                                pred_m = float(X_pred[:, i].mean())
                                diff   = abs(npz_m - pred_m)
                                ok     = "✅" if diff < 0.15 else "⚠ DIFF"
                                print(f"  {name:<22} {npz_m:>10.4f} {pred_m:>10.4f} {diff:>10.4f} {ok:>5}")
            else:
                print("  [WARN] Không capture được features từ precompute_all_patches")
                print("         → Kiểm tra cách Predict.py gọi model.predict_proba(X)")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # ── 3. Gợi ý fix ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BƯỚC 3: Kết luận và gợi ý")
    print("=" * 70)
    _print_diagnosis(npz_features)


def _print_diagnosis(npz_features):
    print(f"""
Dựa trên phân tích:

[A] Nếu số features KHÔNG khớp:
    → Predict.py tính feature khác Noise.py
    → Cần sửa precompute_all_patches() trong Predict.py để xuất ra
      đúng {len(npz_features)} features theo thứ tự: {npz_features}

[B] Nếu số features khớp nhưng phân phối khác:
    → Có lỗi trong cách tính (normalize khác, probe detection khác)
    → So sánh detect_probe_origin() và _sample_ray_vals() giữa 2 file

[C] Nếu tất cả khớp nhưng kết quả vẫn tệ:
    → Overfitting training set
    → Cần thêm data augmentation hoặc regularization mạnh hơn

Để sửa ngay mà không cần thay đổi Predict.py:
    → Chạy lại Noise.py để tạo NPZ → train lại với đúng feature set
    → Hoặc rewrite precompute_all_patches() để match Noise.py 100%
""")


def _suggest_predict_fix():
    print("""
Để diagnostic hoạt động, cần thêm vào Predict.py:

def precompute_all_patches(norm_img, fan_mask, fan_raw, model,
                            patch_h=PATCH_H, patch_w=PATCH_W):
    # ... code hiện tại ...
    # Thêm: lưu feature_names vào result hoặc return X trước khi model.predict_proba
    pass

Hoặc: chạy script check_feature_names.py (xem bên dưới):
""")


# ── Quick check không cần Predict.py ──────────────────────────────
def quick_check_npz(data_dir: str):
    """
    Kiểm tra nhanh NPZ: in ra features và shadow ratio.
    Không cần Predict.py.
    """
    npz_path = os.path.join(data_dir, "dataset.npz")
    d = np.load(npz_path, allow_pickle=True)

    print("=" * 60)
    print("  QUICK CHECK — dataset.npz")
    print("=" * 60)

    feature_names = [str(s) for s in d["feature_names"]]
    X = d["X"]; Y = d["Y"]
    stems = d["stems"]

    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"X.shape = {X.shape}")
    print(f"Shadow ratio: {Y.mean()*100:.1f}%")

    # Phân phối shadow vs non-shadow
    print(f"\n  Phân biệt shadow vs background (mean ± std):")
    print(f"  {'Feature':<22} {'Shadow_mean':>13} {'BG_mean':>10} {'Sep':>8}")
    print(f"  {'─'*55}")

    shadow_idx = (Y == 1)
    bg_idx     = (Y == 0)
    for i, name in enumerate(feature_names):
        sh_m = X[shadow_idx, i].mean()
        bg_m = X[bg_idx, i].mean()
        sep  = abs(sh_m - bg_m)
        flag = " ★" if sep > 0.2 else ""
        print(f"  {name:<22} {sh_m:>13.4f} {bg_m:>10.4f} {sep:>8.4f}{flag}")

    # Per-image shadow stats
    print(f"\n  Per-image shadow coverage:")
    print(f"  {'Stem':<35} {'Shadow%':>8} {'n_shadow':>10} {'n_total':>10}")
    print(f"  {'─'*65}")
    for s in np.unique(stems)[:10]:
        idx = (np.asarray(stems, str) == s)
        y_s = Y[idx]
        print(f"  {s:<35} {y_s.mean()*100:>8.1f}% {y_s.sum():>10} {len(y_s):>10}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed")
    parser.add_argument("--img_dir",   default=r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed\images_gray")
    parser.add_argument("--quick",     action="store_true", help="Chỉ check NPZ, không cần Predict.py")
    parser.add_argument("--n_images",  default=3, type=int)
    args = parser.parse_args()

    if args.quick:
        quick_check_npz(args.data_dir)
    else:
        run_diagnostic(args.data_dir, args.img_dir, args.n_images)
        quick_check_npz(args.data_dir)