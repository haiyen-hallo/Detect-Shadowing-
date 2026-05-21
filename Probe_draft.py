import numpy as np
import matplotlib.pyplot as plt


def plot_virtual_probe_origin(image, fan_mask, cx, cy, R=None, title="Virtual probe origin"):
    """
    Vẽ ảnh siêu âm + fan_mask + tâm đầu dò ảo nằm phía trên ảnh.

    image: ảnh gốc, shape (H, W)
    fan_mask: mask vùng fan, shape (H, W)
    cx, cy: tọa độ tâm đầu dò ảo theo hệ tọa độ ảnh gốc
            cy có thể âm, ví dụ cy = -120
    R: bán kính từ tâm ảo đến bề mặt đầu dò, nếu có thì vẽ thêm cung tròn
    """

    H, W = image.shape[:2]

    # Nếu cy âm, cần thêm khoảng trống phía trên để nhìn thấy tâm ảo
    pad_top = max(0, int(abs(cy)) + 30)

    # Dịch toàn bộ hệ tọa độ xuống dưới để vẽ được cy âm
    cy_plot = cy + pad_top
    cx_plot = cx

    # Tạo canvas mới có thêm vùng phía trên
    if image.ndim == 2:
        canvas = np.zeros((H + pad_top, W), dtype=image.dtype)
        canvas[pad_top:pad_top + H, :] = image
    else:
        canvas = np.zeros((H + pad_top, W, image.shape[2]), dtype=image.dtype)
        canvas[pad_top:pad_top + H, :, :] = image

    # Dịch fan_mask xuống dưới
    mask_canvas = np.zeros((H + pad_top, W), dtype=fan_mask.dtype)
    mask_canvas[pad_top:pad_top + H, :] = fan_mask

    plt.figure(figsize=(7, 8))

    # Vẽ ảnh gốc
    plt.imshow(canvas, cmap="gray")

    # Overlay fan_mask cho dễ nhìn
    plt.imshow(mask_canvas, cmap="jet", alpha=0.25)

    # Vẽ tâm đầu dò ảo
    plt.scatter(cx_plot, cy_plot, c="red", s=100, marker="x", label=f"Virtual origin ({cx}, {cy})")

    # Vẽ đường từ tâm ảo tới vùng fan
    ys, xs = np.where(fan_mask > 0)
    if len(ys) > 0:
        y_min = int(ys.min())
        top_xs = xs[ys <= y_min + 10]

        if len(top_xs) > 0:
            x_left = int(np.min(top_xs))
            x_right = int(np.max(top_xs))
            y_top = y_min

            # Dịch y_top khi plot
            y_top_plot = y_top + pad_top

            plt.scatter([x_left, x_right], [y_top_plot, y_top_plot],
                        c="yellow", s=40, label="Top fan boundary")

            plt.plot([cx_plot, x_left], [cy_plot, y_top_plot], "r--", linewidth=1)
            plt.plot([cx_plot, x_right], [cy_plot, y_top_plot], "r--", linewidth=1)

    # Nếu có bán kính R thì vẽ cung/bề mặt đầu dò
    if R is not None:
        theta = np.linspace(0, 2 * np.pi, 500)
        arc_x = cx + R * np.sin(theta)
        arc_y = cy + R * np.cos(theta)

        arc_y_plot = arc_y + pad_top

        valid = (
            (arc_x >= 0) & (arc_x < W) &
            (arc_y_plot >= 0) & (arc_y_plot < H + pad_top)
        )

        plt.plot(arc_x[valid], arc_y_plot[valid], color="cyan", linewidth=1.5, label="Probe surface / arc")

    # Đường y=0 của ảnh gốc
    plt.axhline(pad_top, color="white", linestyle="--", linewidth=1, label="Original image top y=0")

    plt.title(title)
    plt.legend(loc="upper right")
    plt.axis("equal")
    plt.axis("off")
    plt.show()