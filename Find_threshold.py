import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET = r"C:\Users\ThinkPad\DATN1\Data\bongcan_processed\dataset.npz"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _as_feature_names(raw_names):
    return [str(name) for name in raw_names.tolist()]


def _precision_recall_f1(tp, fp, fn, beta=1.0):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta2 = beta * beta
    denom = beta2 * precision + recall
    fbeta = ((1.0 + beta2) * precision * recall / denom) if denom > 0 else 0.0
    return precision, recall, fbeta


def _score_thresholds(values, labels, thr_min, thr_max, step, beta):
    thresholds = np.arange(thr_min, thr_max + step * 0.5, step)
    rows = []

    for thr in thresholds:
        # Shadow patch is expected to be darker: mean < threshold.
        pred = values < thr
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum((~pred) & (labels == 1)))
        tn = int(np.sum((~pred) & (labels == 0)))

        precision, recall, fbeta = _precision_recall_f1(tp, fp, fn, beta)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = 0.5 * (recall + specificity)

        rows.append({
            "threshold": float(thr),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "balanced_acc": float(balanced_acc),
            "fbeta": float(fbeta),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })

    return rows, max(rows, key=lambda row: row["fbeta"])


def _percentiles(values):
    if len(values) == 0:
        return {}
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(values, qs)
    return {f"p{q}": round(float(v), 6) for q, v in zip(qs, vals)}


def _plot_distribution(shadow_vals, bg_vals, best_thr, out_png):
    plt.figure(figsize=(12, 6))
    plt.hist(
        shadow_vals,
        bins=100,
        alpha=0.72,
        density=True,
        color="#e15759",
        label="Shadow patches",
    )
    plt.hist(
        bg_vals,
        bins=100,
        alpha=0.52,
        density=True,
        color="#4e79a7",
        label="Background patches",
    )
    plt.axvline(
        best_thr,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label=f"Best SHADOW_MAX_MEAN = {best_thr:.3f}",
    )
    plt.title("Patch Mean Intensity Distribution")
    plt.xlabel("Patch mean intensity after preprocessing (0 = dark, 1 = bright)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def find_best_shadow_max_mean(
    dataset_path,
    output_dir=None,
    positive_min_coverage=0.25,
    negative_max_coverage=0.0,
    threshold_min=0.05,
    threshold_max=0.95,
    threshold_step=0.001,
    beta=1.0,
    save_plot=True,
):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")

    output_dir = Path(output_dir) if output_dir else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["Y"].astype(np.int32)
    feature_names = _as_feature_names(data["feature_names"])

    if "mean" not in feature_names:
        raise ValueError(f"dataset.npz không có feature 'mean'. Features: {feature_names}")

    mean_idx = feature_names.index("mean")
    patch_mean = np.clip(np.nan_to_num(X[:, mean_idx], nan=0.0), 0.0, 1.0)

    if "coverage" in data.files:
        coverage = data["coverage"].astype(np.float32)
        pos_mask = coverage >= positive_min_coverage
        neg_mask = coverage <= negative_max_coverage
        used_mask = pos_mask | neg_mask
        labels = pos_mask[used_mask].astype(np.int32)
        selection_rule = {
            "positive": f"coverage >= {positive_min_coverage}",
            "negative": f"coverage <= {negative_max_coverage}",
            "ignored": f"0 < coverage < {positive_min_coverage}",
        }
    else:
        used_mask = np.ones(len(y), dtype=bool)
        labels = y.astype(np.int32)
        selection_rule = {
            "positive": "Y == 1",
            "negative": "Y == 0",
            "ignored": "none",
        }

    values = patch_mean[used_mask]
    shadow_vals = values[labels == 1]
    bg_vals = values[labels == 0]

    if len(shadow_vals) == 0 or len(bg_vals) == 0:
        raise ValueError(
            "Không đủ patch shadow/background sau khi lọc coverage. "
            "Hãy giảm --positive_min_coverage hoặc kiểm tra dataset.npz."
        )

    rows, best = _score_thresholds(
        values,
        labels,
        threshold_min,
        threshold_max,
        threshold_step,
        beta,
    )

    default_050 = min(rows, key=lambda row: abs(row["threshold"] - 0.50))
    best_thr = float(best["threshold"])

    result = {
        "SHADOW_MAX_MEAN": round(best_thr, 6),
        "source_dataset": str(dataset_path),
        "feature": "mean",
        "optimization_metric": f"F{beta:g}",
        "selection_rule": selection_rule,
        "n_total_patches": int(len(y)),
        "n_used_patches": int(len(values)),
        "n_shadow_patches": int(len(shadow_vals)),
        "n_background_patches": int(len(bg_vals)),
        "shadow_mean": round(float(np.mean(shadow_vals)), 6),
        "background_mean": round(float(np.mean(bg_vals)), 6),
        "shadow_percentiles": _percentiles(shadow_vals),
        "background_percentiles": _percentiles(bg_vals),
        "best_threshold_metrics": best,
        "threshold_0_50_metrics": default_050,
    }

    json_path = output_dir / "shadow_threshold.json"
    py_path = output_dir / "shadow_threshold.py"
    curve_path = output_dir / "shadow_threshold_curve.csv"
    plot_path = output_dir / "Patch_Mean_Distribution_Histogram.png"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by Find_threshold.py\n")
        f.write("# Use the same value in both training and prediction.\n")
        f.write(f"SHADOW_MAX_MEAN = {best_thr:.6f}\n")

    with open(curve_path, "w", encoding="utf-8") as f:
        f.write("threshold,precision,recall,specificity,balanced_acc,fbeta,tp,fp,fn,tn\n")
        for row in rows:
            f.write(
                f"{row['threshold']:.6f},"
                f"{row['precision']:.6f},"
                f"{row['recall']:.6f},"
                f"{row['specificity']:.6f},"
                f"{row['balanced_acc']:.6f},"
                f"{row['fbeta']:.6f},"
                f"{row['tp']},{row['fp']},{row['fn']},{row['tn']}\n"
            )

    if save_plot:
        _plot_distribution(shadow_vals, bg_vals, best_thr, plot_path)

    print("\n" + "=" * 64)
    print("KẾT QUẢ TỐI ƯU SHADOW_MAX_MEAN THEO PATCH MEAN")
    print(f"Dataset:               {dataset_path}")
    print(f"Patch dùng để tính:    {len(values):,}/{len(y):,}")
    print(f"Shadow patches:        {len(shadow_vals):,}")
    print(f"Background patches:    {len(bg_vals):,}")
    print(f"Mean shadow:           {np.mean(shadow_vals):.4f}")
    print(f"Mean background:       {np.mean(bg_vals):.4f}")
    print(f"SHADOW_MAX_MEAN tối ưu: {best_thr:.3f}")
    print(
        f"F{beta:g}:                  {best['fbeta']:.4f} "
        f"(precision={best['precision']:.4f}, recall={best['recall']:.4f}, "
        f"specificity={best['specificity']:.4f})"
    )
    print(f"So sánh threshold 0.50: F{beta:g}={default_050['fbeta']:.4f}")
    print("-" * 64)
    print(f"Đã lưu config JSON:     {json_path}")
    print(f"Đã lưu Python module:   {py_path}")
    print(f"Đã lưu threshold curve: {curve_path}")
    if save_plot:
        print(f"Đã lưu histogram:       {plot_path}")
    print("=" * 64)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tìm SHADOW_MAX_MEAN tối ưu từ dataset.npz đã preprocessing."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--positive_min_coverage", type=float, default=0.25)
    parser.add_argument("--negative_max_coverage", type=float, default=0.0)
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    find_best_shadow_max_mean(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        positive_min_coverage=args.positive_min_coverage,
        negative_max_coverage=args.negative_max_coverage,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        beta=args.beta,
        save_plot=not args.no_plot,
    )
