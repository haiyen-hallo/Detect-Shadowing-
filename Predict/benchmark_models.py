import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from Predict_mask import (  # noqa: E402
    GLCM_BINS,
    PATCH_H,
    PATCH_W,
    RAW_FEATURE_NAMES,
    compute_metrics,
    compute_overlap,
    compute_patch_features,
    find_file,
    load_mask_binary,
    load_model_config,
    normalize,
    pixel_to_patch_grid,
    postprocess_patch_grid,
    prepare_features_for_model,
    TUNED_MODEL_DEFAULTS,
)


MODEL_FILES = {
    "extratrees": "extratrees_model.pkl",
    "stacking": "stacking_model.pkl",
    "rf": "rf_model.pkl",
}


def _find_data_dirs(data_dir):
    img_dir = next(
        (
            os.path.join(data_dir, d)
            for d in ["images_gray", "images", "imgs", "bongcan"]
            if os.path.isdir(os.path.join(data_dir, d))
        ),
        None,
    )
    mask_dir = next(
        (
            os.path.join(data_dir, d)
            for d in ["masks", "mask", "labels", "annotations", "ground_truth"]
            if os.path.isdir(os.path.join(data_dir, d))
        ),
        None,
    )
    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Missing image/mask folder in {data_dir}")
    return img_dir, mask_dir


def _patch_gt_stats(mask, n_rows, n_cols):
    gt_counts = np.zeros((n_rows, n_cols), dtype=np.int64)
    areas = np.zeros((n_rows, n_cols), dtype=np.int64)
    covered = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    for r in range(n_rows):
        for c in range(n_cols):
            y0 = r * PATCH_H
            y1 = min(y0 + PATCH_H, h)
            x0 = c * PATCH_W
            x1 = min(x0 + PATCH_W, w)
            patch = mask[y0:y1, x0:x1]
            gt_counts[r, c] = int(patch.sum())
            areas[r, c] = int(patch.size)
            covered[y0:y1, x0:x1] = True
    border = ~covered
    border_gt = int(mask[border].sum())
    border_area = int(border.sum())
    return gt_counts, areas, border_gt, border_area


def _pixel_counts_from_grid(pred_grid, item):
    pred = pred_grid.astype(bool)
    gt_counts = item["gt_counts"]
    areas = item["areas"]
    tp = int(gt_counts[pred].sum())
    fp = int((areas[pred] - gt_counts[pred]).sum())
    fn = int(gt_counts[~pred].sum()) + item["border_gt"]
    tn = int((areas[~pred] - gt_counts[~pred]).sum()) + (item["border_area"] - item["border_gt"])
    return tp, fp, fn, tn


def _metrics_from_counts(tp, fp, fn, tn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _image_overlap_from_counts(tp, fp, fn, item):
    gt_pixels = item["gt_pixels"]
    pred_pixels = tp + fp
    union = tp + fp + fn
    return {
        "iou": round(tp / max(union, 1), 4),
        "dice": round((2 * tp) / max(gt_pixels + pred_pixels, 1), 4),
        "sensitivity": round(tp / max(gt_pixels, 1), 4),
        "precision": round(tp / max(pred_pixels, 1), 4),
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
    }


def load_test_items(data_dir, mask_patch_thr):
    data_dir = os.path.abspath(data_dir)
    models_dir = os.path.join(data_dir, "models")
    split_path = os.path.join(models_dir, "test_split_images.json")
    with open(split_path, encoding="utf-8") as f:
        stems = json.load(f)["test_images"]

    img_dir, mask_dir = _find_data_dirs(data_dir)
    items = []
    for stem in stems:
        img_path = find_file(img_dir, stem)
        mask_path = find_file(mask_dir, stem)
        if img_path is None or mask_path is None:
            continue

        img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        gray = cv2.medianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), ksize=3)
        h, w = gray.shape
        gt_pixel = load_mask_binary(mask_path, h, w)
        n_rows = h // PATCH_H
        n_cols = w // PATCH_W
        gt_patch = pixel_to_patch_grid(gt_pixel, n_rows, n_cols, h, w, mask_patch_thr)
        gt_counts, areas, border_gt, border_area = _patch_gt_stats(gt_pixel, n_rows, n_cols)

        items.append(
            {
                "stem": stem,
                "norm": normalize(gray),
                "gt_patch": gt_patch,
                "gt_counts": gt_counts,
                "areas": areas,
                "border_gt": border_gt,
                "border_area": border_area,
                "gt_pixels": int(gt_pixel.sum()),
                "h": h,
                "w": w,
                "n_rows": n_rows,
                "n_cols": n_cols,
            }
        )
    return items


def score_model(data_dir, model_name, items):
    models_dir = os.path.join(os.path.abspath(data_dir), "models")
    model_path = os.path.join(models_dir, MODEL_FILES[model_name])
    model = joblib.load(model_path)
    load_model_config(models_dir, model)

    scored = []
    for item in items:
        empty_mask = np.zeros(item["norm"].shape, dtype=np.uint8)
        with contextlib.redirect_stdout(io.StringIO()):
            feats = compute_patch_features(item["norm"], empty_mask, PATCH_H, PATCH_W, GLCM_BINS)
        x_raw = feats["patch_features"]
        x_sel = prepare_features_for_model(x_raw, RAW_FEATURE_NAMES, model, item["stem"])
        scores = model.predict_proba(x_sel)[:, 1].reshape(item["n_rows"], item["n_cols"])
        scored.append({**item, "scores": scores})
    return scored


def evaluate_scored(scored, threshold, postprocess=False, model_name=None):
    pixel_total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    patch_gt = []
    patch_pred = []
    per_image = []

    for item in scored:
        pred_grid = (item["scores"] >= threshold).astype(np.uint8)
        if postprocess:
            tuned = TUNED_MODEL_DEFAULTS.get(model_name or "", {})
            pred_grid = postprocess_patch_grid(
                pred_grid,
                min_patch_area=tuned.get("pp_min_area", 4),
                mode=tuned.get("pp_mode", "legacy"),
                close_iter=tuned.get("pp_close_iter"),
                min_patch_height=tuned.get("pp_min_height", 1),
            )

        tp, fp, fn, tn = _pixel_counts_from_grid(pred_grid, item)
        pixel_total["tp"] += tp
        pixel_total["fp"] += fp
        pixel_total["fn"] += fn
        pixel_total["tn"] += tn

        patch_gt.extend(item["gt_patch"].ravel().tolist())
        patch_pred.extend(pred_grid.ravel().tolist())
        ov = _image_overlap_from_counts(tp, fp, fn, item)
        ov["stem"] = item["stem"]
        ov["pred_patches"] = int(pred_grid.sum())
        ov["gt_patch_pixels"] = int(item["gt_patch"].sum())
        per_image.append(ov)

    pixel_metrics = _metrics_from_counts(**pixel_total)
    patch_metrics = compute_metrics(np.asarray(patch_gt, dtype=np.uint8), np.asarray(patch_pred, dtype=np.uint8))
    mean_overlap = {
        key: round(float(np.mean([r[key] for r in per_image])), 4)
        for key in ["iou", "dice", "sensitivity", "precision"]
    }
    return {
        "threshold": threshold,
        "postprocess": postprocess,
        "pixel_metrics": pixel_metrics,
        "patch_metrics": patch_metrics,
        "mean_overlap": mean_overlap,
        "per_image": per_image,
    }


def sweep_thresholds(scored, thresholds, postprocess=False, model_name=None):
    rows = [evaluate_scored(scored, float(thr), postprocess, model_name) for thr in thresholds]
    return max(rows, key=lambda r: r["pixel_metrics"]["iou"]), rows


def fmt_row(values, widths):
    return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))


def print_summary(results):
    widths = [10, 5, 4, 7, 7, 7, 7, 9, 9, 9]
    print()
    print(fmt_row(["model", "mode", "thr", "px_iou", "px_dice", "px_prec", "px_rec", "patch_iou", "mean_iou", "pred/gt"], widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for model_name, model_results in results.items():
        for label, res in model_results.items():
            px = res["pixel_metrics"]
            pa = res["patch_metrics"]
            mo = res["mean_overlap"]
            gt = sum(r["gt_pixels"] for r in res["per_image"])
            pred = sum(r["pred_pixels"] for r in res["per_image"])
            ratio = round(pred / max(gt, 1), 2)
            print(
                fmt_row(
                    [
                        model_name,
                        label,
                        f"{res['threshold']:.2f}",
                        f"{px['iou']:.4f}",
                        f"{px['dice']:.4f}",
                        f"{px['precision']:.4f}",
                        f"{px['recall']:.4f}",
                        f"{pa['iou']:.4f}",
                        f"{mo['iou']:.4f}",
                        ratio,
                    ],
                    widths,
                )
            )


def print_worst_images(results, label="base", n=8):
    print()
    print(f"Worst images at {label}:")
    widths = [10, 36, 7, 7, 7, 9, 9]
    print(fmt_row(["model", "stem", "iou", "prec", "rec", "pred_px", "gt_px"], widths))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for model_name, model_results in results.items():
        rows = sorted(model_results[label]["per_image"], key=lambda r: r["iou"])[:n]
        for r in rows:
            print(
                fmt_row(
                    [
                        model_name,
                        r["stem"][:36],
                        f"{r['iou']:.4f}",
                        f"{r['precision']:.4f}",
                        f"{r['sensitivity']:.4f}",
                        r["pred_pixels"],
                        r["gt_pixels"],
                    ],
                    widths,
                )
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=r"C:\Users\ThinkPad\Graduation_project\Data\bongcan_processed")
    parser.add_argument("--mask_patch_thr", default=0.30, type=float)
    parser.add_argument("--base_thr", default=0.55, type=float)
    parser.add_argument("--thr_min", default=0.05, type=float)
    parser.add_argument("--thr_max", default=0.95, type=float)
    parser.add_argument("--thr_step", default=0.01, type=float)
    args = parser.parse_args()

    thresholds = np.round(
        np.arange(args.thr_min, args.thr_max + args.thr_step / 2, args.thr_step),
        4,
    )
    print(f"Loading test set from {args.data_dir}")
    items = load_test_items(args.data_dir, args.mask_patch_thr)
    print(f"Images: {len(items)} | thresholds: {len(thresholds)}")

    results = {}
    for model_name in ["extratrees", "stacking", "rf"]:
        model_path = os.path.join(os.path.abspath(args.data_dir), "models", MODEL_FILES[model_name])
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: missing {model_path}")
            continue
        print(f"\nScoring {model_name}...")
        scored = score_model(args.data_dir, model_name, items)
        tuned_thr = TUNED_MODEL_DEFAULTS.get(model_name, {}).get("threshold", args.base_thr)
        base = evaluate_scored(scored, args.base_thr, postprocess=False, model_name=model_name)
        tuned = evaluate_scored(scored, tuned_thr, postprocess=True, model_name=model_name)
        best_raw, _ = sweep_thresholds(scored, thresholds, postprocess=False, model_name=model_name)
        best_pp, _ = sweep_thresholds(scored, thresholds, postprocess=True, model_name=model_name)
        results[model_name] = {
            "base": base,
            "tuned": tuned,
            "best": best_raw,
            "best_pp": best_pp,
        }

    print_summary(results)
    print_worst_images(results, "base", n=8)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
