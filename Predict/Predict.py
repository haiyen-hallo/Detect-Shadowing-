
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image


for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import Predict_mask as pm
except ImportError as exc:
    print(f"[ERROR] Khong import duoc Predict_mask.py: {exc}")
    sys.exit(1)


IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
}

DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "bongcan_processed"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "Data" / "bongcan_test"


def _model_candidates(model_name: str):
    model_key = (model_name or "auto").lower()
    candidates = {
        "auto": ["extratrees_model.pkl", "stacking_model.pkl", "rf_model.pkl"],
        "extratrees": ["extratrees_model.pkl"],
        "extra": ["extratrees_model.pkl"],
        "et": ["extratrees_model.pkl"],
        "stacking": ["stacking_model.pkl"],
        "stack": ["stacking_model.pkl"],
        "rf": ["rf_model.pkl"],
        "random_forest": ["rf_model.pkl"],
        "random-forest": ["rf_model.pkl"],
    }
    canonical = {
        "auto": "auto",
        "extratrees": "extratrees",
        "extra": "extratrees",
        "et": "extratrees",
        "stacking": "stacking",
        "stack": "stacking",
        "rf": "rf",
        "random_forest": "rf",
        "random-forest": "rf",
    }
    if model_key not in candidates:
        raise ValueError(f"--model khong hop le: {model_name}")
    return model_key, canonical[model_key], candidates[model_key]


def load_predict_model(models_dir: str, model_name: str):
    model_key, canonical_model, wanted_files = _model_candidates(model_name)
    raw_model = None
    loaded_model_key = canonical_model
    loaded_fname = None

    for fname in wanted_files:
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[load] Dang load model: {fname} ({size_mb:.1f} MB)...", flush=True)
        t0 = time.perf_counter()
        try:
            raw_model = joblib.load(path)
        except KeyboardInterrupt:
            print("\n[load] Ban da dung chuong trinh khi model dang load.")
            sys.exit(130)
        except Exception as exc:
            print(f"[WARN] Khong load duoc {fname}: {type(exc).__name__}: {exc}")
            if model_key != "auto":
                sys.exit(1)
            continue

        if fname.startswith("extratrees"):
            loaded_model_key = "extratrees"
        elif fname.startswith("stacking"):
            loaded_model_key = "stacking"
        elif fname.startswith("rf"):
            loaded_model_key = "rf"
        loaded_fname = fname
        print(f"[load] Model: {fname} loaded in {time.perf_counter() - t0:.1f}s")
        break

    if raw_model is None:
        print(f"[ERROR] Khong tim thay model {', '.join(wanted_files)} trong {models_dir}")
        sys.exit(1)

    return raw_model, loaded_model_key, loaded_fname


def load_dataset_feature_names(data_dir: str):
    candidates = [
        os.path.join(data_dir, "dataset.npz"),
        os.path.join(os.path.dirname(data_dir), "dataset.npz"),
    ]
    for npz_path in candidates:
        if os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            names = [str(s) for s in data["feature_names"]]
            print(f"[config] Loaded {len(names)} raw features tu: {npz_path}")
            return names

    names = [
        "contrast", "homogeneity", "correlation", "energy",
        "mean", "std", "skewness", "entropy",
        "row_ratio", "col_ratio", "local_contrast", "below_bright",
        "above_max_mean", "col_dark_ratio", "mean_drop",
    ]
    print(f"[config] Fallback raw features ({len(names)} features)")
    return names


def resolve_image_paths(image_path: str = None, image_dir: str = None,
                        recursive: bool = False):
    paths = []
    if image_path:
        image_path = os.path.abspath(image_path)
        if not os.path.isfile(image_path):
            print(f"[ERROR] Khong tim thay file anh: {image_path}")
            sys.exit(1)
        paths.append(image_path)

    if image_dir:
        image_dir = os.path.abspath(image_dir)
        if not os.path.isdir(image_dir):
            print(f"[ERROR] Khong tim thay thu muc anh: {image_dir}")
            sys.exit(1)
        iterator = Path(image_dir).rglob("*") if recursive else Path(image_dir).glob("*")
        paths.extend(str(p.resolve()) for p in iterator
                     if p.is_file() and p.suffix.lower() in IMG_EXTS)

    paths = sorted(set(paths))
    if not paths:
        print("[ERROR] Khong co anh dau vao. Dung --image_path hoac --image_dir.")
        sys.exit(1)
    return paths


def default_output_dir(image_paths, image_dir: str = None):
    if image_dir:
        return os.path.join(os.path.abspath(image_dir), "predict_results")
    parent = os.path.dirname(os.path.abspath(image_paths[0]))
    return os.path.join(parent, "predict_results")


def _draw_prediction_contours(canvas, mask, color=(40, 40, 255), thickness=3):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.drawContours(canvas, contours, -1, color, thickness, cv2.LINE_AA)
    return canvas


def make_prediction_overlay(gray: np.ndarray, pred_mask: np.ndarray,
                            patch_mask: np.ndarray, score_grid: np.ndarray,
                            stem: str, threshold: float,
                            contour_pp: bool) -> np.ndarray:
    H, W = gray.shape
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    red_layer = np.zeros_like(overlay)
    red_layer[pred_mask > 0] = (35, 35, 220)
    overlay = cv2.addWeighted(overlay, 1.0, red_layer, 0.55, 0)
    _draw_prediction_contours(overlay, pred_mask, (40, 40, 255), 3)

    patch_view = base.copy()
    patch_layer = np.zeros_like(patch_view)
    patch_layer[patch_mask > 0] = (30, 30, 180)
    patch_view = cv2.addWeighted(patch_view, 1.0, patch_layer, 0.45, 0)
    _draw_prediction_contours(patch_view, patch_mask, (40, 40, 255), 2)

    n_rows, n_cols = score_grid.shape
    for r in range(n_rows + 1):
        y = r * pm.PATCH_H
        if 0 <= y < H:
            cv2.line(patch_view, (0, y), (W - 1, y), (55, 55, 55), 1)
    for c in range(n_cols + 1):
        x = c * pm.PATCH_W
        if 0 <= x < W:
            cv2.line(patch_view, (x, 0), (x, H - 1), (55, 55, 55), 1)

    legend_w = 240
    legend = np.full((H, legend_w, 3), 25, dtype=np.uint8)

    def txt(text, y, color=(220, 220, 220), scale=0.40):
        cv2.putText(legend, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    pred_px = int(pred_mask.sum())
    patch_px = int(patch_mask.sum())
    patch_count = int((patch_mask > 0).sum() / max(pm.PATCH_H * pm.PATCH_W, 1))
    score_max = float(np.max(score_grid)) if score_grid.size else 0.0
    score_mean = float(np.mean(score_grid)) if score_grid.size else 0.0

    txt(stem[:28], 20, (230, 230, 230), 0.38)
    txt(f"Threshold : {threshold:.3f}", 46, (180, 255, 180))
    txt(f"Patch mask: {patch_count} patches", 70, (190, 190, 255))
    txt(f"Patch px  : {patch_px:,}", 94, (190, 190, 255))
    txt(f"Final px  : {pred_px:,}", 118, (190, 190, 255))
    txt(f"Score max : {score_max:.3f}", 142, (210, 210, 210))
    txt(f"Score mean: {score_mean:.3f}", 166, (210, 210, 210))
    txt(f"Contour PP: {'ON' if contour_pp else 'OFF'}", 190, (210, 210, 210))
    if H > 235:
        cv2.rectangle(legend, (10, H - 48), (28, H - 30), (35, 35, 220), -1)
        txt("Model shadow", H - 34, (190, 190, 255))

    sep = np.full((H, 4, 3), 60, dtype=np.uint8)
    return np.hstack([base, sep, patch_view, sep, overlay, sep, legend])


def make_contour_preview(gray: np.ndarray, pred_mask: np.ndarray,
                         stem: str) -> np.ndarray:
    H, _ = gray.shape
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _draw_prediction_contours(canvas, pred_mask, (40, 40, 255), 3)

    legend = np.full((H, 220, 3), 25, dtype=np.uint8)
    cv2.putText(legend, stem[:24], (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(legend, (10, 58), (48, 58), (40, 40, 255), 3)
    cv2.putText(legend, "Model contour", (58, 63), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (190, 190, 255), 1, cv2.LINE_AA)
    cv2.putText(legend, f"Pixels: {int(pred_mask.sum()):,}", (8, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)

    sep = np.full((H, 4, 3), 60, dtype=np.uint8)
    return np.hstack([canvas, sep, legend])


def predict_one_image(img_path: str, model, threshold: float,
                      apply_postprocess: bool, contour_pp: bool,
                      pp_mode: str, pp_close_iter: int,
                      pp_min_area: int, pp_min_height: int,
                      contour_min_area_px: int,
                      contour_min_height_px: int,
                      contour_max_components: int,
                      contour_close_px: int,
                      contour_open_px: int,
                      contour_smooth_eps: float,
                      contour_smooth_iter: int,
                      contour_shape_mode: str):
    stem = Path(img_path).stem
    img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    gray = cv2.medianBlur(cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY), ksize=3)
    H, W = gray.shape

    norm = pm.normalize(gray)
    res = pm.predict_grid_from_image(
        norm, model, stem, threshold, apply_postprocess,
        pp_mode=pp_mode,
        pp_close_iter=pp_close_iter,
        pp_min_area=pp_min_area,
        pp_min_height=pp_min_height,
    )

    pred_grid = (res["pred_final"] > 0).astype(np.uint8)
    patch_mask = pm.pred_grid_to_pixel(pred_grid, H, W)
    pred_mask = patch_mask
    if contour_pp:
        pred_mask = pm.doctor_contour_postprocess(
            patch_mask,
            min_area_px=contour_min_area_px,
            min_height_px=contour_min_height_px,
            max_components=contour_max_components,
            close_px=contour_close_px,
            open_px=contour_open_px,
            smooth_eps=contour_smooth_eps,
            smooth_iter=contour_smooth_iter,
            shape_mode=contour_shape_mode,
        )

    return {
        "stem": stem,
        "img_path": img_path,
        "gray": gray,
        "score_grid": res["score_grid"],
        "pred_grid": pred_grid,
        "patch_mask": patch_mask,
        "pred_mask": pred_mask.astype(np.uint8),
        "H": int(H),
        "W": int(W),
        "n_rows": int(res["n_rows"]),
        "n_cols": int(res["n_cols"]),
        "n_pred_patches": int(pred_grid.sum()),
        "pred_pixels": int(pred_mask.sum()),
    }


def run_predict(
    image_paths,
    data_dir: str,
    models_dir: str,
    out_dir: str,
    model_name: str = "auto",
    thr_final: float = None,
    tuned_defaults: bool = False,
    apply_postprocess: bool = False,
    pp_mode: str = None,
    pp_close_iter: int = None,
    pp_min_area: int = None,
    pp_min_height: int = None,
    contour_pp: bool = True,
    contour_min_area_px: int = 700,
    contour_min_height_px: int = 32,
    contour_max_components: int = 4,
    contour_close_px: int = 55,
    contour_open_px: int = 3,
    contour_smooth_eps: float = 0.004,
    contour_smooth_iter: int = 3,
    contour_shape_mode: str = "balanced",
    save_outputs: bool = True,
):
    data_dir = os.path.abspath(data_dir)
    models_dir = os.path.abspath(models_dir)
    out_dir = os.path.abspath(out_dir)
    overlay_dir = os.path.join(out_dir, "overlays")
    contour_dir = os.path.join(out_dir, "contours")
    mask_dir = os.path.join(out_dir, "masks")
    patch_mask_dir = os.path.join(out_dir, "patch_masks")

    if save_outputs:
        for folder in [overlay_dir, contour_dir, mask_dir, patch_mask_dir]:
            os.makedirs(folder, exist_ok=True)

    model, loaded_model_key, loaded_fname = load_predict_model(models_dir, model_name)
    raw_feature_names = load_dataset_feature_names(data_dir)
    cfg = pm.load_model_config(models_dir, model, loaded_model_key)
    val_thr_from_train = float(cfg.get("val_threshold", 0.55))

    tuned = pm.TUNED_MODEL_DEFAULTS.get(loaded_model_key, {})
    if tuned_defaults and tuned:
        if thr_final is None:
            thr_final = tuned["threshold"]
        if pp_mode is None:
            pp_mode = tuned["pp_mode"]
        if pp_close_iter is None:
            pp_close_iter = tuned["pp_close_iter"]
        if pp_min_area is None:
            pp_min_area = tuned["pp_min_area"]
        if pp_min_height is None:
            pp_min_height = tuned["pp_min_height"]
    elif thr_final is None:
        thr_final = val_thr_from_train

    pp_mode = pp_mode or "legacy"
    pp_min_area = 4 if pp_min_area is None else int(pp_min_area)
    pp_min_height = 1 if pp_min_height is None else int(pp_min_height)

    print("\n[predict] Pipeline: Noise.py features -> Train-derived -> model")
    print(f"          Data dir      : {data_dir}")
    print(f"          Models dir    : {models_dir}")
    print(f"          Model         : {loaded_fname}")
    print(f"          Dataset schema: {len(raw_feature_names)} features")
    print(f"          Select        : {len(pm.SELECTED_FEATURES)} features")
    print(f"          Threshold     : {thr_final} (Train saved: {val_thr_from_train})")
    print(f"          Grid PP       : {'ON' if apply_postprocess else 'OFF'}")
    print(f"          PP config     : mode={pp_mode}, close_iter={pp_close_iter}, "
          f"min_area={pp_min_area}, min_height={pp_min_height}")
    print(f"          Contour PP    : {'ON' if contour_pp else 'OFF'}")
    print("-" * 78)

    per_image = []
    n_ok = 0
    n_fail = 0

    for img_path in image_paths:
        print(f"\n  -> {Path(img_path).name}")
        try:
            item = predict_one_image(
                img_path=img_path,
                model=model,
                threshold=thr_final,
                apply_postprocess=apply_postprocess,
                contour_pp=contour_pp,
                pp_mode=pp_mode,
                pp_close_iter=pp_close_iter,
                pp_min_area=pp_min_area,
                pp_min_height=pp_min_height,
                contour_min_area_px=contour_min_area_px,
                contour_min_height_px=contour_min_height_px,
                contour_max_components=contour_max_components,
                contour_close_px=contour_close_px,
                contour_open_px=contour_open_px,
                contour_smooth_eps=contour_smooth_eps,
                contour_smooth_iter=contour_smooth_iter,
                contour_shape_mode=contour_shape_mode,
            )

            stem = item["stem"]
            mask_path = os.path.join(mask_dir, f"{stem}_mask.png")
            patch_mask_path = os.path.join(patch_mask_dir, f"{stem}_patch_mask.png")
            overlay_path = os.path.join(overlay_dir, f"{stem}_predict.png")
            contour_path = os.path.join(contour_dir, f"{stem}_contour.png")

            if save_outputs:
                Image.fromarray((item["pred_mask"] * 255).astype(np.uint8)).save(mask_path)
                Image.fromarray((item["patch_mask"] * 255).astype(np.uint8)).save(patch_mask_path)

                overlay = make_prediction_overlay(
                    item["gray"], item["pred_mask"], item["patch_mask"],
                    item["score_grid"], stem, thr_final, contour_pp,
                )
                pm.save_bgr_image(overlay_path, overlay)

                contour = make_contour_preview(item["gray"], item["pred_mask"], stem)
                pm.save_bgr_image(contour_path, contour)

            result = {
                "stem": stem,
                "img_path": img_path,
                "height": item["H"],
                "width": item["W"],
                "grid": [item["n_rows"], item["n_cols"]],
                "n_pred_patches": item["n_pred_patches"],
                "pred_pixels": item["pred_pixels"],
                "mask_path": mask_path if save_outputs else None,
                "patch_mask_path": patch_mask_path if save_outputs else None,
                "overlay_path": overlay_path if save_outputs else None,
                "contour_path": contour_path if save_outputs else None,
            }
            per_image.append(result)
            n_ok += 1
            print(f"     Grid {item['n_rows']}x{item['n_cols']}: "
                  f"{item['n_pred_patches']} patches, {item['pred_pixels']} px")

        except Exception as exc:
            print(f"     Loi: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            n_fail += 1

    output = {
        "summary": {
            "n_images": len(image_paths),
            "n_ok": n_ok,
            "n_fail": n_fail,
            "total_pred_patches": int(sum(r["n_pred_patches"] for r in per_image)),
            "total_pred_pixels": int(sum(r["pred_pixels"] for r in per_image)),
            "model": loaded_fname,
            "threshold": float(thr_final),
            "grid_postprocess": bool(apply_postprocess),
            "contour_postprocess": bool(contour_pp),
        },
        "per_image": per_image,
    }

    if save_outputs:
        with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"  DONE: OK={n_ok}, Failed={n_fail}")
    print(f"  Total patches: {output['summary']['total_pred_patches']}")
    if save_outputs:
        print(f"  Output: {out_dir}")
        print(f"  Masks : {mask_dir}")
        print(f"  Views : {overlay_dir}")
    print("=" * 78)
    return output


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict shadow mask for external images using Predict_mask.py flow."
    )
    parser.add_argument("--image_path", default=None,
                        help="Duong dan 1 anh can predict")
    parser.add_argument("--image_dir", default=None,
                        help="Thu muc anh can predict")
    parser.add_argument("--test_dir", default=None,
                        help="Alias cu cho --image_dir")
    parser.add_argument("--recursive", action="store_true",
                        help="Doc anh trong cac thu muc con cua --image_dir")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR),
                        help="Thu muc data da process, dung de doc dataset.npz va models/")
    parser.add_argument("--models_dir", default=None,
                        help="Thu muc chua model .pkl; mac dinh la data_dir/models")
    parser.add_argument("--out_dir", default=None,
                        help="Thu muc luu ket qua")
    parser.add_argument("--no_save", action="store_true",
                        help="Chi in ket qua, khong ghi anh/json")
    parser.add_argument("--model", default="auto",
                        choices=["auto", "extratrees", "extra", "et",
                                 "stacking", "stack", "rf",
                                 "random_forest", "random-forest"],
                        help="Chon model .pkl")
    parser.add_argument("--thr_final", default=None, type=float,
                        help="Threshold predict; mac dinh doc tu model_config.json")
    parser.add_argument("--tuned", action="store_true",
                        help="Dung threshold va PP da benchmark theo tung model")
    parser.add_argument("--pp", action="store_true",
                        help="Bat grid post-processing nhu Predict_mask.py")
    parser.add_argument("--no_pp", action="store_true",
                        help="Tat grid post-processing")
    parser.add_argument("--pp_mode", default=None, choices=["legacy", "vertical"])
    parser.add_argument("--pp_close_iter", default=None, type=int)
    parser.add_argument("--pp_min_area", default=None, type=int)
    parser.add_argument("--pp_min_height", default=None, type=int)
    parser.add_argument("--contour_pp", dest="contour_pp", action="store_true",
                        help="Bat contour smoothing/filtering")
    parser.add_argument("--no_contour_pp", dest="contour_pp", action="store_false",
                        help="Tat contour smoothing/filtering")
    parser.set_defaults(contour_pp=True)
    parser.add_argument("--contour_min_area_px", default=700, type=int)
    parser.add_argument("--contour_min_height_px", default=32, type=int)
    parser.add_argument("--contour_max_components", default=4, type=int)
    parser.add_argument("--contour_close_px", default=55, type=int)
    parser.add_argument("--contour_open_px", default=3, type=int)
    parser.add_argument("--contour_smooth_eps", default=0.004, type=float)
    parser.add_argument("--contour_smooth_iter", default=3, type=int)
    parser.add_argument("--contour_shape_mode", default="balanced",
                        choices=["balanced", "light", "strict"])
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    image_dir = args.image_dir or args.test_dir
    if args.image_path is None and image_dir is None and DEFAULT_IMAGE_DIR.exists():
        image_dir = str(DEFAULT_IMAGE_DIR)

    data_dir = os.path.abspath(args.data_dir)
    models_dir = os.path.abspath(args.models_dir or os.path.join(data_dir, "models"))
    image_paths = resolve_image_paths(args.image_path, image_dir, args.recursive)
    out_dir = args.out_dir or default_output_dir(image_paths, image_dir)
    apply_pp = (args.pp or args.tuned) and not args.no_pp

    print("=" * 80)
    print("  PREDICT - SHADOW DETECTION")
    print("  Flow: Predict_mask.py feature/model pipeline")
    print(f"  Images: {len(image_paths)}")
    print(f"  Model : {args.model}")
    print(f"  Save  : {'OFF' if args.no_save else 'ON'}")
    print("=" * 80 + "\n")

    run_predict(
        image_paths=image_paths,
        data_dir=data_dir,
        models_dir=models_dir,
        out_dir=out_dir,
        model_name=args.model,
        thr_final=args.thr_final,
        tuned_defaults=args.tuned,
        apply_postprocess=apply_pp,
        pp_mode=args.pp_mode,
        pp_close_iter=args.pp_close_iter,
        pp_min_area=args.pp_min_area,
        pp_min_height=args.pp_min_height,
        contour_pp=args.contour_pp,
        contour_min_area_px=args.contour_min_area_px,
        contour_min_height_px=args.contour_min_height_px,
        contour_max_components=args.contour_max_components,
        contour_close_px=args.contour_close_px,
        contour_open_px=args.contour_open_px,
        contour_smooth_eps=args.contour_smooth_eps,
        contour_smooth_iter=args.contour_smooth_iter,
        contour_shape_mode=args.contour_shape_mode,
        save_outputs=not args.no_save,
    )
