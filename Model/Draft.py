"""
Train.py — Stacking Classifier (HGBT + GBT + RF → LR)
=======================================================

Hỗ trợ 18 features (12 gốc + 6 Geometric).
In feature importance phân loại rõ nhóm geometric vs GLCM/Intensity.

Chạy:
    python Train.py                          # Stacking (mặc định)
    python Train.py --model rf               # RF baseline nhanh
    python Train.py --model both --cv        # Cả hai + 5-fold CV
    python Train.py --model stack --smote    # SMOTE khi ratio > 1:20
"""

import os, json, argparse, time, warnings
import numpy as np
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    os.environ.setdefault("JOBLIB_BACKEND", "threading")

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    f1_score, ConfusionMatrixDisplay,
)
import joblib

warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Nhóm geometric features để highlight trong analysis
GEO_FEATURE_NAMES = {
    "dist_origin_norm", "angle_axis_norm",
    "ray_mean_above", "lateral_drop",
    "vert_continuity", "reverb_score",
}


# ═══════════════════════════════════════════════════════════════════
# LOAD DATASET
# ═══════════════════════════════════════════════════════════════════

def load_dataset(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    Y = d["Y"].astype(np.int32)
    feature_names = [str(s) for s in d["feature_names"]]

    n_pos = int(Y.sum()); n_neg = len(Y) - n_pos
    ratio = n_neg / max(n_pos, 1)

    print(f"\n[load] {npz_path}")
    print(f"[load] X={X.shape}  n_features={len(feature_names)}")

    geo = [f for f in feature_names if f in GEO_FEATURE_NAMES]
    non_geo = [f for f in feature_names if f not in GEO_FEATURE_NAMES]
    print(f"[load] Non-geometric ({len(non_geo)}): {non_geo}")
    print(f"[load] Geometric     ({len(geo)}): {geo}")
    print(f"[load] shadow={n_pos:,} ({n_pos/len(Y)*100:.1f}%)"
          f"  bg={n_neg:,}  ratio=1:{ratio:.1f}")

    bad = np.isnan(X).sum() + np.isinf(X).sum()
    if bad > 0:
        print(f"  {bad} NaN/Inf → 0")
        X = np.nan_to_num(X, nan=0., posinf=1., neginf=-1.)

    return X, Y, feature_names, ratio


# ═══════════════════════════════════════════════════════════════════
# SMOTE
# ═══════════════════════════════════════════════════════════════════

def apply_smote(X_tr, y_tr, target_ratio=2):
    if not HAS_SMOTE:
        print("[SMOTE] Bỏ qua — pip install imbalanced-learn")
        return X_tr, y_tr
    n_sh = int(y_tr.sum()); n_bg = len(y_tr) - n_sh
    s = min(n_sh * target_ratio / max(n_bg, 1), 1.0)
    print(f"\n[SMOTE] target 1:{target_ratio}  before: sh={n_sh:,} bg={n_bg:,}")
    t0 = time.time()
    sm = SMOTE(sampling_strategy=s, k_neighbors=5, random_state=42)
    X_r, y_r = sm.fit_resample(X_tr.astype(np.float32), y_tr)
    print(f"[SMOTE] after: sh={y_r.sum():,} "
          f"bg={(~y_r.astype(bool)).sum():,}  ({time.time()-t0:.1f}s)")
    return X_r.astype(np.float64), y_r


# ═══════════════════════════════════════════════════════════════════
# THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_thresholds(y_score, y_te):
    prec, rec, thrs = precision_recall_curve(y_te, y_score)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)

    def _find(min_r):
        mask = rec[:-1] >= min_r
        if not mask.any():
            idx = int(np.argmax(rec[:-1]))
        else:
            idx = int(np.where(mask)[0][np.argmax(prec[:-1][mask])])
        return {
            "threshold": round(float(thrs[idx]), 4),
            "precision": round(float(prec[idx]), 4),
            "recall"   : round(float(rec[idx]),  4),
            "f1"       : round(float(2*prec[idx]*rec[idx]
                                      /(prec[idx]+rec[idx]+1e-9)), 4),
        }

    idx_f1 = int(np.argmax(f1))
    return {
        "best_f1": {
            "threshold": round(float(thrs[idx_f1]), 4),
            "precision": round(float(prec[idx_f1]), 4),
            "recall"   : round(float(rec[idx_f1]),  4),
            "f1"       : round(float(f1[idx_f1]),   4),
        },
        "sens80": _find(0.80),
        "sens90": _find(0.90),
    }, (prec, rec, thrs, f1)


# ═══════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def get_feature_importance(model):
    """Lấy feature importance từ RF (standalone hoặc trong Stacking)."""
    try:
        clf = model.named_steps.get("classifier", model) \
              if hasattr(model, "named_steps") else model
        if hasattr(clf, "estimators_"):
            for est in clf.estimators_:
                cand = est.steps[-1][1] if hasattr(est, "steps") else est
                if hasattr(cand, "feature_importances_"):
                    return cand.feature_importances_, cand
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_, clf
    except Exception:
        pass
    return None, None


def print_feature_importance(imp, feature_names):
    """In importance với highlight geometric features."""
    pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    print(f"\n  Feature Importance (RF)  [★ = geometric feature]:")
    print(f"  {'Feature':<22} {'Imp':>8}   Bar")
    print(f"  {'─'*55}")

    geo_total  = 0.0
    non_total  = 0.0
    for fname, imp_v in pairs:
        bar  = "█" * int(imp_v * 40)
        mark = " ★" if fname in GEO_FEATURE_NAMES else "  "
        print(f"  {fname:<22}{mark} {imp_v:>6.4f}  {bar}")
        if fname in GEO_FEATURE_NAMES:
            geo_total += imp_v
        else:
            non_total += imp_v

    print(f"\n  ──── Đóng góp theo nhóm ────")
    print(f"  Geometric features   : {geo_total:.4f}  ({geo_total*100:.1f}%)")
    print(f"  GLCM + Intensity     : {non_total:.4f}  ({non_total*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════
# EVALUATE
# ═══════════════════════════════════════════════════════════════════

def evaluate(name, model, X_te, y_te, out_dir, feature_names=None):
    print(f"\n{'═'*70}")
    print(f"  [{name}]")
    print(f"{'═'*70}")

    y_sc   = model.predict_proba(X_te)[:, 1]
    y_pred = (y_sc >= 0.5).astype(int)
    print(classification_report(y_te, y_pred,
          target_names=["bg(0)", "shadow(1)"], digits=4))

    auc = roc_auc_score(y_te, y_sc)
    ap  = average_precision_score(y_te, y_sc)
    cm  = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = (tp+tn) / len(y_te)
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    f1v  = f1_score(y_te, y_pred, zero_division=0)

    print(f"  Acc={acc:.4f}  AUC={auc:.4f}  AvgP={ap:.4f}")
    print(f"  Sens={sens:.4f}  Spec={spec:.4f}  F1={f1v:.4f}")

    thr_info, pr_data = analyze_thresholds(y_sc, y_te)
    print(f"\n  Threshold (copy --thr vào Predict.py):")
    print(f"  {'Key':<10} {'thr':>6} {'Prec':>6} {'Sens':>6} {'F1':>6} {'Acc':>6}")
    print(f"  {'─'*48}")
    results_thr = {}
    for key, t in thr_info.items():
        yp2 = (y_sc >= t["threshold"]).astype(int)
        cm2 = confusion_matrix(y_te, yp2)
        tn2,fp2,fn2,tp2 = cm2.ravel()
        acc2 = (tp2+tn2)/len(y_te)
        print(f"  {key:<10} {t['threshold']:>6.4f} {t['precision']:>6.4f} "
              f"{t['recall']:>6.4f} {t['f1']:>6.4f}  acc={acc2:.4f}")
        results_thr[key] = {**t, "accuracy": round(acc2,4),
                             "tn":int(tn2),"fp":int(fp2),"fn":int(fn2),"tp":int(tp2)}

    # Feature importance
    imp_arr, _ = get_feature_importance(model)
    if imp_arr is not None and feature_names:
        print_feature_importance(imp_arr, feature_names)

    if HAS_MPL:
        _plot_eval(name, y_sc, y_te, cm, auc, ap, acc, f1v,
                   thr_info, pr_data, out_dir, feature_names, imp_arr)

    return {
        "model"      : name,
        "accuracy"   : round(float(acc),  4),
        "roc_auc"    : round(float(auc),  4),
        "avg_prec"   : round(float(ap),   4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "f1"         : round(float(f1v),  4),
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),
        "thresholds" : results_thr,
    }


def _plot_eval(name, y_sc, y_te, cm, auc, ap, acc, f1v,
               thr_info, pr_data, out_dir, feature_names, imp_arr):
    fig = plt.figure(figsize=(26, 5))
    gs  = gridspec.GridSpec(1, 5, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # 0: Confusion matrix
    ConfusionMatrixDisplay(cm, display_labels=["bg","shadow"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"{name}\nAcc={acc:.3f} F1={f1v:.3f}", fontsize=9)

    # 1: PR curve
    p_a, r_a, _, _ = pr_data
    axes[1].plot(r_a[:-1], p_a[:-1], lw=2.5, color="#4e79a7")
    clr = {"best_f1":"red","sens80":"#f28e2b","sens90":"#59a14f"}
    for key, t in thr_info.items():
        axes[1].scatter([t["recall"]], [t["precision"]], s=100, zorder=6,
                        c=clr[key], edgecolors="k", lw=0.5,
                        label=f"{key} thr={t['threshold']} F1={t['f1']:.3f}")
    axes[1].fill_between([0.8,1],[0.8,0.8],[1,1],alpha=0.08,color="green")
    axes[1].axhline(0.8,color="gray",ls="--",lw=1,alpha=0.5)
    axes[1].axvline(0.8,color="gray",ls="--",lw=1,alpha=0.5)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR  AP={ap:.4f}", fontsize=9)
    axes[1].legend(fontsize=6); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)

    # 2: ROC
    fpr_a, tpr_a, _ = roc_curve(y_te, y_sc)
    axes[2].plot(fpr_a, tpr_a, lw=2.5, color="#59a14f")
    axes[2].plot([0,1],[0,1],"k--",lw=1)
    axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR")
    axes[2].set_title(f"ROC  AUC={auc:.4f}", fontsize=9)
    axes[2].grid(alpha=0.3)

    # 3: Feature importance (all, color by group)
    if imp_arr is not None and feature_names:
        sidx   = np.argsort(imp_arr)
        names  = [feature_names[i] for i in sidx]
        vals   = imp_arr[sidx]
        colors = ["#e15759" if n in GEO_FEATURE_NAMES else "#4e79a7"
                  for n in names]
        bars = axes[3].barh(names, vals, color=colors)
        axes[3].set_title("Feature Importance\n(red=geometric ★)", fontsize=9)
        axes[3].grid(alpha=0.2, axis="x")

        # 4: Pie chart geometric vs non-geometric
        geo_total  = sum(imp_arr[i] for i, n in enumerate(feature_names)
                         if n in GEO_FEATURE_NAMES)
        non_total  = 1.0 - geo_total
        axes[4].pie(
            [geo_total, non_total],
            labels=[f"Geometric\n{geo_total:.1%}",
                    f"GLCM+Intensity\n{non_total:.1%}"],
            colors=["#e15759", "#4e79a7"],
            autopct="%1.1f%%", startangle=90)
        axes[4].set_title("Feature Group\nContribution", fontsize=9)
    else:
        for ax in axes[3:]:
            ax.text(0.5,0.5,"N/A",ha="center",va="center",transform=ax.transAxes)

    fig.suptitle(f"Model: {name}", fontsize=11, y=1.02)
    plt.tight_layout()
    sn   = name.lower().replace(" ","_").replace("(","").replace(")","")
    path = os.path.join(out_dir, f"{sn}_eval.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  [plot] → {path}")


# ═══════════════════════════════════════════════════════════════════
# BUILD MODELS
# ═══════════════════════════════════════════════════════════════════

def build_rf(ratio):
    return RandomForestClassifier(
        n_estimators     = 600,
        max_depth        = None,
        min_samples_leaf = 1,
        max_features     = "sqrt",
        class_weight     = "balanced",
        n_jobs           = -1,
        random_state     = 42,
        oob_score        = True,
    )


def build_stacking(ratio):
    """
    Stacking: StandardScaler → [HGBT, GBT, RF] → LR meta.

    Tại sao Stacking tốt với 18 features (bao gồm geometric):
    - RF: học được tương tác phi tuyến giữa các geometric features
    - GBT: boundary sắc nét, tốt cho lateral_drop / ray_mean_above
    - HGBT: xử lý imbalance tốt, học được reverb_score
    - LR meta: kết hợp tối ưu vote của 3 model
    """
    spw = min(ratio, 15.0)

    hgbt = HistGradientBoostingClassifier(
        max_iter         = 300,
        max_leaf_nodes   = 31,
        learning_rate    = 0.05,
        min_samples_leaf = 10,
        class_weight     = "balanced",
        random_state     = 42,
    )
    gbt = GradientBoostingClassifier(
        n_estimators  = 300,
        max_depth     = 5,
        learning_rate = 0.05,
        subsample     = 0.8,
        random_state  = 42,
    )
    rf = RandomForestClassifier(
        n_estimators     = 400,
        max_depth        = None,
        min_samples_leaf = 1,
        max_features     = "sqrt",
        class_weight     = "balanced",
        n_jobs           = -1,
        random_state     = 42,
        oob_score        = True,
    )
    meta_lr = LogisticRegression(
        C            = 1.0,
        class_weight = {0: 1, 1: min(spw, 10.)},
        max_iter     = 1000,
        random_state = 42,
    )
    stacking = StackingClassifier(
        estimators      = [("hgbt",hgbt), ("gbt",gbt), ("rf",rf)],
        final_estimator = meta_lr,
        cv              = 5,
        passthrough     = False,
        n_jobs          = -1,
    )
    return Pipeline([
        ("scaler",     StandardScaler()),
        ("classifier", stacking),
    ])


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

def print_summary(all_metrics):
    print(f"\n{'═'*90}")
    print(f"  TỔNG KẾT")
    print(f"{'═'*90}")
    H = (f"  {'Model':<32} {'Acc':>7} {'AUC':>7} {'AvgP':>7} "
         f"{'Sens':>7} {'Spec':>7} {'F1':>7}")
    print(H); print(f"  {'─'*80}")
    for m in all_metrics:
        ok_s = "" if m["sensitivity"] >= 0.80 else "  "
        ok_f = "" if m["f1"]          >= 0.75 else "  "
        ok_p = "" if m["avg_prec"]    >= 0.60 else "  "
        print(f"  {m['model']:<32} {m['accuracy']:>7.4f} {m['roc_auc']:>7.4f} "
              f"{m['avg_prec']:>7.4f} {m['sensitivity']:>7.4f} "
              f"{m['specificity']:>7.4f} {m['f1']:>7.4f}  "
              f"S{ok_s} F{ok_f} P{ok_p}")

    best = max(all_metrics,
               key=lambda m:(m["sensitivity"]+m["f1"]+m["avg_prec"])/3)
    print(f"\n  ★ Best: {best['model']}")

    print(f"\n  Ngưỡng --thr cho Predict.py:")
    print(f"  {'Model':<22} {'Key':<10} {'--thr':>6}  "
          f"(sens={' ':>5} prec={' ':>5})")
    print(f"  {'─'*55}")
    for m in all_metrics:
        for key in ["best_f1", "sens80", "sens90"]:
            if key in m.get("thresholds", {}):
                t = m["thresholds"][key]
                print(f"  {m['model'][:20]:<22} {key:<10} {t['threshold']:>6.4f}  "
                      f"(sens={t['recall']:.3f} prec={t['precision']:.3f} "
                      f"f1={t['f1']:.3f})")
    print(f"{'═'*90}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run(dataset_path, model_choice="stack", test_size=0.2,
        do_cv=False, do_smote=False):

    out_dir = str(Path(dataset_path).parent / "models")
    os.makedirs(out_dir, exist_ok=True)

    X, Y, feature_names, ratio = load_dataset(dataset_path)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, Y, test_size=test_size, stratify=Y, random_state=42)
    print(f"\n[split] Train={len(X_tr):,}  Test={len(X_te):,}")
    print(f"[split] Shadow train: {y_tr.sum():,} ({y_tr.mean()*100:.1f}%)")
    print(f"[split] Shadow test : {y_te.sum():,} ({y_te.mean()*100:.1f}%)")

    if do_smote:
        X_tr, y_tr = apply_smote(X_tr, y_tr, target_ratio=2)

    all_metrics = []; fitted = {}

    # ── RF baseline ────────────────────────────────────────────────
    if model_choice in ("rf", "both"):
        print(f"\n[RF] Training RandomForest ...")
        t0 = time.time()
        rf = build_rf(ratio)
        rf.fit(X_tr, y_tr)
        print(f"[RF] OOB={rf.oob_score_:.4f}  Done {time.time()-t0:.1f}s")
        m = evaluate("RandomForest", rf, X_te, y_te, out_dir, feature_names)
        all_metrics.append(m); fitted["RF"] = rf
        joblib.dump(rf, os.path.join(out_dir, "rf_model.pkl"))
        print(f"[RF] → rf_model.pkl")

    # ── Stacking ───────────────────────────────────────────────────
    if model_choice in ("stack", "both"):
        print(f"\n[Stack] Training Stacking (HGBT+GBT+RF → LR) ...")
        print(f"[Stack] Ước tính: 5-15 phút ...")
        t0 = time.time()
        stack = build_stacking(ratio)
        stack.fit(X_tr, y_tr)
        print(f"[Stack] Done {time.time()-t0:.1f}s")
        m = evaluate("Stacking(HGBT+GBT+RF)", stack,
                     X_te, y_te, out_dir, feature_names)
        all_metrics.append(m); fitted["Stack"] = stack
        joblib.dump(stack, os.path.join(out_dir, "stacking_model.pkl"))
        print(f"[Stack] → stacking_model.pkl")

        # Lưu inner RF để Predict.py --model rf dùng được
        try:
            rf_inner = stack.named_steps["classifier"].estimators_[-1]
            if hasattr(rf_inner, "steps"):
                rf_inner = rf_inner.steps[-1][1]
            joblib.dump(rf_inner, os.path.join(out_dir, "rf_model.pkl"))
            print(f"[Stack] Inner RF → rf_model.pkl")
        except Exception:
            pass

    # ── Cross-validation ───────────────────────────────────────────
    if do_cv and fitted:
        name_cv, model_cv = list(fitted.items())[-1]
        print(f"\n[CV] {name_cv}  5-fold ...")
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        ap_sc = cross_val_score(model_cv, X_tr, y_tr, cv=skf,
                                scoring="average_precision", n_jobs=-1)
        f1_sc = cross_val_score(model_cv, X_tr, y_tr, cv=skf,
                                scoring="f1", n_jobs=-1)
        print(f"  AvgP: {ap_sc.mean():.4f} ± {ap_sc.std():.4f}")
        print(f"  F1  : {f1_sc.mean():.4f} ± {f1_sc.std():.4f}")

    print_summary(all_metrics)

    clean = [{k:v for k,v in m.items() if k!="y_score"} for m in all_metrics]
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"\n  [results] → {results_path}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train shadow detection — 18 features")
    parser.add_argument("--dataset",
        default=r"C:\Users\ThinkPad\DATN\Data\bongcan_processed\dataset.npz")
    parser.add_argument("--model", default="stack",
        choices=["rf","stack","both"])
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--cv",    action="store_true")
    parser.add_argument("--smote", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("  TRAIN — Shadow Detection (18 Features)")
    print("  Geometric: dist_origin, angle_axis, ray_mean_above,")
    print("             lateral_drop, vert_continuity, reverb_score")
    print("=" * 70)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Model     : {args.model}")
    print(f"  Test size : {args.test_size}")
    print(f"  SMOTE     : {'on' if args.smote else 'off'}")
    print(f"  CV        : {'5-fold' if args.cv else 'off'}")
    print("=" * 70 + "\n")
    print("  pip install scikit-learn imbalanced-learn\n")

    run(args.dataset, args.model, args.test_size, args.cv, args.smote)