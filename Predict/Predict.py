"""
Predict.py — Shadow Detection (Fan Beam chuẩn + 18 Features)
============================================================

MÔ HÌNH VẬT LÝ CHUẨN (hình 4):
  ★ Tâm hội tụ ảo (cx, cy) nằm PHÍA TRÊN bề mặt đầu dò
    → cy có thể âm (nằm ngoài ảnh), KHÔNG clip về 0
  ● Bề mặt đầu dò = arc tròn bán kính R từ (cx, cy)
  → 256 tia đều đi TỪ (cx, cy), bắt đầu TẠI bề mặt (r = R)

3 LỖI ĐÃ FIX so với version trước:
  1. cy KHÔNG clip về 0 → tâm ảo đúng vị trí phía trên ảnh
  2. r_arr bắt đầu từ R (bề mặt đầu dò), không phải PATCH_H*0.5
  3. fill_corridors: get_pts dùng rr trực tiếp (không + thêm R)

Chạy:
    python Predict.py --test_dir "..." --models_dir "..."
"""

import os, sys, json, glob, argparse, traceback, warnings
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import joblib

warnings.filterwarnings("ignore")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

PATCH_H   = 16
PATCH_W   = 16
GLCM_BINS = 32

N_THETA     = 256
N_R_SAMPLES = 200

CLEAN_MAX_STD  = 0.12
DIRTY_MAX_STD  = 0.28
EDGE_MAX_STD   = 0.20

BRIGHT_WIN      = 5
DARK_WIN        = 12
BRIGHT_MEAN_THR = 0.22
DARK_MEAN_THR   = 0.45
DROP_MIN        = 0.04
SMOOTH_THR      = 0.30

RECOVERY_THR     = 0.70
MAX_RECOVERY_PCT = 0.35

TE_LOW_PCT = 45

W_RF           = 0.35
W_PHYS         = 0.65
THR_FINAL      = 0.38
RF_MIN_PROB    = 0.25
MAX_SHADOW_PCT = 0.15

FEATURE_NAMES = [
    "contrast", "homogeneity", "correlation", "energy",
    "mean", "std", "skewness", "entropy",
    "row_ratio", "col_ratio", "local_contrast", "below_bright",
    "dist_origin_norm", "angle_axis_norm", "ray_mean_above",
    "lateral_drop", "vert_continuity", "reverb_score",
]

SHADOW_MAX_MEAN = 0.5

DERIVED_FEATURE_NAMES = [
    "above_below_ratio",
    "shadow_drop",
    "dark_bright_above",
    "lateral_contrast_nm",
    "depth_darkness",
    "vert_dark_depth",
    "anti_reverb_dark",
    "phys_shadow_score",
]


def engineer_probe_features(X, feature_names):
    """Add the same 8 probe-derived features used in Model/Train.py."""
    fn = {name: i for i, name in enumerate(feature_names)}
    eps = 1e-6

    def col(name):
        return X[:, fn[name]].astype(np.float64) if name in fn else np.zeros(len(X))

    mean      = col("mean")
    ray_above = col("ray_mean_above")
    lat_drop  = col("lateral_drop")
    vert_cont = col("vert_continuity")
    reverb    = col("reverb_score")
    dist_norm = col("dist_origin_norm")

    above_below = np.clip(ray_above / (mean + eps), 0.0, 10.0)
    shadow_drop = np.clip(ray_above - mean, 0.0, 1.0)
    darkness = np.clip(SHADOW_MAX_MEAN - mean, 0.0, SHADOW_MAX_MEAN) / SHADOW_MAX_MEAN
    dark_bright = darkness * np.clip(ray_above, 0.0, 1.0)
    lat_norm = np.clip(lat_drop / (mean + eps), -2.0, 5.0)
    depth_dark = dist_norm * np.clip(1.0 - mean, 0.0, 1.0)
    vert_depth = vert_cont * dist_norm
    anti_reverb = np.clip(1.0 - reverb, 0.0, 1.0) * np.clip(1.0 - mean, 0.0, 1.0)
    phys = np.clip(above_below / 5.0, 0.0, 1.0) * shadow_drop * darkness

    new_cols = np.column_stack([
        above_below, shadow_drop, dark_bright, lat_norm,
        depth_dark, vert_depth, anti_reverb, np.clip(phys, 0.0, 1.0),
    ]).astype(np.float32)
    new_cols = np.nan_to_num(new_cols, nan=0., posinf=1., neginf=-1.)
    return np.hstack([X, new_cols]), feature_names + DERIVED_FEATURE_NAMES


def expected_n_features(model):
    """Return fitted sklearn feature count for direct estimators/pipelines."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            n = expected_n_features(step)
            if n is not None:
                return n
    if hasattr(model, "named_estimators_"):
        for est in model.named_estimators_.values():
            n = expected_n_features(est)
            if n is not None:
                return n
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            n = expected_n_features(est)
            if n is not None:
                return n
    return None


def load_model(models_dir):
    for fname in ["stacking_model.pkl", "rf_model.pkl"]:
        p = os.path.join(models_dir, fname)
        if os.path.exists(p):
            m = joblib.load(p)
            if hasattr(m, "predict_proba"):
                print(f"[load] {fname}")
                return m, fname
    print(f"Không tìm thấy model: {models_dir}")
    sys.exit(1)


def normalize(gray):
    img = gray.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img-lo)/(hi-lo) if hi-lo > 1e-6 else np.zeros_like(img)


def detect_and_split_dual_panel(gray_img):
    h, w = gray_img.shape[:2]
    if w < h * 1.2:
        return [(gray_img, 0)]
    center = w//2; sr = int(w*0.15)
    x0 = max(0, center-sr); x1 = min(w, center+sr)
    col_means = gray_img[:, x0:x1].mean(axis=0)
    min_mean  = float(np.min(col_means))
    if min_mean < 20:
        sg = x0 + int(np.argmin(col_means))
        print(f"      [dual panel] split x={sg}  min={min_mean:.1f}")
        return [(gray_img[:,:sg], 0), (gray_img[:,sg:], sg)]
    return [(gray_img, 0)]


def get_fan_mask(norm_img, thresh=0.05):
    bw = ((norm_img > thresh)*255).astype(np.uint8)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  k)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(raw, [max(cnts,key=cv2.contourArea)], -1, 255, -1)
    ek     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PATCH_H,PATCH_H))
    eroded = cv2.erode(raw, ek, iterations=1)
    return eroded, raw


# ═══════════════════════════════════════════════════════════════════
# FAN BEAM MODEL
# ═══════════════════════════════════════════════════════════════════

def fit_probe_circle(fan_mask_raw):
    """
    Circle fit lên top arc của fan_raw.

    FIX 1: cy CÓ THỂ ÂM — không clip về 0.
    Tâm hội tụ ảo nằm PHÍA TRÊN ảnh (cy < 0) là hoàn toàn đúng vật lý.
    """
    H, W   = fan_mask_raw.shape
    ys, xs = np.where(fan_mask_raw > 0)

    def _fallback():
        counts   = np.bincount(ys,minlength=H) if len(ys)>0 else np.zeros(H)
        vrows    = np.where(counts>=8)[0]
        y_top    = int(vrows[0]) if len(vrows) else 0
        cx_f     = W/2.0
        fan_w_f  = float(np.percentile(xs,98)-np.percentile(xs,2)) \
                   if len(xs)>0 else W*0.8
        cy_f     = y_top - fan_w_f*0.25    # phía trên arc
        R_f      = fan_w_f*0.55
        return cx_f, cy_f, R_f, -np.radians(65), np.radians(65)

    if len(ys) < 50:
        return _fallback()

    top_pts = []
    for col in range(W):
        col_rows = ys[xs == col]
        if len(col_rows) > 0:
            top_pts.append((float(col), float(col_rows.min())))
    if len(top_pts) < 10:
        return _fallback()

    raw_x = np.array([p[0] for p in top_pts])
    raw_y = np.array([p[1] for p in top_pts])
    k_sm  = max(1, len(raw_y)//25)
    raw_y = np.convolve(raw_y, np.ones(k_sm)/k_sm, mode='same')

    n = len(raw_x); m = max(2, int(n*0.08))
    px = raw_x[m:n-m]; py = raw_y[m:n-m]
    if len(px) < 6:
        return _fallback()

    # Kasa circle fit
    A = np.column_stack([2*px, 2*py, np.ones(len(px))])
    b = px**2 + py**2
    try:
        res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = float(res[0])
        cy = float(res[1])
        R  = float(np.sqrt(max(res[2]+cx**2+cy**2, 1.0)))

        cx = float(np.clip(cx, W*0.05, W*0.95))

        # FIX 1: Không clip cy về 0!
        # Tâm ảo PHẢI NẰM TRÊN bề mặt (cy ≤ min(py))
        cy_min_arc = float(py.min())
        fan_width  = float(px.max()-px.min())
        if cy > cy_min_arc:
            # Arc quá phẳng → ước lượng thủ công
            cy = cy_min_arc - fan_width*0.25
            R  = float(np.sqrt(((px-cx)**2+(py-cy)**2).mean()))

        if R < PATCH_H or R > max(H,W)*5:
            return _fallback()

    except Exception:
        return _fallback()

    phi_left  = float(np.arctan2(px[0]-cx,  py[0]-cy))
    phi_right = float(np.arctan2(px[-1]-cx, py[-1]-cy))
    if phi_left > phi_right:
        phi_left, phi_right = phi_right, phi_left

    theta1 = np.radians(90.0) - abs(phi_left)
    scan   = phi_right - phi_left
    print(f"      [probe] cx={cx:.1f} cy={cy:.1f} R={R:.1f}  "
          f"θ₁={np.degrees(theta1):.1f}°  scan={np.degrees(scan):.1f}°")

    return cx, cy, R, phi_left, phi_right


def build_ray_maps(fan_mask, cx, cy, R, phi_left, phi_right,
                    n_rays=N_THETA, n_r=N_R_SAMPLES):
    """
    FIX 2: r_arr bắt đầu từ R (bề mặt đầu dò), không phải từ 0.
    map_x[i,j] = cx + r_arr[j]*sin(phi_i)   với r_arr[0] = R
    """
    H, W = fan_mask.shape
    phis    = np.linspace(phi_left, phi_right, n_rays)
    sin_phi = np.sin(phis); cos_phi = np.cos(phis)

    ys_fan, xs_fan = np.where(fan_mask > 0)
    if len(ys_fan) == 0:
        return None, None, None, None

    dists_fan = np.sqrt((xs_fan-cx)**2+(ys_fan-cy)**2)
    r_max     = float(dists_fan.max())*1.05

    # FIX 2: bắt đầu từ R
    r_arr = np.linspace(R, r_max, n_r)

    map_x = (cx + r_arr[None,:]*sin_phi[:,None]).astype(np.float32)
    map_y = (cy + r_arr[None,:]*cos_phi[:,None]).astype(np.float32)

    xi = np.clip(map_x.astype(np.int32), 0, W-1)
    yi = np.clip(map_y.astype(np.int32), 0, H-1)
    in_bounds = (map_x>=0)&(map_x<W)&(map_y>=0)&(map_y<H)
    in_fan_v  = fan_mask[yi,xi] > 0
    valid     = in_bounds & in_fan_v

    return map_x, map_y, r_arr, valid


def _build_glcm(pq, bins):
    g = np.zeros((bins,bins), dtype=np.float32)
    ph,pw = pq.shape
    for dy,dx in [(0,1),(1,1),(1,0),(1,-1)]:
        if   dy==0: rp,cp=pq[:,:pw-1],pq[:,1:]
        elif dx==0: rp,cp=pq[:ph-1,:],pq[1:,:]
        elif dx>0:  rp,cp=pq[:ph-1,:pw-1],pq[1:,1:]
        else:       rp,cp=pq[:ph-1,1:],pq[1:,:pw-1]
        rv,cv_=rp.ravel(),cp.ravel()
        np.add.at(g,(rv,cv_),1); np.add.at(g,(cv_,rv),1)
    s=float(g.sum())
    if s>0: g/=s
    return g


def _sample_ray(norm_img, ox, oy, px, py, t0, t1, n=20):
    H,W=norm_img.shape; dx=px-ox; dy=py-oy; vals=[]
    for k in range(n):
        t=t0+(t1-t0)*k/max(n-1,1)
        sx=int(round(ox+dx*t)); sy=int(round(oy+dy*t))
        if 0<=sx<W and 0<=sy<H: vals.append(float(norm_img[sy,sx]))
    return np.array(vals,dtype=np.float32) if vals else np.zeros(1)


def precompute_all_patches(norm_img, fan_mask, cx, cy, R, model):
    H,W=norm_img.shape; ph,pw=PATCH_H,PATCH_W; bins=GLCM_BINS
    n_rows=H//ph; n_cols=W//pw
    if n_rows==0 or n_cols==0:
        return (np.zeros((1,1),np.float32),np.zeros((1,1),np.float32),
                np.zeros((1,1),bool),1,1)

    q=np.clip((norm_img*(bins-1)).astype(np.int32),0,bins-1)
    ib=np.arange(bins,dtype=np.float32)
    II,JJ=np.meshgrid(ib,ib,indexing="ij")
    dq=(II-JJ)**2; dh=1.0+dq
    ks=max(3,(ph*3)|1)
    nbr=cv2.GaussianBlur(norm_img,(ks,ks),sigmaX=ks/3.0)
    brt=(norm_img>0.70).astype(np.float32)

    ox,oy=float(cx),float(cy)
    ys_fan,xs_fan=np.where(fan_mask>0)
    if len(xs_fan)>0:
        fan_diag=float(np.sqrt(((xs_fan-ox)**2+(ys_fan-oy)**2).max()))
        fan_cx=float(xs_fan.mean())
        fan_half_w=max(float(xs_fan.max()-xs_fan.min())/2.0,1.0)
    else:
        fan_diag=float(max(H,W)); fan_cx=W/2.0; fan_half_w=W/2.0

    f={k:np.zeros((n_rows,n_cols),dtype=np.float64) for k in FEATURE_NAMES}
    pm=np.zeros((n_rows,n_cols),dtype=np.float32)
    inf=np.zeros((n_rows,n_cols),dtype=bool)

    for r in range(n_rows):
        for c in range(n_cols):
            y0,y1=r*ph,(r+1)*ph; x0,x1=c*pw,(c+1)*pw
            reg=fan_mask[y0:y1,x0:x1]
            if float(reg.sum())/(255.0*reg.size)<0.60: continue
            inf[r,c]=True
            pf=norm_img[y0:y1,x0:x1]; pq=q[y0:y1,x0:x1]

            g=_build_glcm(pq,bins)
            f["contrast"][r,c]=float((g*dq).sum())
            f["homogeneity"][r,c]=float((g/dh).sum())
            mx=float((II*g).sum()); my=float((JJ*g).sum())
            sx=float(np.sqrt(max(float((g*(II-mx)**2).sum()),1e-12)))
            sy=float(np.sqrt(max(float((g*(JJ-my)**2).sum()),1e-12)))
            f["correlation"][r,c]=float(((II-mx)*(JJ-my)*g).sum()/(sx*sy))
            f["energy"][r,c]=float((g**2).sum())
            pnz=g[g>0]
            f["entropy"][r,c]=float(-np.sum(pnz*np.log2(pnz+1e-12)))

            vals=pf.ravel().astype(np.float64)
            m_val=float(vals.mean()); s_val=float(vals.std())
            f["mean"][r,c]=m_val; f["std"][r,c]=s_val
            if s_val>1e-6:
                f["skewness"][r,c]=float(
                    np.clip(np.mean((vals-m_val)**3)/(s_val**3),-10,10))

            f["row_ratio"][r,c]=r/max(n_rows-1,1)
            f["col_ratio"][r,c]=c/max(n_cols-1,1)
            nbr_m=float(nbr[y0:y1,x0:x1].mean())
            f["local_contrast"][r,c]=m_val/(nbr_m+1e-6)
            ab=brt[:r*ph,x0:x1] if r>0 else np.zeros((1,1))
            f["below_bright"][r,c]=float(ab.mean()) if ab.size else 0.0
            pm[r,c]=m_val

            py_c=(r+0.5)*ph; px_c=(c+0.5)*pw
            dx_p=px_c-ox; dy_p=py_c-oy
            dist=float(np.sqrt(dx_p**2+dy_p**2))
            f["dist_origin_norm"][r,c]=dist/(fan_diag+1e-6)
            f["angle_axis_norm"][r,c]=(px_c-fan_cx)/(fan_half_w+1e-6)

            if dist>R+ph:
                t_surf=R/max(dist,1e-6)
                above_v=_sample_ray(norm_img,ox,oy,px_c,py_c,t_surf+0.05,0.90,n=20)
                f["ray_mean_above"][r,c]=float(above_v.mean())

                nbr_lat=[]
                for dc in [-2,-1,1,2]:
                    nc=c+dc
                    if 0<=nc<n_cols:
                        nbr_lat.append(float(norm_img[y0:y1,nc*pw:(nc+1)*pw].mean()))
                f["lateral_drop"][r,c]=(float(np.mean(nbr_lat))-m_val
                                        if nbr_lat else 0.0)

                t_below_end=min((dist+80)/max(dist,1e-6),1.5)
                below_v=_sample_ray(norm_img,ox,oy,px_c,py_c,1.10,t_below_end,n=20)
                f["vert_continuity"][r,c]=float(
                    (below_v<0.40).sum())/max(len(below_v),1)

                full_v=_sample_ray(norm_img,ox,oy,px_c,py_c,t_surf,t_below_end,n=30)
                col_dark=float((full_v<0.40).mean())
                col_bright=float((full_v>0.60).mean())
                f["reverb_score"][r,c]=col_bright if col_dark>0.5 else 0.0
            else:
                f["ray_mean_above"][r,c]=m_val
                f["lateral_drop"][r,c]=0.0
                f["vert_continuity"][r,c]=0.0
                f["reverb_score"][r,c]=0.0

    X=np.stack([f[k].ravel() for k in FEATURE_NAMES],axis=1).astype(np.float32)
    X=np.nan_to_num(X,nan=0.,posinf=1.,neginf=-1.)
    n_expected = expected_n_features(model)
    if n_expected == X.shape[1] + len(DERIVED_FEATURE_NAMES):
        X, _ = engineer_probe_features(X, FEATURE_NAMES)
    elif n_expected is not None and n_expected != X.shape[1]:
        raise ValueError(
            f"Model expects {n_expected} features, but Predict.py built "
            f"{X.shape[1]}. Check feature engineering/model version."
        )
    ysc=model.predict_proba(X)[:,1]
    ysc[~inf.ravel()]=0.0
    return ysc.reshape(n_rows,n_cols), pm, inf, n_rows, n_cols


def sample_all_rays(norm_img, map_x, map_y, valid):
    mat=cv2.remap(norm_img,map_x,map_y,
                  interpolation=cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT,borderValue=0.0)
    mat[~valid]=0.0
    return mat


def compute_te(imat, valid):
    cnt=valid.sum(axis=1).clip(min=1)
    return (imat.sum(axis=1)/cnt).astype(np.float32)


def classify_shadow_type(below_vals):
    if len(below_vals)==0: return None,0.0
    std_b=float(np.std(below_vals)); mean_b=float(np.mean(below_vals))
    if std_b<CLEAN_MAX_STD: return "clean",std_b
    elif mean_b<0.30 and std_b<DIRTY_MAX_STD: return "dirty",std_b
    elif mean_b<0.60 and std_b<EDGE_MAX_STD: return "edge",std_b
    return None,std_b


def find_reflectors(intensities, valid_mask):
    n=len(intensities)
    if n<BRIGHT_WIN+DARK_WIN+2: return []
    smooth=np.convolve(intensities,np.ones(3)/3.0,mode='same')
    results=[]; min_dist=max(BRIGHT_WIN,DARK_WIN); last_refl=-min_dist

    for j in range(BRIGHT_WIN,n-DARK_WIN):
        if not valid_mask[j]: continue
        if j-last_refl<min_dist: continue
        above_v=smooth[max(0,j-BRIGHT_WIN):j][valid_mask[max(0,j-BRIGHT_WIN):j]]
        below_v=intensities[j+1:min(n,j+1+DARK_WIN)][
                    valid_mask[j+1:min(n,j+1+DARK_WIN)]]
        if len(above_v)==0 or len(below_v)==0: continue
        a_mean=float(above_v.mean()); b_mean=float(below_v.mean())
        drop=a_mean-b_mean; std_b=float(below_v.std())
        if (a_mean>BRIGHT_MEAN_THR and b_mean<DARK_MEAN_THR
                and drop>DROP_MIN and std_b<SMOOTH_THR):
            full_below=intensities[j+1:][valid_mask[j+1:]]
            shadow_type,_=classify_shadow_type(full_below)
            if shadow_type is None: continue
            results.append((j,shadow_type,float(1.0-b_mean)))
            last_refl=j
    return results


def check_false_shadow(intensities, valid_mask, refl_idx):
    n=len(intensities)
    if refl_idx<0 or refl_idx>=n-1: return True
    below_v=intensities[refl_idx+1:][valid_mask[refl_idx+1:]]
    if len(below_v)<3: return True
    return float((below_v>RECOVERY_THR).sum())/len(below_v) > MAX_RECOVERY_PCT


def _group_shadow_rays(shadow_confirmed, n_theta, edge_pct=0.07):
    el=int(n_theta*edge_pct); eh=n_theta-el
    filtered=[(i,ri,st,ps) for (i,ri,st,ps) in shadow_confirmed
               if el<=i<=eh]
    if not filtered: return []
    filtered.sort(key=lambda x:x[0])
    groups=[]; cur=[filtered[0]]
    for item in filtered[1:]:
        if item[0]-cur[-1][0]<=3: cur.append(item)
        else: groups.append(cur); cur=[item]
    if cur: groups.append(cur)
    return groups


def fill_shadow_corridors(shadow_confirmed, cx, cy, R,
                           map_x, map_y, valid, r_arr, norm_img,
                           score_grid, in_fan, n_rows, n_cols, H, W,
                           thr_final, rf_min):
    """
    FIX 3: get_pts dùng rr trực tiếp.
    r_arr đã bắt đầu từ R nên KHÔNG cộng thêm R.
    """
    SHADOW_DARK_THR=0.62; POLY_MARGIN_DEG=0.8
    ox,oy=float(cx),float(cy)
    n_theta=map_x.shape[0]
    pixel_mask=np.zeros((n_rows*PATCH_H,n_cols*PATCH_W),dtype=np.uint8)
    reflectors=[]

    confirmed_pass=[]
    for (i,ri,stype,phys_sc) in shadow_confirmed:
        px_r=int(round(float(map_x[i,ri]))); py_r=int(round(float(map_y[i,ri])))
        if not (0<=px_r<W and 0<=py_r<H): continue
        pc_r=px_r//PATCH_W; pr_r=py_r//PATCH_H
        rf_r=float(score_grid[pr_r,pc_r]) \
             if 0<=pr_r<n_rows and 0<=pc_r<n_cols else 0.0
        final_sc=W_RF*rf_r+W_PHYS*phys_sc
        if final_sc<thr_final or rf_r<rf_min: continue
        confirmed_pass.append((i,ri,stype,phys_sc,px_r,py_r,final_sc))

    if not confirmed_pass:
        return np.zeros((n_rows,n_cols),dtype=np.float32),[]

    groups=_group_shadow_rays(
        [(i,ri,st,ps) for (i,ri,st,ps,*_) in confirmed_pass], n_theta)

    for group in groups:
        i_min=group[0][0]; i_max=group[-1][0]
        ri_med=int(np.median([ri for (_,ri,*_) in group]))
        t_l=float(np.arctan2(float(map_x[i_min,ri_med])-ox,
                              float(map_y[i_min,ri_med])-oy))-np.radians(POLY_MARGIN_DEG)
        t_r=float(np.arctan2(float(map_x[i_max,ri_med])-ox,
                              float(map_y[i_max,ri_med])-oy))+np.radians(POLY_MARGIN_DEG)

        r_start=float(r_arr[min(ri for (_,ri,*_) in group)])
        r_end=r_start
        for (i,ri,*_) in group:
            for j in range(ri,map_x.shape[1]):
                if not valid[i,j]: break
                r_end=max(r_end,float(r_arr[j]))
        if r_end<=r_start: continue

        n_seg=max(8,int((r_end-r_start)/PATCH_H*3))

        def get_pts(theta_,rs,re,ns):
            pts=[]
            for rr in np.linspace(rs,re,ns):
                # FIX 3: rr đã tính từ tâm ảo (bao gồm R), không + thêm R
                px2=int(np.clip(round(ox+rr*np.sin(theta_)),0,pixel_mask.shape[1]-1))
                py2=int(np.clip(round(oy+rr*np.cos(theta_)),0,pixel_mask.shape[0]-1))
                pts.append([px2,py2])
            return pts

        pts_l=get_pts(t_l,r_start,r_end,n_seg)
        pts_r=get_pts(t_r,r_start,r_end,n_seg)
        if len(pts_l)<2: continue

        poly=np.array(pts_l+pts_r[::-1],dtype=np.int32)
        cv2.fillPoly(pixel_mask,[poly],255)

        rep=group[len(group)//2]
        reflectors.append({
            "px":int(round(float(map_x[rep[0],rep[1]]))),
            "py":int(round(float(map_y[rep[0],rep[1]]))),
            "type":rep[2],"phys":round(rep[3],3),
        })

    pred_final=np.zeros((n_rows,n_cols),dtype=np.float32)
    for r in range(n_rows):
        for c in range(n_cols):
            y0=r*PATCH_H; y1=y0+PATCH_H; x0=c*PATCH_W; x1=x0+PATCH_W
            reg=pixel_mask[y0:y1,x0:x1]
            if reg.size==0: continue
            if float(reg.sum())/(255.0*reg.size)<0.30: continue
            if float(norm_img[y0:y1,x0:x1].mean())>SHADOW_DARK_THR: continue
            pred_final[r,c]=1.0

    g8=(pred_final>0).astype(np.uint8)*255
    nl,labels,stats,_=cv2.connectedComponentsWithStats(g8,connectivity=8)
    clean=np.zeros_like(pred_final)
    for idx in range(1,nl):
        if stats[idx,cv2.CC_STAT_AREA]>=2: clean[labels==idx]=1.0
    return clean, reflectors


def predict_panel(norm_img, fan_mask, fan_raw, model,
                   thr_final=THR_FINAL, n_theta=N_THETA, n_r=N_R_SAMPLES,
                   te_low_pct=TE_LOW_PCT, rf_min=RF_MIN_PROB,
                   max_shadow_frac=MAX_SHADOW_PCT):
    import time
    H,W=norm_img.shape; t0=time.time()

    cx,cy,R,phi_left,phi_right=fit_probe_circle(fan_raw)
    arc_center=(int(round(np.clip(cx,0,W-1))),
                int(round(np.clip(cy,0,H-1))))

    sg,pm,inf,n_rows,n_cols=precompute_all_patches(
        norm_img,fan_mask,cx,cy,R,model)
    print(f"      RF: {time.time()-t0:.1f}s  in_fan={int(inf.sum())}")

    t1=time.time()
    res=build_ray_maps(fan_mask,cx,cy,R,phi_left,phi_right,n_theta,n_r)
    if res[0] is None:
        return _empty(fan_mask,arc_center,cx,cy,R,n_rows,n_cols,H,W)
    map_x,map_y,r_arr,valid=res

    imat=sample_all_rays(norm_img,map_x,map_y,valid)
    te=compute_te(imat,valid)
    print(f"      Rays: {time.time()-t1:.2f}s")

    te_valid=te[te>0]
    if len(te_valid)==0:
        return _empty(fan_mask,arc_center,cx,cy,R,n_rows,n_cols,H,W)
    te_thr=float(np.percentile(te_valid,te_low_pct))
    shadow_mask=te<te_thr
    shadow_idx=np.where(shadow_mask)[0].tolist()
    print(f"      shadow_rays={len(shadow_idx)}/{n_theta}  te_thr={te_thr:.3f}")

    shadow_confirmed=[]; n_clean=0; n_dirty=0; n_edge=0; n_rej=0
    for i in shadow_idx:
        for (ri,stype,phys_sc) in find_reflectors(imat[i],valid[i]):
            if check_false_shadow(imat[i],valid[i],ri):
                n_rej+=1; continue
            shadow_confirmed.append((i,ri,stype,phys_sc))
            if stype=="clean": n_clean+=1
            elif stype=="dirty": n_dirty+=1
            elif stype=="edge": n_edge+=1

    print(f"      confirmed: clean={n_clean} dirty={n_dirty} "
          f"edge={n_edge} rejected={n_rej}")

    if not shadow_confirmed:
        return _empty(fan_mask,arc_center,cx,cy,R,n_rows,n_cols,H,W)

    n_in_fan=max(1,int(inf.sum()))
    max_p=max(1,int(n_in_fan*max_shadow_frac))

    pred_final,reflectors=fill_shadow_corridors(
        shadow_confirmed[:max_p*2],
        cx,cy,R,map_x,map_y,valid,r_arr,norm_img,
        sg,inf,n_rows,n_cols,H,W,thr_final,rf_min)

    n_final=int(pred_final.sum()); pct=n_final/n_in_fan*100
    t_total=time.time()-t0
    print(f"      TOTAL {t_total:.1f}s  shadow={n_final}/{n_in_fan}({pct:.1f}%)")

    te_map=np.ones((n_rows,n_cols),dtype=np.float32)
    for i in range(n_theta):
        tv=float(te[i])
        for j in range(0,n_r,4):
            if not valid[i,j]: continue
            ppx=int(round(float(map_x[i,j]))); ppy=int(round(float(map_y[i,j])))
            if 0<=ppx<W and 0<=ppy<H:
                pc=ppx//PATCH_W; pr=ppy//PATCH_H
                if 0<=pr<n_rows and 0<=pc<n_cols:
                    te_map[pr,pc]=min(te_map[pr,pc],tv)

    return {
        "fan":fan_mask,"origin":arc_center,"cx":cx,"cy":cy,"R":R,
        "score_grid":sg,"pred_final":pred_final,"te_map":te_map,
        "reflectors":reflectors,"te_per_ray":te,
        "shadow_ray_mask":shadow_mask,
        "map_x":map_x,"map_y":map_y,"valid":valid,
        "n_rows":n_rows,"n_cols":n_cols,"H":H,"W":W,
        "n_final":n_final,
        "summary":(f"confirmed={len(shadow_confirmed)} "
                   f"clean={n_clean} dirty={n_dirty} edge={n_edge} "
                   f"patches={n_final}/{n_in_fan}({pct:.1f}%) "
                   f"time={t_total:.1f}s"),
    }


def _empty(fan,arc_center,cx,cy,R,n_rows,n_cols,H,W):
    return {"fan":fan,"origin":arc_center,"cx":cx,"cy":cy,"R":R,
            "score_grid":np.zeros((n_rows,n_cols)),
            "pred_final":np.zeros((n_rows,n_cols)),
            "te_map":np.ones((n_rows,n_cols)),
            "reflectors":[],"te_per_ray":np.array([]),
            "shadow_ray_mask":np.array([]),
            "map_x":None,"map_y":None,"valid":None,
            "n_rows":n_rows,"n_cols":n_cols,"H":H,"W":W,
            "n_final":0,"summary":"no valid fan"}


def predict_image(img_path,model,thr_final,n_theta,n_r,te_low_pct,rf_min,max_shadow_frac):
    img_np=np.array(Image.open(img_path).convert("RGB"),dtype=np.uint8)
    gray=cv2.medianBlur(cv2.cvtColor(img_np,cv2.COLOR_RGB2GRAY),ksize=3)
    H,W=gray.shape
    panels=detect_and_split_dual_panel(gray)
    results=[]
    for idx,(panel_gray,x_off) in enumerate(panels):
        ph,pw=panel_gray.shape[:2]
        norm=normalize(panel_gray)
        fan,fan_raw=get_fan_mask(norm,thresh=0.05)
        print(f"    Panel {idx+1}/{len(panels)} ({pw}×{ph})...")
        res=predict_panel(norm,fan,fan_raw,model,
                           thr_final,n_theta,n_r,te_low_pct,rf_min,max_shadow_frac)
        res["x_offset"]=int(x_off); res["H"]=ph; res["W"]=pw
        results.append(res)
        print(f"      {res['summary']}")
    total=sum(r["n_final"] for r in results)
    return {"gray":gray,"results":results,"H":int(H),"W":int(W),"total_patches":int(total)}


TYPE_COLOR={"clean":(200,50,50),"dirty":(50,150,200),"edge":(50,200,150)}


def draw_grid(img,n_rows,n_cols,color=(55,55,55)):
    H,W=img.shape[:2]
    for r in range(n_rows+1):
        y=r*PATCH_H
        if 0<=y<H: cv2.line(img,(0,y),(W-1,y),color,1)
    for c in range(n_cols+1):
        x=c*PATCH_W
        if 0<=x<W: cv2.line(img,(x,0),(x,H-1),color,1)


def make_compare(gray,results,H,W):
    left=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    right=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    total=0

    for res in results:
        x_off=res["x_offset"]; ph_r=res["H"]; pw_r=res["W"]
        n_rows=res["n_rows"]; n_cols=res["n_cols"]
        ox,oy=res["origin"]
        cx_r=res.get("cx",ox); cy_r=res.get("cy",oy); R_r=res.get("R",0)
        pf=res["pred_final"]; sg=res["score_grid"]; tem=res["te_map"]
        fan=res["fan"]
        y1=min(ph_r,H); x1=min(x_off+pw_r,W)
        lp=left[:y1,x_off:x1]; rp=right[:y1,x_off:x1]

        te_min=float(tem[tem<1].min()) if (tem<1).any() else 0.0
        te_max=float(tem.max()); te_rng=max(te_max-te_min,1e-6)
        for r in range(n_rows):
            for c in range(n_cols):
                py0=r*PATCH_H; py1_=min(py0+PATCH_H,y1)
                px0=c*PATCH_W; px1_=min(px0+PATCH_W,x1-x_off)
                if py0>=py1_ or px0>=px1_: continue
                tv=float(tem[r,c]); sc=float(sg[r,c])
                if tv<=0 and sc<=0: continue
                t=np.clip((tv-te_min)/te_rng,0.,1.)
                clr=(0,int(150*t),int(150*(1-t)))
                roi=lp[py0:py1_,px0:px1_]
                mc=np.full_like(roi,clr)
                lp[py0:py1_,px0:px1_]=cv2.addWeighted(roi,0.45,mc,0.55,0)
                if PATCH_H>=14 and sc>0.25:
                    cv2.putText(lp,f"{sc:.2f}",(px0+1,py0+PATCH_H-3),
                                cv2.FONT_HERSHEY_SIMPLEX,0.22,(255,255,255),1)
        draw_grid(lp,n_rows,n_cols)

        # Vẽ arc đầu dò
        if R_r>0:
            cv2.ellipse(lp,
                        (int(round(np.clip(cx_r,0,pw_r-1))),
                         int(round(np.clip(cy_r,0,ph_r-1)))),
                        (int(round(R_r)),int(round(R_r))),
                        0,-150,-30,(0,220,220),2)
        # Tâm ảo (có thể trên ảnh)
        ox_d=int(np.clip(cx_r,0,pw_r-1)); oy_d=int(np.clip(cy_r,0,ph_r-1))
        cv2.circle(lp,(ox_d,oy_d),7,(0,165,255),-1)

        smask=res.get("shadow_ray_mask",np.array([]))
        mx=res.get("map_x"); my_=res.get("map_y"); v=res.get("valid")
        if mx is not None and len(smask)>0:
            nd=0
            for i in range(len(smask)):
                if not smask[i] or nd>=20: break
                pts=[(int(round(float(mx[i,j]))),int(round(float(my_[i,j]))))
                     for j in range(0,mx.shape[1],4) if v[i,j]]
                for k in range(len(pts)-1):
                    if all(0<=p[q]<[pw_r,ph_r][q]
                           for p in [pts[k],pts[k+1]] for q in [0,1]):
                        cv2.line(lp,pts[k],pts[k+1],(0,0,120),1)
                nd+=1

        for rf in res["reflectors"]:
            rx,ry=rf["px"],rf["py"]
            if 0<=rx<pw_r and 0<=ry<ph_r:
                clr=TYPE_COLOR.get(rf.get("type","clean"),(0,200,255))
                cv2.circle(lp,(rx,ry),5,clr,-1)
                cv2.circle(lp,(rx,ry),7,(0,0,0),1)

        fc,_=cv2.findContours(fan,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rp,fc,-1,(60,60,60),1)
        for r in range(n_rows):
            for c in range(n_cols):
                if pf[r,c]<=0: continue
                py0=r*PATCH_H; py1_=min(py0+PATCH_H,y1)
                px0=c*PATCH_W; px1_=min(px0+PATCH_W,x1-x_off)
                if py0>=py1_ or px0>=px1_: continue
                roi=rp[py0:py1_,px0:px1_]
                mc=np.full_like(roi,(30,30,180))
                rp[py0:py1_,px0:px1_]=cv2.addWeighted(roi,0.38,mc,0.62,0)
        draw_grid(rp,n_rows,n_cols)
        for r in range(n_rows):
            for c in range(n_cols):
                if pf[r,c]<=0: continue
                py0=r*PATCH_H; py1_=min(py0+PATCH_H,y1)
                px0=c*PATCH_W; px1_=min(px0+PATCH_W,x1-x_off)
                cv2.rectangle(rp,(px0,py0),(px1_-1,py1_-1),(0,0,220),2)
        total+=res["n_final"]

    def hdr(img,txt,clr):
        cv2.rectangle(img,(0,0),(img.shape[1],22),(20,20,20),-1)
        cv2.putText(img,txt,(6,15),cv2.FONT_HERSHEY_SIMPLEX,0.38,clr,1)
    hdr(left, "TE|RF(18f)|cyan=arc|●tâm ảo|◆clean dirty edge",(150,220,150))
    hdr(right,f"OVERLAY | {total} patches | clean/dirty/edge",(150,150,220))
    sep=np.full((H,4,3),50,dtype=np.uint8)
    return np.hstack([left,sep,right])


def make_simple_overlay(gray,results,H,W):
    vis=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR); total=0
    for res in results:
        x_off=res["x_offset"]; ph_r=res["H"]; pw_r=res["W"]
        n_rows=res["n_rows"]; n_cols=res["n_cols"]; pf=res["pred_final"]
        y1=min(ph_r,H); x1=min(x_off+pw_r,W)
        for r in range(n_rows):
            for c in range(n_cols):
                if pf[r,c]<=0: continue
                py0=r*PATCH_H; py1_=min(py0+PATCH_H,y1)
                px0=c*PATCH_W; px1_=min(px0+PATCH_W,x1-x_off)
                roi=vis[py0:py1_,x_off+px0:x_off+px1_]
                mc=np.full_like(roi,(30,30,180))
                vis[py0:py1_,x_off+px0:x_off+px1_]=cv2.addWeighted(roi,0.4,mc,0.6,0)
        pu=cv2.resize(pf.astype(np.float32),(pw_r,ph_r),interpolation=cv2.INTER_NEAREST)
        su8=(pu>0.5).astype(np.uint8)*255
        cnts,_=cv2.findContours(su8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            cnt[:,0,0]+=x_off
            cv2.drawContours(vis,[cnt],-1,(0,0,220),2)
        total+=res["n_final"]
    cv2.putText(vis,f"shadow: {total} patches",(6,H-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.40,(210,210,210),1)
    return vis


def run(test_dir,models_dir,thr_final,n_theta,n_r,
        te_low_pct,rf_min,max_shadow_frac,debug):
    test_dir=os.path.abspath(test_dir)
    out_dir=os.path.join(os.path.dirname(test_dir),Path(test_dir).name+"_results")
    overlay_dir=os.path.join(out_dir,"overlays")
    os.makedirs(overlay_dir,exist_ok=True)

    model,fname=load_model(models_dir)
    img_paths=[]
    for ext in ["*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.PNG"]:
        img_paths+=glob.glob(os.path.join(test_dir,ext))
    img_paths=sorted(set(img_paths))
    if not img_paths:
        print(f"Không có ảnh: {test_dir}"); sys.exit(1)

    print(f"\n[test] {len(img_paths)} ảnh  model={fname}")
    print(f"[FIX] cy không clip về 0 | r_arr bắt từ R | get_pts dùng rr trực tiếp")
    print(f"{'─'*65}")

    summary=[]; ok=0; failed=0
    for img_path in img_paths:
        stem=Path(img_path).stem
        try:
            print(f"\n  → {stem}")
            out=predict_image(img_path,model,thr_final,n_theta,n_r,
                               te_low_pct,rf_min,max_shadow_frac)
            gray=out["gray"]; H_=out["H"]; W_=out["W"]
            compare=make_compare(gray,out["results"],H_,W_)
            cv2.imwrite(os.path.join(overlay_dir,f"{stem}_compare.png"),compare)
            simple=make_simple_overlay(gray,out["results"],H_,W_)
            cv2.imwrite(os.path.join(overlay_dir,f"{stem}.png"),simple)
            summary.append({"image":stem,"patches":out["total_patches"]})
            mark="✓" if out["total_patches"]>0 else "○"
            print(f"    {mark} {out['total_patches']} patches")
            ok+=1
        except Exception as e:
            print(f"    ✗ {e}"); traceback.print_exc(); failed+=1

    n_det=sum(1 for r in summary if r["patches"]>0)
    with open(os.path.join(out_dir,"predictions.json"),"w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    print(f"\n{'═'*65}")
    print(f"  OK={ok}  Failed={failed}  Shadow: {n_det}/{ok} ảnh")
    print(f"  Tổng: {sum(r['patches'] for r in summary)} patches → {overlay_dir}")
    print(f"{'═'*65}")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--test_dir",required=True)
    p.add_argument("--models_dir",
        default=r"C:\Users\ThinkPad\DATN\Data\bongcan_processed\models")
    p.add_argument("--thr_final",default=THR_FINAL,type=float)
    p.add_argument("--n_theta",  default=N_THETA,  type=int)
    p.add_argument("--n_r",      default=N_R_SAMPLES,type=int)
    p.add_argument("--te_low_pct",default=TE_LOW_PCT,type=int)
    p.add_argument("--rf_min",   default=RF_MIN_PROB,type=float)
    p.add_argument("--max_shadow_pct",default=15,type=int)
    p.add_argument("--debug",action="store_true")
    args=p.parse_args()

    print("="*65)
    print("  PREDICT — Fan Beam (cy âm OK | r bắt từ R | 3 fixes)")
    print("="*65+"\n")

    run(args.test_dir,args.models_dir,args.thr_final,
        args.n_theta,args.n_r,args.te_low_pct,args.rf_min,
        args.max_shadow_pct/100.0,args.debug)
