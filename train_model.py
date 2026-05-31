"""
train_model.py  —  Sehra et al. (2018) on COCOMO-81 benchmark
==============================================================

Dataset: COCOMO-81 (63 projects, 1980s defense/aerospace software)
  - Drops rows with NaN / zero feature values
  - Computes LOO signed relative errors → stores 10th/90th pct as
    empirical 80% prediction interval (pi_lo_pct, pi_hi_pct)

Pipeline (mirrored exactly in app.py):
  raw X  →  log(X)  →  MinMaxScaler  →  × sqrt(fahp_w)  →  SVR
  raw y  →  log1p(y) →  MinMaxScaler  →  SVR target

Run:
  python train_model.py

Author: Sehra et al. 2018 implementation for Mitacs demo
"""

import numpy as np
import pandas as pd
import pickle
import os
import warnings
from itertools import product as iterproduct

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DATA
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "loc", "rely", "data", "cplx", "time", "stor",
    "virt", "turn", "acap", "aexp", "pcap", "vexp",
    "lexp", "modp", "tool", "sced", "dev_mode_ord",
]
FEATURE_NAMES = [
    "LOC (KLOC)", "RELY", "DATA", "CPLX", "TIME", "STOR",
    "VIRT", "TURN", "ACAP", "AEXP", "PCAP", "VEXP",
    "LEXP", "MODP", "TOOL", "SCED", "Dev Mode",
]

# ── Load COCOMO-81 ─────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "cocomo81.csv")
df_c81   = pd.read_csv(csv_path)
n_cocomo81_raw = len(df_c81)

# ── Encode development mode (ordinal: organic=1, semidetached=2, embedded=3)
# log(1)=0, log(2)=0.69, log(3)=1.10 — well-spaced after log transform.
# Captures the distinct COCOMO base coefficients (a=2.4/3.0/3.6, b=1.05/1.12/1.20).
DEV_MODE_ORD = {"organic": 1, "semidetached": 2, "embedded": 3}
df_c81["dev_mode_ord"] = df_c81["dev_mode"].map(DEV_MODE_ORD).astype(float)

# ── Clean and prepare ────────────────────────────────────────────────────────
df_all = df_c81[FEATURE_COLS + ["actual", "dev_mode"]].copy()
df_all = df_all.dropna()
df_all = df_all[(df_all[FEATURE_COLS] > 0).all(axis=1)]
df_all = df_all[df_all["actual"] > 0]

n_cocomo81 = len(df_all)
n_nasa93   = 0

X_raw = df_all[FEATURE_COLS].values.astype(float)
y_raw = df_all["actual"].values.astype(float)

print(f"Dataset: {len(df_all)} projects  (COCOMO-81: {n_cocomo81})")
print(f"Effort : min={y_raw.min():.1f}  max={y_raw.max():.1f}  mean={y_raw.mean():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CHANG (1996) EXTENT-ANALYSIS FAHP
# ─────────────────────────────────────────────────────────────────────────────

TFN_SCALE = {
    1: (1.0, 1.0, 1.0),
    2: (1.0, 2.0, 3.0),
    3: (1.0, 3.0, 5.0),
    4: (1.0, 4.0, 7.0),
    5: (3.0, 5.0, 7.0),
    6: (3.0, 6.0, 9.0),
    7: (5.0, 7.0, 9.0),
    8: (7.0, 8.0, 9.0),
    9: (7.0, 9.0, 9.0),
}

def _tfn_reciprocal(tfn):
    l, m, u = tfn
    return (1.0 / u, 1.0 / m, 1.0 / l)

def _continuous_to_tfn(ratio: float):
    ratio = max(1.0 / 9.0, min(9.0, ratio))
    if ratio < 1.0:
        return _tfn_reciprocal(_continuous_to_tfn(1.0 / ratio))
    lo = max(1, min(9, int(np.floor(ratio))))
    hi = max(1, min(9, int(np.ceil(ratio))))
    if lo == hi:
        return TFN_SCALE[lo]
    t = ratio - lo
    return tuple(TFN_SCALE[lo][k] * (1 - t) + TFN_SCALE[hi][k] * t for k in range(3))

def build_pairwise_matrix(values: np.ndarray):
    n   = len(values)
    mat = [[(1.0, 1.0, 1.0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ratio     = float(values[i]) / float(max(values[j], 1e-9))
            tfn       = _continuous_to_tfn(ratio)
            mat[i][j] = tfn
            mat[j][i] = _tfn_reciprocal(tfn)
    return mat

def synthetic_extent_values(mat):
    n        = len(mat)
    row_sums = []
    for i in range(n):
        sl = sum(mat[i][j][0] for j in range(n))
        sm = sum(mat[i][j][1] for j in range(n))
        su = sum(mat[i][j][2] for j in range(n))
        row_sums.append((sl, sm, su))
    tl, tm, tu = [sum(r[k] for r in row_sums) for k in range(3)]
    inv_total  = (1.0 / tu, 1.0 / tm, 1.0 / tl)
    return [(sl * inv_total[0], sm * inv_total[1], su * inv_total[2])
            for (sl, sm, su) in row_sums]

def degree_of_possibility(M1, M2):
    _,  m1, u1 = M1   # l1 not needed by the Chang (1996) formula
    l2, m2, _  = M2   # u2 not needed
    if m1 >= m2:
        return 1.0
    if u1 <= l2:          # M1 entirely below M2 → impossible for M1 ≥ M2
        return 0.0
    denom = (m2 - l2) - (m1 - u1)   # == u1 - m1 + m2 - l2, always > 0 here
    if abs(denom) < 1e-10:
        return 0.0
    return max(0.0, min(1.0, (u1 - l2) / denom))   # correct Chang (1996) formula

def consistency_ratio(mat) -> float:
    n     = len(mat)
    crisp = np.array([[mat[i][j][1] for j in range(n)] for i in range(n)], dtype=float)
    col_sums = crisp.sum(axis=0)
    norm     = crisp / (col_sums + 1e-10)
    weights  = norm.mean(axis=1)
    Aw       = crisp @ weights
    lam_max  = float(np.mean(Aw / (weights + 1e-10)))
    CI       = (lam_max - n) / max(n - 1, 1)
    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI       = RI_table.get(n, 1.54)
    return CI / RI if RI > 0 else 0.0

def fahp_weights_from_values(values: np.ndarray, label="") -> np.ndarray:
    mat = build_pairwise_matrix(values)
    cr  = consistency_ratio(mat)
    if cr > 0.1:
        print(f"  ! {label} CR={cr:.3f} > 0.1 (auto-generated ratios)")
    else:
        print(f"  OK {label} CR={cr:.3f}")
    S       = synthetic_extent_values(mat)
    n       = len(S)
    W_prime = np.array([
        min(degree_of_possibility(S[i], S[j]) for j in range(n) if j != i)
        for i in range(n)
    ])
    total = W_prime.sum()
    return W_prime / total if total > 1e-10 else np.ones(n) / n

# ── 1a. Project-level weights ─────────────────────────────────────────────────
print("\n── FAHP: computing project weights ──")
W_effort  = fahp_weights_from_values(y_raw,       label="Effort criterion")
W_loc     = fahp_weights_from_values(X_raw[:, 0], label="LOC criterion")
W_project = (0.5 * W_effort + 0.5 * W_loc)
W_project = W_project / W_project.sum()

# ── 1b. Feature weights via Spearman correlation ──────────────────────────────
feature_corr = np.array([
    abs(spearmanr(X_raw[:, fi], W_project).correlation)
    for fi in range(len(FEATURE_COLS))
])
feature_corr = np.clip(feature_corr, 0.01, None)
fahp_w       = feature_corr / feature_corr.sum()

print("\nFAHP-derived feature weights (top 5):")
for idx in np.argsort(fahp_w)[::-1][:5]:
    print(f"  {FEATURE_NAMES[idx]:>14s}: {fahp_w[idx]:.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
X_log = np.log(np.clip(X_raw, 1e-3, None))
y_log = np.log1p(y_raw)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_sc     = scaler_X.fit_transform(X_log)
y_sc     = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel()

kernel_w = np.sqrt(fahp_w)
X_w      = X_sc * kernel_w

print(f"\nPreprocessing: X_w shape={X_w.shape}  "
      f"y_sc range=[{y_sc.min():.3f}, {y_sc.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HYPERPARAMETER SEARCH  (LOO-MMRE grid)
# ─────────────────────────────────────────────────────────────────────────────
C_grid       = [0.5, 1, 5, 10, 50, 100, 200, 500]
gamma_grid   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, "scale"]
epsilon_grid = [0.05, 0.1, 0.2]

def loo_mmre_and_errors(C, gamma, epsilon=0.1):
    """Returns (MMRE, signed_relative_errors) for given C, gamma, epsilon."""
    loo    = LeaveOneOut()
    errors = []
    sre    = []   # signed relative errors for PI computation
    for train_idx, test_idx in loo.split(X_w):
        svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        svr.fit(X_w[train_idx], y_sc[train_idx])
        p_sc  = svr.predict(X_w[test_idx])
        p_log = scaler_y.inverse_transform(p_sc.reshape(-1, 1))[0, 0]
        p_pm  = max(float(np.expm1(p_log)), 0.1)
        actual = y_raw[test_idx[0]]
        re     = (p_pm - actual) / actual      # signed: positive=over, negative=under
        errors.append(abs(re))
        sre.append(re)
    return float(np.mean(errors)), np.array(sre)

def loo_mmre(C, gamma, epsilon=0.1):
    m, _ = loo_mmre_and_errors(C, gamma, epsilon)
    return m

print("\n── Grid search (LOO-MMRE) ──")
best_mmre, best_C, best_gamma, best_epsilon = 1e9, None, None, 0.1

for C, gam, eps in iterproduct(C_grid, gamma_grid, epsilon_grid):
    mmre = loo_mmre(C, gam, eps)
    if mmre < best_mmre:
        best_mmre, best_C, best_gamma, best_epsilon = mmre, C, gam, eps

print(f"Coarse best: LOO-MMRE={best_mmre:.4f}  C={best_C}  gamma={best_gamma}  eps={best_epsilon}")

# Fine search around best coarse result
if isinstance(best_C, (int, float)):
    C_fine    = [best_C * f for f in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
    gam_cands = ([best_gamma * f for f in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
                 if isinstance(best_gamma, float) else [best_gamma])
    eps_cands = sorted({max(0.01, best_epsilon * f) for f in [0.5, 1.0, 2.0]})
    for C, gam, eps in iterproduct(C_fine, gam_cands, eps_cands):
        mmre = loo_mmre(C, gam, eps)
        if mmre < best_mmre:
            best_mmre, best_C, best_gamma, best_epsilon = mmre, C, gam, eps

print(f"Fine   best: LOO-MMRE={best_mmre:.4f}  C={best_C:.4f}  gamma={best_gamma}  eps={best_epsilon:.4f}")

# ── Compute empirical 80% prediction interval from LOO signed errors ──────────
print("\n── Computing empirical prediction interval ──")
_, loo_sre = loo_mmre_and_errors(best_C, best_gamma, best_epsilon)
pi_lo_pct  = float(np.percentile(loo_sre, 10))   # 10th pct signed RE
pi_hi_pct  = float(np.percentile(loo_sre, 90))   # 90th pct signed RE
print(f"80% PI: [{pi_lo_pct*100:+.1f}%,  {pi_hi_pct*100:+.1f}%] of prediction")

# ── PRED(N): fraction of predictions within N% of actual ──────────────────────
loo_abs_re = np.abs(loo_sre)
pred25 = float(np.mean(loo_abs_re <= 0.25))
pred50 = float(np.mean(loo_abs_re <= 0.50))
print(f"PRED(25): {pred25*100:.1f}%  ({int(round(pred25*len(y_raw)))}/{len(y_raw)} projects within 25%)")
print(f"PRED(50): {pred50*100:.1f}%  ({int(round(pred50*len(y_raw)))}/{len(y_raw)} projects within 50%)")

# ── Per-mode MMRE breakdown ────────────────────────────────────────────────────
print("\n── Per-mode MMRE ──")
dev_mode_col = df_all["dev_mode"].values
mode_mmre_results = {}
for mode_key, mode_label in [("organic","Organic"),("semidetached","Semi-detached"),("embedded","Embedded")]:
    mask = dev_mode_col == mode_key
    if mask.sum() > 0:
        m = float(np.mean(loo_abs_re[mask]))
        p25m = float(np.mean(loo_abs_re[mask] <= 0.25))
        mode_mmre_results[mode_key] = {"mmre": round(m,4), "n": int(mask.sum()), "pred25": round(p25m,4)}
        print(f"  {mode_label:>14s}: n={mask.sum():>2d}  MMRE={m:.4f}  PRED(25)={p25m*100:.1f}%")

# ── Ablation: LOO-MMRE without FAHP weighting ─────────────────────────────────
print("\n── Ablation: SVR without FAHP weights (uniform) ──")
def loo_mmre_ablation():
    loo    = LeaveOneOut()
    errors = []
    for train_idx, test_idx in loo.split(X_sc):
        svr = SVR(kernel="rbf", C=best_C, gamma=best_gamma, epsilon=best_epsilon)
        svr.fit(X_sc[train_idx], y_sc[train_idx])
        p_sc2 = svr.predict(X_sc[test_idx])
        p_log = scaler_y.inverse_transform(p_sc2.reshape(-1,1))[0,0]
        p_pm  = max(float(np.expm1(p_log)), 0.1)
        errors.append(abs(p_pm - y_raw[test_idx[0]]) / y_raw[test_idx[0]])
    return float(np.mean(errors))

ablation_mmre    = loo_mmre_ablation()
fahp_improve_pct = (ablation_mmre - best_mmre) / ablation_mmre * 100
print(f"Without FAHP: LOO-MMRE = {ablation_mmre:.4f}")
print(f"With    FAHP: LOO-MMRE = {best_mmre:.4f}  (FAHP reduces error by {fahp_improve_pct:.1f}%)")

nasa93_mmre    = None
nasa93_pred25  = None
n_nasa93_valid = 0

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FINAL MODEL  (trained on all projects)
# ─────────────────────────────────────────────────────────────────────────────
final_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma, epsilon=best_epsilon)
final_model.fit(X_w, y_sc)

train_pred_sc  = final_model.predict(X_w)
train_pred_log = scaler_y.inverse_transform(train_pred_sc.reshape(-1, 1)).ravel()
train_pred_pm  = np.expm1(train_pred_log)
train_mmre     = float(np.mean(np.abs(y_raw - train_pred_pm) / y_raw))
train_rmse     = float(np.sqrt(np.mean((y_raw - train_pred_pm) ** 2)))

print(f"\nFinal model — LOO-MMRE={best_mmre:.4f}  train MMRE={train_mmre:.4f}  "
      f"train RMSE={train_rmse:.1f} PM")

# ── Cross-dataset validation on NASA-93 ───────────────────────────────────────
nasa93_path = os.path.join(os.path.dirname(__file__), "nasa93.csv")
if os.path.exists(nasa93_path):
    print("\n── Cross-dataset validation: train=COCOMO-81, test=NASA-93 ──")
    df_n93 = pd.read_csv(nasa93_path)
    df_n93["dev_mode_ord"] = df_n93["dev_mode"].map(DEV_MODE_ORD).astype(float)
    df_n93_clean = df_n93[FEATURE_COLS + ["actual"]].dropna()
    df_n93_clean = df_n93_clean[(df_n93_clean[FEATURE_COLS] > 0).all(axis=1)]
    df_n93_clean = df_n93_clean[df_n93_clean["actual"] > 0]
    n_nasa93_valid = len(df_n93_clean)
    if n_nasa93_valid > 0:
        X_n93_raw = df_n93_clean[FEATURE_COLS].values.astype(float)
        y_n93_raw = df_n93_clean["actual"].values.astype(float)
        X_n93_log = np.log(np.clip(X_n93_raw, 1e-3, None))
        X_n93_sc  = scaler_X.transform(X_n93_log)
        X_n93_w   = X_n93_sc * kernel_w
        p_n93_sc  = final_model.predict(X_n93_w)
        p_n93_log = scaler_y.inverse_transform(p_n93_sc.reshape(-1,1)).ravel()
        p_n93_pm  = np.expm1(p_n93_log)
        n93_re    = np.abs(y_n93_raw - p_n93_pm) / y_n93_raw
        nasa93_mmre   = float(np.mean(n93_re))
        nasa93_pred25 = float(np.mean(n93_re <= 0.25))
        print(f"NASA-93: n={n_nasa93_valid}  MMRE={nasa93_mmre:.4f}  PRED(25)={nasa93_pred25*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE model.pkl
# ─────────────────────────────────────────────────────────────────────────────
artifacts = {
    "model":         final_model,
    "scaler_X":      scaler_X,
    "scaler_y":      scaler_y,
    "fahp_weights":  fahp_w,
    "kernel_w":      kernel_w,
    "X_train":       X_w,
    "y_train":       y_raw,

    "feature_names": FEATURE_NAMES,
    "log_transform": True,
    "best_C":        round(float(best_C), 6),
    "best_gamma":    best_gamma if isinstance(best_gamma, str)
                        else round(float(best_gamma), 6),
    "best_epsilon":  round(float(best_epsilon), 6),
    "loo_mmre":      round(best_mmre, 4),
    "train_mmre":    round(train_mmre, 4),
    "train_rmse":    round(train_rmse, 2),
    "n_train":       len(y_raw),
    "n_cocomo81":    n_cocomo81,
    "n_nasa93":      n_nasa93,
    "importance":    fahp_w,

    # Empirical prediction interval (80%) from LOO signed relative errors
    "pi_lo_pct":     pi_lo_pct,
    "pi_hi_pct":     pi_hi_pct,

    # Extended evaluation metrics
    "pred25":            round(pred25, 4),
    "pred50":            round(pred50, 4),
    "ablation_mmre":     round(ablation_mmre, 4),
    "fahp_improve_pct":  round(fahp_improve_pct, 1),
    "mode_mmre":         mode_mmre_results,
    "nasa93_mmre":       round(nasa93_mmre, 4) if nasa93_mmre is not None else None,
    "nasa93_pred25":     round(nasa93_pred25, 4) if nasa93_pred25 is not None else None,
    "n_nasa93_valid":    n_nasa93_valid,

    # Project-level weights for explainability
    "project_weights_effort": W_effort,
    "project_weights_loc":    W_loc,
    "project_weights":        W_project,

    # Raw training data for SHAP background reference
    "X_raw":         X_raw,
    "y_raw_pm":      y_raw,
}

out_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(out_path, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n✓  model.pkl saved  ({out_path})")
print(f"   LOO-MMRE        : {best_mmre:.4f}")
print(f"   PRED(25)        : {pred25*100:.1f}%")
print(f"   PRED(50)        : {pred50*100:.1f}%")
print(f"   Ablation MMRE   : {ablation_mmre:.4f}  (no FAHP)")
print(f"   FAHP improvement: {fahp_improve_pct:.1f}%")
if nasa93_mmre is not None:
    print(f"   NASA-93 MMRE    : {nasa93_mmre:.4f}  (cross-dataset, n={n_nasa93_valid})")
print(f"   Train-MMRE      : {train_mmre:.4f}")
print(f"   n_train         : {len(y_raw)}  (COCOMO-81: {n_cocomo81})")
print(f"   80% PI          : [{pi_lo_pct*100:+.1f}%,  {pi_hi_pct*100:+.1f}%]")
print(f"   Best HPs        : C={best_C:.4f}  gamma={best_gamma}  epsilon={best_epsilon:.4f}")
print(f"   FAHP top-3      : {[FEATURE_NAMES[i] for i in np.argsort(fahp_w)[::-1][:3]]}")
print(f"\nRestart your Streamlit app after running this script.")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\n── Spot-check first project ──")
v_log = np.log(np.clip(X_raw[0:1], 1e-3, None))
v_sc  = scaler_X.transform(v_log)
v_w   = v_sc * kernel_w
p_sc  = final_model.predict(v_w)
p_pm  = float(np.expm1(scaler_y.inverse_transform(p_sc.reshape(-1, 1))[0, 0]))
print(f"  Predicted: {p_pm:.1f} PM  |  Actual: {y_raw[0]:.1f} PM  "
      f"|  RE: {abs(p_pm - y_raw[0]) / y_raw[0]:.2%}")
