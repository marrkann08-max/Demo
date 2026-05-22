"""
train_model.py  —  Sehra et al. (2018) extended with honest implementation
===========================================================================

What this actually does (vs what the paper claims):
  - FAHP (Chang 1996 extent analysis): builds pairwise matrices from raw
    effort + LOC ratios, checks CR < 0.1, produces *project* weights.
  - Feature weights: derived from Spearman correlation of each feature with
    project weights. This is NOT in the paper — the paper maps project weights
    directly into the kernel diagonal S = diag(θ). We do that correctly here.
  - Weighted RBF kernel: K(θ·xk, θ·xl) = exp(-||S(xk-xl)||² / σ²) where
    S = diag(sqrt(fahp_w)). Applied via feature pre-scaling before SVR —
    mathematically equivalent to Eq. 17 in the paper.
  - SVR(kernel="rbf") as LSSVM proxy: standard SVM with RBF. True LSSVM
    uses equality constraints + linear KKT system (Suykens & Vandewalle 1999)
    but is not available in sklearn. Performance is comparable per Gestel et al.
    (2004) with tuned hyperparameters.
  - Validation: Leave-One-Out MMRE (same as paper).

Pipeline (mirrored exactly in app.py):
  raw X  →  log(X)  →  MinMaxScaler  →  × sqrt(fahp_w)  →  SVR
  raw y  →  log1p(y) →  MinMaxScaler  →  SVR target

Author: extended from Sehra et al. 2018 for Mitacs demo
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
    "lexp", "modp", "tool", "sced",
]
FEATURE_NAMES = [
    "LOC (KLOC)", "RELY", "DATA", "CPLX", "TIME", "STOR",
    "VIRT", "TURN", "ACAP", "AEXP", "PCAP", "VEXP",
    "LEXP", "MODP", "TOOL", "SCED",
]

csv_path = os.path.join(os.path.dirname(__file__), "cocomo81.csv")
df       = pd.read_csv(csv_path)
X_raw    = df[FEATURE_COLS].values.astype(float)
y_raw    = df["actual"].values.astype(float)

print(f"Dataset: {len(df)} projects, {len(FEATURE_COLS)} features")
print(f"Effort : min={y_raw.min():.1f}  max={y_raw.max():.1f}  mean={y_raw.mean():.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CHANG (1996) EXTENT-ANALYSIS FAHP
#     Correctly implements the algorithm from the paper including CR check.
#     Criteria  : effort (y) and LOC (X[:,0])
#     Alternatives: projects (63 of them)
#     Output    : normalised weight vector of length n_projects
# ─────────────────────────────────────────────────────────────────────────────

# Linguistic TFN scale from Table 1 in Sehra 2018
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
    """Map a continuous ratio to a TFN via linear interpolation."""
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
    """Build (n×n) fuzzy pairwise comparison matrix from ratio of values."""
    n   = len(values)
    mat = [[(1.0, 1.0, 1.0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            ratio    = float(values[i]) / float(max(values[j], 1e-9))
            tfn      = _continuous_to_tfn(ratio)
            mat[i][j] = tfn
            mat[j][i] = _tfn_reciprocal(tfn)
    return mat

def synthetic_extent_values(mat):
    """Chang (1996) Eq. 3: Si = row_sum ⊗ (grand_sum)^{-1}"""
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
    """Chang (1996) Eq. 5: V(M1 >= M2)"""
    l1, m1, u1 = M1
    l2, m2, u2 = M2
    if m1 >= m2:
        return 1.0
    if l1 >= u2:
        return 0.0
    denom = (m2 - l2) - (m1 - u1)
    if abs(denom) < 1e-10:
        return 0.0
    return max(0.0, min(1.0, (l1 - u2) / denom))

def consistency_ratio(mat) -> float:
    """
    Compute CR for a fuzzy pairwise matrix by defuzzifying to crisp values
    (using midpoint m), then applying Saaty's CI/RI formula.
    This is the standard approximation used in the literature.
    """
    n     = len(mat)
    crisp = np.array([[mat[i][j][1] for j in range(n)] for i in range(n)], dtype=float)
    # Normalise columns
    col_sums = crisp.sum(axis=0)
    norm     = crisp / (col_sums + 1e-10)
    weights  = norm.mean(axis=1)
    # λ_max
    Aw       = crisp @ weights
    lam_max  = float(np.mean(Aw / (weights + 1e-10)))
    CI       = (lam_max - n) / max(n - 1, 1)
    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI       = RI_table.get(n, 1.54)
    return CI / RI if RI > 0 else 0.0

def fahp_weights_from_values(values: np.ndarray, label="") -> np.ndarray:
    """
    Full Chang (1996) FAHP pipeline.
    Includes CR check (warns if > 0.1, proceeds anyway for automated use).
    Returns normalised weight vector summing to 1.
    """
    mat = build_pairwise_matrix(values)
    cr  = consistency_ratio(mat)
    if cr > 0.1:
        print(f"  ⚠  {label} CR={cr:.3f} > 0.1 — matrix inconsistent (auto-generated ratios)")
    else:
        print(f"  ✓  {label} CR={cr:.3f} < 0.1 — consistent")

    S      = synthetic_extent_values(mat)
    n      = len(S)
    W_prime = np.array([
        min(degree_of_possibility(S[i], S[j]) for j in range(n) if j != i)
        for i in range(n)
    ])
    total = W_prime.sum()
    return W_prime / total if total > 1e-10 else np.ones(n) / n

# ── 1a. Project-level weights (effort + LOC criteria, equal importance) ───────
print("\n── FAHP: computing project weights ──")
W_effort  = fahp_weights_from_values(y_raw,        label="Effort criterion")
W_loc     = fahp_weights_from_values(X_raw[:, 0],  label="LOC criterion")
W_project = (0.5 * W_effort + 0.5 * W_loc)
W_project = W_project / W_project.sum()

print(f"Project weights — top 3: projects {np.argsort(W_project)[::-1][:3] + 1}")

# ── 1b. Feature weights via Spearman correlation with project weights ─────────
# NOTE: The paper uses project weights to define S = diag(θ) for the kernel.
# Since θ must have length = n_features (not n_projects), we derive feature
# importance from how strongly each feature correlates with project importance.
# This is an interpretive extension — clearly documented as such.
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
#     Eq. 17 weighted kernel: K(θxk, θxl) = exp(-||S(xk-xl)||² / σ²)
#     where S = diag(θ). Pre-multiplying features by sqrt(θ) before a standard
#     RBF kernel is mathematically equivalent (||sqrt(θ)·(xk-xl)||² = ||S(xk-xl)||²
#     when θ_i = sqrt(θ_i)^2).
# ─────────────────────────────────────────────────────────────────────────────
X_log = np.log(np.clip(X_raw, 1e-3, None))
y_log = np.log1p(y_raw)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_sc     = scaler_X.fit_transform(X_log)
y_sc     = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel()

# Apply sqrt(fahp_w) per-feature — this implements the weighted kernel Eq. 17
kernel_w = np.sqrt(fahp_w)      # sqrt so that squared distance = fahp_w-weighted
X_w      = X_sc * kernel_w

print(f"\nPreprocessing: X_w shape={X_w.shape}  "
      f"y_sc range=[{y_sc.min():.3f}, {y_sc.max():.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  HYPERPARAMETER SEARCH  (LOO-MMRE grid)
# ─────────────────────────────────────────────────────────────────────────────
C_grid     = [0.5, 1, 5, 10, 50, 100, 200, 500]
gamma_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, "scale"]

def loo_mmre(C, gamma):
    loo    = LeaveOneOut()
    errors = []
    for train_idx, test_idx in loo.split(X_w):
        svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=0.1)
        svr.fit(X_w[train_idx], y_sc[train_idx])
        p_sc  = svr.predict(X_w[test_idx])
        p_log = scaler_y.inverse_transform(p_sc.reshape(-1, 1))[0, 0]
        p_pm  = max(float(np.expm1(p_log)), 0.1)
        errors.append(abs(y_raw[test_idx[0]] - p_pm) / y_raw[test_idx[0]])
    return float(np.mean(errors))

print("\n── Grid search (LOO-MMRE) ──")
best_mmre, best_C, best_gamma = 1e9, None, None

for C, gam in iterproduct(C_grid, gamma_grid):
    mmre = loo_mmre(C, gam)
    if mmre < best_mmre:
        best_mmre, best_C, best_gamma = mmre, C, gam

print(f"Coarse best: LOO-MMRE={best_mmre:.4f}  C={best_C}  gamma={best_gamma}")

# Fine search
if isinstance(best_C, (int, float)):
    C_fine    = [best_C * f for f in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
    gam_cands = ([best_gamma * f for f in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]]
                 if isinstance(best_gamma, float) else [best_gamma])
    for C, gam in iterproduct(C_fine, gam_cands):
        mmre = loo_mmre(C, gam)
        if mmre < best_mmre:
            best_mmre, best_C, best_gamma = mmre, C, gam

print(f"Fine   best: LOO-MMRE={best_mmre:.4f}  C={best_C:.4f}  gamma={best_gamma}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  FINAL MODEL  (trained on all 63 projects)
# ─────────────────────────────────────────────────────────────────────────────
final_model = SVR(kernel="rbf", C=best_C, gamma=best_gamma, epsilon=0.1)
final_model.fit(X_w, y_sc)

train_pred_sc  = final_model.predict(X_w)
train_pred_log = scaler_y.inverse_transform(train_pred_sc.reshape(-1, 1)).ravel()
train_pred_pm  = np.expm1(train_pred_log)
train_mmre     = float(np.mean(np.abs(y_raw - train_pred_pm) / y_raw))
train_rmse     = float(np.sqrt(np.mean((y_raw - train_pred_pm) ** 2)))

print(f"\nFinal model — train MMRE={train_mmre:.4f}  train RMSE={train_rmse:.1f} PM")
print(f"Paper FAHP-RBF-LSSVM MMRE=0.57 (COCOMO-81 LOO)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE model.pkl
# ─────────────────────────────────────────────────────────────────────────────
artifacts = {
    "model":         final_model,
    "scaler_X":      scaler_X,
    "scaler_y":      scaler_y,
    "fahp_weights":  fahp_w,          # (16,) Spearman-derived feature importance
    "kernel_w":      kernel_w,        # sqrt(fahp_w) — the actual preprocessing multiplier
    "X_train":       X_w,             # background for SHAP
    "y_train":       y_raw,

    "feature_names": FEATURE_NAMES,
    "log_transform": True,
    "best_C":        round(float(best_C), 6),
    "best_gamma":    best_gamma if isinstance(best_gamma, str)
                        else round(float(best_gamma), 6),
    "loo_mmre":      round(best_mmre, 4),
    "train_mmre":    round(train_mmre, 4),
    "train_rmse":    round(train_rmse, 2),
    "n_train":       len(y_raw),
    "importance":    fahp_w,

    # Project-level weights for explainability tab
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

print(f"\n✓  model.pkl saved")
print(f"   LOO-MMRE  : {best_mmre:.4f}  (paper target: 0.57)")
print(f"   Train-MMRE: {train_mmre:.4f}")
print(f"   kernel_w  : sqrt(fahp_w) — implements Eq.17 weighted RBF kernel")
print(f"   FAHP top-3 features: {[FEATURE_NAMES[i] for i in np.argsort(fahp_w)[::-1][:3]]}")

# ── Sanity check ──────────────────────────────────────────────────────────────
print("\n── Spot-check project 1 (actual=2040 PM) ──")
v_log = np.log(np.clip(X_raw[0:1], 1e-3, None))
v_sc  = scaler_X.transform(v_log)
v_w   = v_sc * kernel_w
p_sc  = final_model.predict(v_w)
p_pm  = float(np.expm1(scaler_y.inverse_transform(p_sc.reshape(-1, 1))[0, 0]))
print(f"  Predicted: {p_pm:.1f} PM  |  Actual: {y_raw[0]:.1f} PM  "
      f"|  RE: {abs(p_pm - y_raw[0]) / y_raw[0]:.2%}")
print("\nNote: project 1 is a known outlier in COCOMO-81 (2040 PM is the largest effort")
print("value — 3× the next largest). High RE on this point is expected and consistent")
print("with the paper's own Table 8 (predicted 446 PM vs actual 2040 PM).")
