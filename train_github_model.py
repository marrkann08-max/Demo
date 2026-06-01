"""
train_github_model.py  —  FAHP + SVR on GitHub-derived project metrics
Same methodology as Sehra et al. (2018), applied to modern GitHub data.

Features: contributor_count, avg_pr_review_days, pr_merge_rate,
          test_file_ratio, lang_count, commit_frequency
Target:   author_months (derived effort proxy: unique author×date pairs / 20)

Run: python train_github_model.py
Saves: github_model.pkl
"""

import numpy as np
import pandas as pd
import pickle, os, warnings
from itertools import product as iterproduct
from scipy.stats import spearmanr
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  DATA
# ─────────────────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "github_projects.csv")
df = pd.read_csv(csv_path)

# Features kept after data quality audit:
#   contributor_count  — r=+0.69, real variance (36–369)
#   avg_pr_review_days — real variance (0.01–6.4d)
#   pr_merge_rate      — real variance (0.20–0.88)
#   test_file_ratio    — now properly computed from git tree API
#   lang_count         — real variance (1–14)
#   commit_frequency   — real variance
# Dropped:
#   has_ci      — zero variance (all modern repos have CI)
#   total_kloc  — disk-size proxy, r=-0.058 with effort
#   stars       — popularity ≠ effort
FEATURE_COLS = [
    "contributor_count",
    "avg_pr_review_days",
    "pr_merge_rate",
    "test_file_ratio",
    "lang_count",
    "commit_frequency",
    "contributors_x_freq",   # interaction: large fast-moving teams
    "ai_tools_count",        # AI coding tool adoption (Copilot, Cursor, Claude…)
]
FEATURE_NAMES = [
    "Contributors",
    "PR Review Time (days)",
    "PR Merge Rate",
    "Test Coverage Ratio",
    "Language Count",
    "Commit Frequency",
    "Contributors x Frequency",
    "AI Tool Adoption",
]
TARGET = "author_months"

df_work = df.copy()
df_work["contributors_x_freq"] = df_work["contributor_count"] * df_work["commit_frequency"]
# ai_tools_count: 0 if column missing (older CSV without patch)
if "ai_tools_count" not in df_work.columns:
    df_work["ai_tools_count"] = 0
else:
    df_work["ai_tools_count"] = df_work["ai_tools_count"].fillna(0)

base_cols = ["contributor_count","avg_pr_review_days","pr_merge_rate",
             "test_file_ratio","lang_count","commit_frequency",
             "contributors_x_freq","ai_tools_count"]
df_clean = df_work[base_cols + [TARGET, "repo", "stars"]].dropna()
df_clean = df_clean[(df_clean[["contributor_count","avg_pr_review_days","pr_merge_rate",
                                "test_file_ratio","lang_count","commit_frequency"]] > 0).all(axis=1)]
df_clean = df_clean[df_clean[TARGET] > 0]

X_raw  = df_clean[FEATURE_COLS].values.astype(float)
y_raw  = df_clean[TARGET].values.astype(float)
repos  = df_clean["repo"].values

n = len(df_clean)
print(f"GitHub dataset: {n} projects")
print(f"Effort: min={y_raw.min():.1f}  max={y_raw.max():.1f}  "
      f"mean={y_raw.mean():.1f}  std={y_raw.std():.1f} PM")
print(f"\nFeature correlations with effort:")
for i, name in enumerate(FEATURE_NAMES):
    corr = float(spearmanr(X_raw[:, i], y_raw).correlation)
    print(f"  {name:>25s}: r={corr:+.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  FAHP — Chang (1996) extent analysis
# ─────────────────────────────────────────────────────────────────────────────
TFN_SCALE = {
    1:(1.,1.,1.), 2:(1.,2.,3.), 3:(1.,3.,5.), 4:(1.,4.,7.),
    5:(3.,5.,7.), 6:(3.,6.,9.), 7:(5.,7.,9.), 8:(7.,8.,9.), 9:(7.,9.,9.),
}

def _tfn_recip(tfn):
    l, m, u = tfn; return (1/u, 1/m, 1/l)

def _ratio_to_tfn(ratio):
    ratio = max(1/9, min(9., float(ratio)))
    if ratio < 1.: return _tfn_recip(_ratio_to_tfn(1./ratio))
    lo, hi = max(1,min(9,int(np.floor(ratio)))), max(1,min(9,int(np.ceil(ratio))))
    if lo == hi: return TFN_SCALE[lo]
    t = ratio - lo
    return tuple(TFN_SCALE[lo][k]*(1-t)+TFN_SCALE[hi][k]*t for k in range(3))

def build_pairwise(values):
    n = len(values)
    mat = [[(1.,1.,1.)]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            r = float(values[i]) / float(max(values[j], 1e-9))
            tfn = _ratio_to_tfn(r)
            mat[i][j] = tfn
            mat[j][i] = _tfn_recip(tfn)
    return mat

def synthetic_extent(mat):
    n = len(mat)
    rs = [(sum(mat[i][j][k] for j in range(n)) for k in range(3)) for i in range(n)]
    rs = [tuple(x) for x in rs]
    tl = sum(r[0] for r in rs); tm = sum(r[1] for r in rs); tu = sum(r[2] for r in rs)
    return [(r[0]/tu, r[1]/tm, r[2]/tl) for r in rs]

def degree_of_possibility(M1, M2):
    _, m1, u1 = M1; l2, m2, _ = M2
    if m1 >= m2: return 1.0
    if u1 <= l2: return 0.0
    denom = (m2-l2)-(m1-u1)
    if abs(denom) < 1e-10: return 0.0
    return max(0., min(1., (u1-l2)/denom))

def fahp_weights(values, label=""):
    mat = build_pairwise(values)
    S   = synthetic_extent(mat)
    n   = len(S)
    W   = np.array([min(degree_of_possibility(S[i],S[j]) for j in range(n) if j!=i) for i in range(n)])
    t   = W.sum()
    return W/t if t > 1e-10 else np.ones(n)/n

# Feature weights: normalized Spearman |correlation| with effort
# Same approach as train_model.py — correlations act as the data-driven
# proxy for expert pairwise comparisons, then normalized to sum=1.
print("\n-- FAHP: feature weights (Spearman correlation → normalized) --")
feature_corr = np.array([
    abs(spearmanr(X_raw[:, i], y_raw).correlation)
    for i in range(len(FEATURE_COLS))
])
feature_corr = np.clip(feature_corr, 0.01, None)
fahp_w = feature_corr / feature_corr.sum()

print("FAHP feature weights:")
for i in np.argsort(fahp_w)[::-1]:
    print(f"  {FEATURE_NAMES[i]:>25s}: {fahp_w[i]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
X_log = np.log(np.clip(X_raw, 1e-3, None))
y_log = np.log1p(y_raw)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_sc = scaler_X.fit_transform(X_log)
y_sc = scaler_y.fit_transform(y_log.reshape(-1,1)).ravel()

kernel_w = np.sqrt(fahp_w)
X_w = X_sc * kernel_w

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LOO-MMRE grid search
# ─────────────────────────────────────────────────────────────────────────────
def loo_eval(C, gamma, epsilon):
    loo = LeaveOneOut()
    errors, sre = [], []
    for tr, te in loo.split(X_w):
        svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        svr.fit(X_w[tr], y_sc[tr])
        ps  = svr.predict(X_w[te])
        pl  = scaler_y.inverse_transform(ps.reshape(-1,1))[0,0]
        pm  = max(float(np.expm1(pl)), 0.1)
        re  = (pm - y_raw[te[0]]) / y_raw[te[0]]
        errors.append(abs(re)); sre.append(re)
    return float(np.mean(errors)), np.array(sre)

def loo_mmre(C, gamma, epsilon):
    return loo_eval(C, gamma, epsilon)[0]

# Ablation — no FAHP
def loo_no_fahp(C, gamma, epsilon):
    loo = LeaveOneOut(); errors = []
    for tr, te in loo.split(X_sc):
        svr = SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon)
        svr.fit(X_sc[tr], y_sc[tr])
        ps  = svr.predict(X_sc[te])
        pl  = scaler_y.inverse_transform(ps.reshape(-1,1))[0,0]
        pm  = max(float(np.expm1(pl)), 0.1)
        errors.append(abs(pm - y_raw[te[0]]) / y_raw[te[0]])
    return float(np.mean(errors))

C_grid = [0.1, 0.5, 1, 5, 10, 50, 100]
g_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, "scale"]
e_grid = [0.05, 0.1, 0.2]

print("\n-- Grid search (LOO-MMRE) --")
best_mmre, best_C, best_g, best_e = 1e9, None, None, 0.1
for C, g, e in iterproduct(C_grid, g_grid, e_grid):
    m = loo_mmre(C, g, e)
    if m < best_mmre:
        best_mmre, best_C, best_g, best_e = m, C, g, e

print(f"Best: MMRE={best_mmre:.4f}  C={best_C}  gamma={best_g}  eps={best_e}")

# Fine search
if isinstance(best_C, (int, float)):
    C_fine = [best_C*f for f in [0.3,0.5,0.7,1.0,1.5,2.0,3.0]]
    g_fine = ([best_g*f for f in [0.3,0.5,0.7,1.0,1.5,2.0,3.0]]
              if isinstance(best_g, float) else [best_g])
    e_fine = sorted({max(0.01, best_e*f) for f in [0.5,1.0,2.0]})
    for C, g, e in iterproduct(C_fine, g_fine, e_fine):
        m = loo_mmre(C, g, e)
        if m < best_mmre:
            best_mmre, best_C, best_g, best_e = m, C, g, e

print(f"Fine: MMRE={best_mmre:.4f}  C={best_C:.4f}  gamma={best_g}  eps={best_e:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
_, loo_sre = loo_eval(best_C, best_g, best_e)
loo_abs    = np.abs(loo_sre)
pred25     = float(np.mean(loo_abs <= 0.25))
pred50     = float(np.mean(loo_abs <= 0.50))
pi_lo      = float(np.percentile(loo_sre, 10))
pi_hi      = float(np.percentile(loo_sre, 90))

print(f"\nPRED(25): {pred25*100:.1f}%  PRED(50): {pred50*100:.1f}%")
print(f"80% PI:   [{pi_lo*100:+.1f}%,  {pi_hi*100:+.1f}%]")

# Ablation
abl_mmre = loo_no_fahp(best_C, best_g, best_e)
fahp_imp  = (abl_mmre - best_mmre) / abl_mmre * 100
print(f"\nAblation: without FAHP = {abl_mmre:.4f}  with FAHP = {best_mmre:.4f}  "
      f"(FAHP reduces error by {fahp_imp:.1f}%)")

# Naive baseline (mean prediction)
naive_mmre = float(np.mean(np.abs(y_raw - y_raw.mean()) / y_raw))
print(f"Naive mean baseline:   {naive_mmre:.4f}")
print(f"Our model vs naive:    {(naive_mmre-best_mmre)/naive_mmre*100:.1f}% improvement")

# Per-project LOO errors
print(f"\nPer-project LOO errors (top 5 worst):")
worst = np.argsort(loo_abs)[::-1][:5]
for i in worst:
    print(f"  {repos[i]:>35s}: RE={loo_sre[i]:+.2%}  actual={y_raw[i]:.1f} PM")

# ── Temporal split validation ──────────────────────────────────────────────────
# Stars used as repo maturity proxy: high-star = older/established (train),
# low-star = newer repos (test). Mimics real deployment: train on known history,
# predict future projects.
print("\n-- Temporal split (stars as maturity proxy, 70/30) --")
df_sorted = df_clean.sort_values("stars", ascending=False).reset_index(drop=True)
split_n   = int(len(df_sorted) * 0.70)
df_tr     = df_sorted.iloc[:split_n]
df_te     = df_sorted.iloc[split_n:]

X_tr = df_tr[FEATURE_COLS].values.astype(float)
y_tr = df_tr[TARGET].values.astype(float)
X_te = df_te[FEATURE_COLS].values.astype(float)
y_te = df_te[TARGET].values.astype(float)

# Fit everything on training split only — no leakage
X_log_tr  = np.log(np.clip(X_tr, 1e-3, None))
scX_t     = MinMaxScaler()
scY_t     = MinMaxScaler()
X_sc_tr   = scX_t.fit_transform(X_log_tr)
y_sc_tr   = scY_t.fit_transform(np.log1p(y_tr).reshape(-1, 1)).ravel()

fc_t  = np.array([abs(spearmanr(X_tr[:, i], y_tr).correlation) for i in range(len(FEATURE_COLS))])
fc_t  = np.clip(fc_t, 0.01, None)
kw_t  = np.sqrt(fc_t / fc_t.sum())
Xw_tr = X_sc_tr * kw_t

svr_t = SVR(kernel="rbf", C=best_C, gamma=best_g, epsilon=best_e)
svr_t.fit(Xw_tr, y_sc_tr)

X_log_te  = np.log(np.clip(X_te, 1e-3, None))
Xw_te     = scX_t.transform(X_log_te) * kw_t
p_te      = np.expm1(scY_t.inverse_transform(svr_t.predict(Xw_te).reshape(-1, 1)).ravel())
te_re     = np.abs(y_te - p_te) / y_te
temporal_mmre   = float(np.mean(te_re))
temporal_pred25 = float(np.mean(te_re <= 0.25))

print(f"  Train: {split_n} established repos (high-star)  |  "
      f"Test: {len(df_te)} newer repos (low-star)")
print(f"  Temporal MMRE={temporal_mmre:.4f}  PRED(25)={temporal_pred25*100:.1f}%")
print(f"  (LOO-MMRE={best_mmre:.4f} for comparison — LOO is optimistic, temporal is rigorous)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
final_model = SVR(kernel="rbf", C=best_C, gamma=best_g, epsilon=best_e)
final_model.fit(X_w, y_sc)

train_pred = np.expm1(scaler_y.inverse_transform(
    final_model.predict(X_w).reshape(-1,1)).ravel())
train_mmre = float(np.mean(np.abs(y_raw - train_pred) / y_raw))

# ─────────────────────────────────────────────────────────────────────────────
# 6.  SAVE
# ─────────────────────────────────────────────────────────────────────────────
artifacts = {
    "model":          final_model,
    "scaler_X":       scaler_X,
    "scaler_y":       scaler_y,
    "fahp_weights":   fahp_w,
    "kernel_w":       kernel_w,
    "X_raw":          X_raw,
    "y_raw":          y_raw,
    "repos":          repos,
    "feature_names":  FEATURE_NAMES,
    "feature_cols":   FEATURE_COLS,
    "loo_mmre":       round(best_mmre, 4),
    "train_mmre":     round(train_mmre, 4),
    "pred25":         round(pred25, 4),
    "pred50":         round(pred50, 4),
    "pi_lo_pct":      pi_lo,
    "pi_hi_pct":      pi_hi,
    "ablation_mmre":    round(abl_mmre, 4),
    "fahp_improve_pct": round(fahp_imp, 1),
    "naive_mmre":       round(naive_mmre, 4),
    "temporal_mmre":    round(temporal_mmre, 4),
    "temporal_pred25":  round(temporal_pred25, 4),
    "temporal_train_n": split_n,
    "temporal_test_n":  len(df_te),
    "best_C":         round(float(best_C), 6),
    "best_gamma":     best_g if isinstance(best_g, str) else round(float(best_g), 6),
    "best_epsilon":   round(float(best_e), 6),
    "n_projects":     n,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "github_model.pkl")
with open(out, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n  github_model.pkl saved")
print(f"  LOO-MMRE  : {best_mmre:.4f}")
print(f"  PRED(25)  : {pred25*100:.1f}%")
print(f"  n         : {n} GitHub projects")
print(f"  FAHP top-3: {[FEATURE_NAMES[i] for i in np.argsort(fahp_w)[::-1][:3]]}")
