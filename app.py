import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle, os, shap, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Software Effort Estimator",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace;}

/* Sidebar styling - minimal and safe */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
}
section[data-testid="stSidebar"] > div {
    background: #0d1117 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
}

.main .block-container{padding:1.8rem 2.5rem 2rem 2.5rem;max-width:1400px;}
.main{background:#f6f8fa;}

.hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
    border-radius:12px;padding:2rem 2.5rem;margin-bottom:1.5rem;
    border:1px solid #21262d;}
.hero-title{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;
    font-weight:600;color:#e6edf3;margin:0 0 0.4rem 0;}
.hero-sub{font-size:0.85rem;color:#8b949e;margin:0;line-height:1.6;}
.hero-badge{display:inline-block;background:#1f6feb22;border:1px solid #1f6feb55;
    color:#58a6ff;font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
    padding:0.2rem 0.6rem;border-radius:20px;margin-right:0.4rem;margin-top:0.5rem;}
.hero-badge-green{background:#23863622;border-color:#23863655;color:#3fb950;}
.hero-badge-orange{background:#bb800922;border-color:#bb800955;color:#e3b341;}

.metric-card{background:white;border:1px solid #d0d7de;border-radius:10px;
    padding:1.25rem 1.5rem;box-shadow:0 1px 4px rgba(0,0,0,.06);height:100%;}
.metric-label{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
    color:#57606a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;}
.metric-value{font-family:'IBM Plex Mono',monospace;font-size:2.1rem;
    font-weight:600;color:#0969da;line-height:1;}
.metric-unit{font-family:'IBM Plex Sans',sans-serif;font-size:0.8rem;
    color:#8c959f;margin-top:0.3rem;}

.section-header{font-family:'IBM Plex Mono',monospace;font-size:0.72rem;
    letter-spacing:0.14em;text-transform:uppercase;color:#57606a;
    border-bottom:2px solid #0969da;padding-bottom:0.35rem;margin:2rem 0 1rem 0;}

.insight-box{border-left:3px solid #0969da;background:#ddf4ff;
    padding:0.7rem 1rem;border-radius:0 6px 6px 0;
    margin:0.4rem 0;font-size:0.86rem;color:#0550ae;line-height:1.5;}
.insight-up{border-left-color:#d1242f;background:#fff0eb;color:#6e2000;}
.insight-down{border-left-color:#1a7f37;background:#dafbe1;color:#0d4a1f;}
.insight-note{border-left-color:#9a6700;background:#fff8c5;
    padding:0.7rem 1rem;border-radius:0 6px 6px 0;
    margin:0.4rem 0;font-size:0.82rem;color:#7d4e00;line-height:1.5;}

.pipeline-box{background:white;border:1px solid #d0d7de;border-radius:10px;
    padding:1.5rem 2rem;margin-bottom:1rem;}
.pipeline-step{display:flex;align-items:flex-start;margin-bottom:1.2rem;}
.step-num{background:#0969da;color:white;border-radius:50%;width:26px;height:26px;
    display:flex;align-items:center;justify-content:center;
    font-family:'IBM Plex Mono',monospace;font-size:0.75rem;font-weight:600;
    flex-shrink:0;margin-right:1rem;margin-top:0.1rem;}
.step-title{font-family:'IBM Plex Mono',monospace;font-size:0.85rem;
    font-weight:600;color:#0969da;margin-bottom:0.2rem;}
.step-desc{font-size:0.83rem;color:#57606a;line-height:1.55;}

#MainMenu, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

art      = load_model()
model    = art["model"]
scaler_X = art["scaler_X"]
scaler_y = art["scaler_y"]
fahp_w   = art["fahp_weights"]
kernel_w = art.get("kernel_w", np.sqrt(fahp_w))
X_train  = art["X_train"]

FEATURES = [
    "LOC (KLOC)", "RELY", "DATA", "CPLX", "TIME", "STOR",
    "VIRT", "TURN", "ACAP", "AEXP", "PCAP", "VEXP",
    "LEXP", "MODP", "TOOL", "SCED",
]
FEAT_DESC = {
    "LOC (KLOC)": "Lines of Code (thousands)",
    "RELY": "Required software reliability",
    "DATA": "Database size relative to program size",
    "CPLX": "Process complexity",
    "TIME": "CPU time constraint",
    "STOR": "Main memory constraint",
    "VIRT": "Virtual machine volatility",
    "TURN": "Turnaround time",
    "ACAP": "Analyst capability",
    "AEXP": "Application experience",
    "PCAP": "Programmer capability",
    "VEXP": "Virtual machine experience",
    "LEXP": "Language experience",
    "MODP": "Modern programming practices",
    "TOOL": "Use of software tools",
    "SCED": "Schedule constraint",
}
RATING_LABELS = ["Very Low", "Low", "Nominal", "High", "Very High"]
RATING_MAP    = {lbl: i for i, lbl in enumerate(RATING_LABELS)}
COCOMO_MULT   = {
    "RELY": [0.75, 0.88, 1.00, 1.15, 1.40],
    "DATA": [0.94, 0.94, 1.00, 1.08, 1.16],
    "CPLX": [0.70, 0.85, 1.00, 1.15, 1.30],
    "TIME": [1.00, 1.00, 1.00, 1.11, 1.30],
    "STOR": [1.00, 1.00, 1.00, 1.06, 1.21],
    "VIRT": [0.87, 0.87, 1.00, 1.15, 1.30],
    "TURN": [0.87, 0.87, 1.00, 1.07, 1.15],
    "ACAP": [1.46, 1.19, 1.00, 0.86, 0.71],
    "AEXP": [1.29, 1.13, 1.00, 0.91, 0.82],
    "PCAP": [1.42, 1.17, 1.00, 0.86, 0.70],
    "VEXP": [1.21, 1.10, 1.00, 0.90, 0.87],
    "LEXP": [1.14, 1.07, 1.00, 0.95, 0.95],
    "MODP": [1.24, 1.10, 1.00, 0.91, 0.82],
    "TOOL": [1.24, 1.10, 1.00, 0.91, 0.83],
    "SCED": [1.23, 1.08, 1.00, 1.04, 1.10],
}


# ── Pipeline ─────────────────────────────────────────────────────────────────
def inputs_to_raw(loc, ratings):
    vec = [loc] + [COCOMO_MULT[f][RATING_MAP[ratings[f]]] for f in FEATURES[1:]]
    return np.array(vec, dtype=float).reshape(1, -1)

def preprocess(vec_raw):
    vl = np.log(np.clip(vec_raw, 1e-3, None))
    vs = scaler_X.transform(vl)
    return vs * kernel_w

def predict(vec_raw):
    vw = preprocess(vec_raw)
    ps = model.predict(vw)
    pl = scaler_y.inverse_transform(ps.reshape(-1, 1))[0, 0]
    return max(float(np.expm1(pl)), 1.0), vw

@st.cache_data(show_spinner=False)
def compute_shap(_vec_w_tuple):
    vec_w = np.array(_vec_w_tuple).reshape(1, -1)
    bg    = shap.sample(X_train, min(30, len(X_train)))
    def predict_pm(X):
        ps  = model.predict(X)
        pls = scaler_y.inverse_transform(ps.reshape(-1, 1)).ravel()
        return np.expm1(pls)
    exp  = shap.KernelExplainer(predict_pm, bg)
    sv   = exp.shap_values(vec_w, nsamples=150, silent=True)
    return sv[0], float(exp.expected_value)


# ── Plots ────────────────────────────────────────────────────────────────────
def _style_ax(ax, title, xlabel=None):
    ax.set_title(title, fontsize=10, fontfamily="monospace",
                 fontweight="bold", color="#24292f", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, fontfamily="monospace", color="#57606a")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#d0d7de")
    ax.tick_params(labelsize=8, colors="#57606a")
    ax.set_facecolor("#f6f8fa")

def plot_shap_bar(sv_pm, effort_pm):
    order  = np.argsort(np.abs(sv_pm))[::-1][:10]
    colors = ["#d1242f" if sv_pm[i] > 0 else "#0969da" for i in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f8fa")
    ax.barh([FEATURES[i] for i in order[::-1]], [sv_pm[i] for i in order[::-1]],
            color=colors[::-1], height=0.62, edgecolor="none")
    ax.axvline(0, color="#57606a", lw=0.8, ls="--", alpha=0.6)
    _style_ax(ax, f"Feature Contributions  →  {effort_pm:.0f} PM",
              "SHAP value (person-months)")
    ax.tick_params(axis="y", labelsize=9, colors="#24292f")
    ax.legend(handles=[
        mpatches.Patch(color="#d1242f", label="Increases effort"),
        mpatches.Patch(color="#0969da", label="Decreases effort"),
    ], fontsize=8, frameon=False)
    plt.tight_layout()
    return fig

def plot_fahp_vs_shap(sv_pm):
    sn = np.abs(sv_pm); sn = sn / (sn.sum() + 1e-9)
    fn = fahp_w / fahp_w.sum()
    x, w = np.arange(len(FEATURES)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f8fa")
    ax.bar(x - w/2, fn, width=w, label="FAHP prior weight",
           color="#0969da", alpha=0.85, edgecolor="none")
    ax.bar(x + w/2, sn, width=w, label="SHAP posterior importance",
           color="#d1242f", alpha=0.85, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Normalised weight", fontsize=8, color="#57606a")
    _style_ax(ax, "FAHP Prior vs SHAP Posterior")
    ax.spines["left"].set_color("#d0d7de")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return fig

def plot_waterfall(sv_pm, base_pm, effort_pm):
    order  = np.argsort(np.abs(sv_pm))[::-1][:8]
    labels = [FEATURES[i] for i in order]
    vals   = [sv_pm[i]    for i in order]
    running = base_pm
    starts, widths, colors = [], [], []
    for v in vals:
        starts.append(min(running, running + v))
        widths.append(abs(v))
        colors.append("#d1242f" if v > 0 else "#0969da")
        running += v
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f6f8fa")
    ax.barh(labels[::-1], widths[::-1], left=starts[::-1],
            color=colors[::-1], height=0.6, edgecolor="none")
    ax.axvline(base_pm,   color="#8b949e", lw=1.2, ls=":",
               label=f"Baseline {base_pm:.0f} PM")
    ax.axvline(effort_pm, color="#24292f", lw=1.5,
               label=f"Prediction {effort_pm:.0f} PM")
    _style_ax(ax, "Cumulative Contribution Waterfall", "Person-months")
    ax.tick_params(axis="y", colors="#24292f", labelsize=9)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


# ── What-if ──────────────────────────────────────────────────────────────────
def whatif(base_raw, base_effort, ratings):
    improvable = {
        "ACAP": ("Increase analyst capability",    -1),
        "AEXP": ("Increase application experience",-1),
        "PCAP": ("Increase programmer capability", -1),
        "CPLX": ("Reduce complexity",              -1),
        "TOOL": ("Adopt better tools",             -1),
        "MODP": ("Improve modern practices",       -1),
        "RELY": ("Lower reliability requirement",  -1),
        "TIME": ("Relax time constraint",          -1),
        "STOR": ("Relax memory constraint",        -1),
    }
    results = []
    for feat, (label, direction) in improvable.items():
        cur = RATING_MAP[ratings[feat]]
        nxt = cur + direction
        if 0 <= nxt < 5:
            nr = ratings.copy(); nr[feat] = RATING_LABELS[nxt]
            ne, _ = predict(inputs_to_raw(base_raw[0, 0], nr))
            saving = base_effort - ne
            if saving > 0.5:
                results.append(dict(action=label, feature=feat,
                    from_=RATING_LABELS[cur], to=RATING_LABELS[nxt],
                    saving=saving, pct=saving / base_effort * 100))
    return sorted(results, key=lambda r: r["saving"], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## `SEE`")
    st.markdown("<p style='color:#8b949e;font-size:0.8rem;margin-top:-0.5rem;'>Software Effort Estimator</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p style='color:#8b949e;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;'>Project Scale</p>", unsafe_allow_html=True)
    loc = st.slider("Lines of Code (KLOC)", 1, 500, 50)

    st.markdown("<p style='color:#8b949e;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.5rem;'>Product</p>", unsafe_allow_html=True)
    rely = st.selectbox("RELY · Reliability",    RATING_LABELS, index=2)
    data = st.selectbox("DATA · Database Size",  RATING_LABELS, index=2)
    cplx = st.selectbox("CPLX · Complexity",     RATING_LABELS, index=2)

    st.markdown("<p style='color:#8b949e;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.5rem;'>Computer</p>", unsafe_allow_html=True)
    time_ = st.selectbox("TIME · Time Constraint",   RATING_LABELS, index=2)
    stor  = st.selectbox("STOR · Memory Constraint", RATING_LABELS, index=2)
    virt  = st.selectbox("VIRT · VM Volatility",     RATING_LABELS, index=2)
    turn  = st.selectbox("TURN · Turnaround",        RATING_LABELS, index=2)

    st.markdown("<p style='color:#8b949e;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.5rem;'>Personnel</p>", unsafe_allow_html=True)
    acap = st.selectbox("ACAP · Analyst Capability",    RATING_LABELS, index=2)
    aexp = st.selectbox("AEXP · App Experience",        RATING_LABELS, index=2)
    pcap = st.selectbox("PCAP · Programmer Capability", RATING_LABELS, index=2)
    vexp = st.selectbox("VEXP · VM Experience",         RATING_LABELS, index=2)
    lexp = st.selectbox("LEXP · Language Experience",   RATING_LABELS, index=2)

    st.markdown("<p style='color:#8b949e;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.5rem;'>Project</p>", unsafe_allow_html=True)
    modp = st.selectbox("MODP · Modern Practices",    RATING_LABELS, index=2)
    tool = st.selectbox("TOOL · Software Tools",      RATING_LABELS, index=2)
    sced = st.selectbox("SCED · Schedule Constraint", RATING_LABELS, index=2)

    st.markdown("---")
    run = st.button("▶  Estimate Effort", use_container_width=True, type="primary")

ratings = dict(
    RELY=rely, DATA=data, CPLX=cplx, TIME=time_, STOR=stor,
    VIRT=virt, TURN=turn, ACAP=acap, AEXP=aexp, PCAP=pcap,
    VEXP=vexp, LEXP=lexp, MODP=modp, TOOL=tool, SCED=sced,
)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-title">📐 Software Effort Estimator</div>
  <div class="hero-sub">
    Explainable AI for software project planning &nbsp;·&nbsp;
    FAHP · Weighted-Kernel SVR · SHAP
  </div>
  <div style="margin-top:0.6rem;">
    <span class="hero-badge">LOO-MMRE {art['loo_mmre']:.4f}</span>
    <span class="hero-badge hero-badge-green">n=63 projects</span>
    <span class="hero-badge hero-badge-orange">COCOMO-81 benchmark</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Landing state: show pipeline explanation ──────────────────────────────────
if not run:
    st.markdown("""
    <div class="pipeline-box">
      <div class="pipeline-step">
        <div class="step-num">1</div>
        <div>
          <div class="step-title">FAHP Feature Weighting</div>
          <div class="step-desc">Fuzzy Analytic Hierarchy Process (Chang 1996 extent analysis) builds pairwise comparison matrices from project effort and LOC. Consistency ratio verified &lt; 0.1. Project weights are mapped to feature importance via Spearman correlation, then embedded into the kernel as S = diag(√θ).</div>
        </div>
      </div>
      <div class="pipeline-step">
        <div class="step-num">2</div>
        <div>
          <div class="step-title">Weighted-Kernel SVR Prediction</div>
          <div class="step-desc">Support Vector Regression with RBF kernel trained in log-space on COCOMO-81 (n=63). Pre-multiplying features by √(FAHP weights) implements the weighted kernel K(θx_k, θx_l) from Eq. 17. Hyperparameters C and γ tuned via Leave-One-Out MMRE grid search — achieving 0.4619 vs paper baseline of 0.57.</div>
        </div>
      </div>
      <div class="pipeline-step" style="margin-bottom:0;">
        <div class="step-num">3</div>
        <div>
          <div class="step-title">SHAP Explainability Layer</div>
          <div class="step-desc">KernelExplainer decomposes every prediction into signed per-feature contributions in person-month units. Enables direct comparison between FAHP prior weights (expert-derived) and SHAP posterior importance (data-driven) — revealing where the model agrees or diverges from expert assumptions.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#8c959f;font-size:0.82rem;text-align:center;margin-top:1rem;'>
        ← Configure project parameters in the sidebar, then click <strong>Estimate Effort</strong>
    </p>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
vec_raw       = inputs_to_raw(loc, ratings)
effort, vec_w = predict(vec_raw)
eaf = float(np.prod([COCOMO_MULT[f][RATING_MAP[ratings[f]]] for f in COCOMO_MULT]))

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, label, value, unit in [
    (c1, "Predicted Effort",     f"{effort:.0f}",              "person-months"),
    (c2, "Calendar Time",        f"{effort/12:.1f}",            "person-years"),
    (c3, "Recommended Team",     f"{max(1,round(effort/18))}",  "engineers"),
    (c4, "Effort Adj. Factor",   f"{eaf:.3f}",                  "combined EAF"),
]:
    col.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        <div class='metric-unit'>{unit}</div>
    </div>""", unsafe_allow_html=True)

# ── SHAP ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Explainability</div>", unsafe_allow_html=True)

with st.spinner("Computing SHAP values…"):
    sv_pm, base_pm = compute_shap(tuple(vec_w.ravel().tolist()))

tab1, tab2, tab3 = st.tabs(["Feature Contributions", "FAHP vs SHAP", "Waterfall"])
with tab1:
    st.pyplot(plot_shap_bar(sv_pm, effort), use_container_width=True)
with tab2:
    st.pyplot(plot_fahp_vs_shap(sv_pm), use_container_width=True)
with tab3:
    st.pyplot(plot_waterfall(sv_pm, base_pm, effort), use_container_width=True)

# ── Insights ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)

top_pos = [(FEATURES[i], sv_pm[i]) for i in np.argsort(sv_pm)[::-1] if sv_pm[i] > 0][:3]
top_neg = [(FEATURES[i], sv_pm[i]) for i in np.argsort(sv_pm)       if sv_pm[i] < 0][:2]

ic1, ic2 = st.columns(2)
with ic1:
    st.markdown("**Effort drivers — pushing up**")
    for feat, val in top_pos:
        st.markdown(f"""<div class='insight-box insight-up'>
            <strong>{feat}</strong> &nbsp;·&nbsp; {FEAT_DESC.get(feat,feat)}<br>
            <span style='font-family:monospace;'>+{val:.1f} PM</span> above baseline
        </div>""", unsafe_allow_html=True)
with ic2:
    st.markdown("**Effort reducers — pushing down**")
    for feat, val in top_neg:
        st.markdown(f"""<div class='insight-box insight-down'>
            <strong>{feat}</strong> &nbsp;·&nbsp; {FEAT_DESC.get(feat,feat)}<br>
            <span style='font-family:monospace;'>{val:.1f} PM</span> below baseline
        </div>""", unsafe_allow_html=True)

# ── FAHP vs SHAP alignment ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>FAHP Prior vs SHAP Posterior — Alignment</div>",
            unsafe_allow_html=True)

st.markdown("""<div class='insight-note'>
    FAHP weights encode <strong>expert prior beliefs</strong> about feature importance.
    SHAP values capture <strong>what the model actually learned</strong> from data.
    Divergence between the two is itself a finding — it reveals where expert assumptions
    and historical patterns disagree.
</div>""", unsafe_allow_html=True)

f3 = np.argsort(fahp_w)[::-1][:3]
s3 = np.argsort(np.abs(sv_pm))[::-1][:3]
agree = len(set(f3) & set(s3)) / 3 * 100
colour = "#1a7f37" if agree >= 67 else ("#9a6700" if agree >= 33 else "#d1242f")
label  = "Strong agreement" if agree >= 67 else ("Partial" if agree >= 33 else "Disagreement")

ac1, ac2 = st.columns([2.5, 1])
with ac1:
    st.dataframe(pd.DataFrame({
        "Rank":             ["#1", "#2", "#3"],
        "FAHP (prior)":     [FEATURES[i] for i in f3],
        "SHAP (posterior)": [FEATURES[i] for i in s3],
        "Match":            ["✅" if f3[i] == s3[i] else ("⚠️" if f3[i] in s3 else "❌") for i in range(3)],
    }), use_container_width=True, hide_index=True)
with ac2:
    st.markdown(f"""<div class='metric-card' style='border-color:{colour};'>
        <div class='metric-label'>Alignment Score</div>
        <div class='metric-value' style='color:{colour};'>{agree:.0f}%</div>
        <div class='metric-unit'>{label}</div>
    </div>""", unsafe_allow_html=True)

# ── What-if ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>What-If Analysis — Reduction Opportunities</div>",
            unsafe_allow_html=True)
wi = whatif(vec_raw, effort, ratings)
if wi:
    st.dataframe(pd.DataFrame([{
        "Action":      r["action"], "Feature": r["feature"],
        "Change":      f"{r['from_']} → {r['to']}",
        "Saves (PM)":  f"{r['saving']:.1f}",
        "% Reduction": f"{r['pct']:.1f}%",
    } for r in wi]), use_container_width=True, hide_index=True)
    b = wi[0]
    st.markdown(f"""<div class='insight-box insight-down'>
        💡 <strong>Top recommendation:</strong> {b['action']}
        &nbsp;({b['feature']}: {b['from_']} → {b['to']})&nbsp;
        saves <strong>{b['saving']:.0f} PM ({b['pct']:.1f}%)</strong>
    </div>""", unsafe_allow_html=True)
else:
    st.info("No improvement opportunities — already at optimal settings.")

# ── Expandable details ────────────────────────────────────────────────────────
with st.expander("📋 COCOMO Cost Driver Reference"):
    rows = [{"Feature": f, "Description": FEAT_DESC.get(f,"")}
            | {lbl: COCOMO_MULT[f][i] for i, lbl in enumerate(RATING_LABELS)}
            for f in COCOMO_MULT]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with st.expander("🔬 Model Details"):
    m1, m2, m3 = st.columns(3)
    m1.metric("LOO-CV MMRE", f"{art['loo_mmre']:.4f}",
              delta=f"{art['loo_mmre']-0.57:+.4f} vs baseline 0.57", delta_color="inverse")
    m2.metric("Train MMRE",  f"{art['train_mmre']:.4f}")
    m3.metric("Train RMSE",  f"{art['train_rmse']:.1f} PM")
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Kernel | RBF (weighted, Eq. 17) |
    | C | {art['best_C']} |
    | γ | {art['best_gamma']} |
    | ε | 0.1 |
    | Training samples | {art['n_train']} |
    | Validation | Leave-One-Out CV |
    """)
    st.dataframe(pd.DataFrame({
        "Feature":   FEATURES,
        "FAHP θ":    [f"{w:.5f}" for w in fahp_w],
        "Kernel √θ": [f"{w:.5f}" for w in kernel_w],
        "Rank":      [str(r+1) for r in np.argsort(np.argsort(-fahp_w))],
    }), use_container_width=True, hide_index=True)

st.markdown("""
<hr style='border:none;border-top:1px solid #d0d7de;margin-top:2.5rem;'>
<p style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#8c959f;text-align:center;'>
    FAHP (Chang 1996) · Weighted-Kernel SVR · SHAP KernelExplainer · COCOMO-81
</p>""", unsafe_allow_html=True)