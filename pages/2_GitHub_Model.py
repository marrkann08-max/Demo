import streamlit as st
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")


# ── Shared CSS (minimal) ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: "Inter", system-ui, sans-serif; }
.metric-card {
    background:#fff;border:1px solid #e2e2e2;border-radius:10px;
    padding:1.1rem 1.3rem;height:100%;
}
.metric-label { font-size:0.68rem;color:#666;letter-spacing:0.07em;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem; }
.metric-value { font-family:"JetBrains Mono",monospace;font-size:2.2rem;font-weight:600;color:#111;line-height:1; }
.metric-unit  { font-size:0.78rem;color:#888;margin-top:0.28rem; }
.metric-range { font-family:"JetBrains Mono",monospace;font-size:0.68rem;color:#888;margin-top:0.4rem;padding-top:0.4rem;border-top:1px solid #f0f0f0; }
.section-header {
    font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
    font-weight:700;color:#555;margin:1.8rem 0 0.8rem;
    display:flex;align-items:center;gap:0.6rem;
}
.section-header::after { content:"";flex:1;height:1px;background:#ebebeb; }
.explain-note { background:#fff;border:1px solid #e2e2e2;border-left:2px solid #aaa;border-radius:0 6px 6px 0;padding:0.65rem 0.9rem;margin:0.5rem 0 0.9rem; }
.explain-label { font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;color:#aaa;margin-bottom:0.32rem; }
.explain-text  { font-size:0.8rem;color:#444;line-height:1.65; }
.data-table { width:100%;border-collapse:collapse;font-size:0.8rem; }
.data-table thead tr { background:#f7f7f7;border-bottom:1px solid #e2e2e2; }
.data-table thead th { padding:6px 10px;font-size:0.66rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#777; }
.data-table tbody td { padding:6px 10px;border-bottom:1px solid #f0f0f0;color:#1a1a1a; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_path = os.path.join(ROOT, "github_model.pkl")

@st.cache_data(show_spinner=False)
def load(mtime):
    with open(os.path.join(ROOT, "github_model.pkl"), "rb") as f:
        return pickle.load(f)

if not os.path.exists(_path):
    st.error("github_model.pkl not found. Run `python train_github_model.py`.")
    st.stop()

art = load(os.path.getmtime(_path))
gh_model    = art["model"]
gh_scaler_X = art["scaler_X"]
gh_scaler_y = art["scaler_y"]
gh_kernel_w = art["kernel_w"]
gh_fahp_w   = art["fahp_weights"]
gh_feat     = art["feature_names"]
gh_pi_lo    = art.get("pi_lo_pct", -0.25)
gh_pi_hi    = art.get("pi_hi_pct",  0.33)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding-bottom:1.2rem;border-bottom:1px solid #e2e2e2;margin-bottom:1.5rem;'>
  <div style='font-size:1.1rem;font-weight:650;color:#111;letter-spacing:-0.02em;'>GitHub-Trained Effort Model</div>
  <div style='font-size:0.79rem;color:#666;margin-top:0.2rem;'>
    FAHP + Weighted-Kernel SVR &nbsp;·&nbsp; Trained on 91 real GitHub projects (2021–present)
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""<div class='explain-note'>
  <div class='explain-label'>What this model is</div>
  <div class='explain-text'>
    Same methodology as the COCOMO-81 model — FAHP feature weighting + weighted-kernel SVR —
    but trained on <strong>real GitHub open-source projects</strong> using signals your team
    already generates: PR review time, commit frequency, team size, test coverage.
    <br><br>
    Effort is proxied from commit author-days (unique contributor × date pairs ÷ 20).
    Open-source volunteers differ from paid engineers — this is a proof of concept.
    <strong>With your company's actual Jira-tracked hours, MMRE would be lower.</strong>
  </div>
</div>""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Project Signals</div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    gh_contributors = st.slider("Team size (contributors)", 5, 400, 50,
        help="Unique people who committed to the project")
    gh_review_days  = st.slider("Avg PR review time (days)", 0.0, 15.0, 2.0, 0.1,
        help="Median time from PR opened to merged — slower = less capable team")
    gh_merge_rate   = st.slider("PR merge rate", 0.1, 1.0, 0.65, 0.01,
        help="Fraction of opened PRs that get merged")
with c2:
    gh_test_ratio   = st.slider("Test file ratio", 0.0, 0.6, 0.20, 0.01,
        help="Fraction of repo files that are test/spec files")
    gh_lang_count   = st.slider("Language count", 1, 20, 4,
        help="Number of programming languages in the repo")
    gh_commit_freq  = st.slider("Commit frequency (per week)", 0.5, 150.0, 10.0, 0.5,
        help="Average commits per week over the project window")

# ── Predict ───────────────────────────────────────────────────────────────────
x_raw = np.array([[gh_contributors, gh_review_days, gh_merge_rate,
                   gh_test_ratio, gh_lang_count, gh_commit_freq]], dtype=float)
x_log = np.log(np.clip(x_raw, 1e-3, None))
x_sc  = gh_scaler_X.transform(x_log)
x_w   = x_sc * gh_kernel_w
p_sc  = gh_model.predict(x_w)
p_log = gh_scaler_y.inverse_transform(p_sc.reshape(-1,1))[0,0]
effort = max(float(np.expm1(p_log)), 1.0)
lo     = max(1.0, effort * (1 + gh_pi_lo))
hi     = effort * (1 + gh_pi_hi)

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Prediction</div>", unsafe_allow_html=True)
r1, r2, r3, r4 = st.columns(4)
r1.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Predicted Effort</div>
    <div class='metric-value'>{effort:.0f}</div>
    <div class='metric-unit'>author-months</div>
    <div class='metric-range'>80% range: {lo:.0f}–{hi:.0f} PM</div>
</div>""", unsafe_allow_html=True)
r2.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>LOO-MMRE</div>
    <div class='metric-value' style='color:#2563EB;'>{art['loo_mmre']:.4f}</div>
    <div class='metric-unit'>leave-one-out CV</div>
    <div class='metric-range'>PRED(25) = {art['pred25']*100:.0f}%</div>
</div>""", unsafe_allow_html=True)
r3.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>FAHP Improvement</div>
    <div class='metric-value' style='color:#15803d;font-size:1.8rem;'>{art['fahp_improve_pct']:.0f}%</div>
    <div class='metric-unit'>vs no FAHP weighting</div>
    <div class='metric-range'>Ablation: {art['ablation_mmre']:.4f} → {art['loo_mmre']:.4f}</div>
</div>""", unsafe_allow_html=True)
r4.markdown(f"""<div class='metric-card'>
    <div class='metric-label'>Training Repos</div>
    <div class='metric-value' style='font-size:1.8rem;'>{art['n_projects']}</div>
    <div class='metric-unit'>open-source projects</div>
    <div class='metric-range'>GitHub API, 2021–present</div>
</div>""", unsafe_allow_html=True)

# ── FAHP weights ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>FAHP Feature Weights — What Drives Effort</div>",
            unsafe_allow_html=True)
order = np.argsort(gh_fahp_w)[::-1]
rows  = "".join(
    f"<tr><td><strong>{gh_feat[i]}</strong></td>"
    f"<td style='font-family:monospace;'>{gh_fahp_w[i]:.4f}</td>"
    f"<td style='width:40%;'><div style='background:#2563EB;height:8px;border-radius:4px;"
    f"width:{gh_fahp_w[i]/gh_fahp_w.max()*100:.0f}%;'></div></td></tr>"
    for i in order
)
st.markdown(f"""<table class='data-table'>
  <thead><tr><th>Feature</th><th>Weight</th><th>Relative Importance</th></tr></thead>
  <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)

st.markdown("""<div class='explain-note' style='margin-top:1rem;'>
  <div class='explain-label'>Interpretation</div>
  <div class='explain-text'>
    <strong>Contributors</strong> is the dominant predictor — more people = more coordination overhead = more effort.
    <strong>Commit Frequency</strong> captures active development intensity.
    <strong>Test Coverage Ratio</strong> reflects engineering maturity.
    On your company's data, the ranking would shift to reflect your specific team dynamics.
  </div>
</div>""", unsafe_allow_html=True)
