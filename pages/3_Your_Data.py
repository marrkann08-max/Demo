import streamlit as st


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: "Inter", system-ui, sans-serif; }
.metric-card { background:#fff;border:1px solid #e2e2e2;border-radius:10px;padding:1.2rem 1.4rem;height:100%; }
.metric-label { font-size:0.68rem;color:#666;letter-spacing:0.07em;text-transform:uppercase;font-weight:600;margin-bottom:0.5rem; }
.metric-value { font-family:"JetBrains Mono",monospace;font-size:2rem;font-weight:600;color:#111;line-height:1; }
.metric-unit  { font-size:0.78rem;color:#888;margin-top:0.28rem; }
.metric-range { font-family:"JetBrains Mono",monospace;font-size:0.68rem;color:#888;margin-top:0.4rem;padding-top:0.4rem;border-top:1px solid #f0f0f0; }
.section-header {
    font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;
    font-weight:700;color:#555;margin:2rem 0 0.8rem;
    display:flex;align-items:center;gap:0.6rem;
}
.section-header::after { content:"";flex:1;height:1px;background:#ebebeb; }
.wi-callout { background:#f0f7ff;border:1px solid #c8dcf5;border-radius:6px;padding:0.65rem 0.9rem;margin-top:0.6rem;font-size:0.81rem;color:#1a3a5c;line-height:1.6; }
.step-box { background:#fff;border:1px solid #e2e2e2;border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem; }
.step-num { display:inline-flex;align-items:center;justify-content:center;width:1.6rem;height:1.6rem;border-radius:50%;background:#2563EB;color:#fff;font-family:"JetBrains Mono",monospace;font-size:0.72rem;font-weight:700;margin-right:0.7rem;flex-shrink:0; }
.data-table { width:100%;border-collapse:collapse;font-size:0.8rem; }
.data-table thead tr { background:#f7f7f7;border-bottom:1px solid #e2e2e2; }
.data-table thead th { padding:6px 10px;font-size:0.66rem;font-weight:600;letter-spacing:0.07em;text-transform:uppercase;color:#777; }
.data-table tbody td { padding:8px 10px;border-bottom:1px solid #f0f0f0;color:#1a1a1a;font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding-bottom:1.2rem;border-bottom:1px solid #e2e2e2;margin-bottom:1.5rem;'>
  <div style='font-size:1.1rem;font-weight:650;color:#111;letter-spacing:-0.02em;'>Your Company Data</div>
  <div style='font-size:0.79rem;color:#666;margin-top:0.2rem;'>
    The Mitacs Accelerate deliverable — a model trained on your projects, not 40-year-old benchmarks
  </div>
</div>
""", unsafe_allow_html=True)

# ── The core argument ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#fff;border:1px solid #e2e2e2;border-left:3px solid #111;border-radius:0 8px 8px 0;padding:1rem 1.3rem;margin-bottom:1.5rem;'>
  <div style='font-size:0.64rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;color:#aaa;margin-bottom:0.5rem;'>The Problem With Generic Models</div>
  <div style='font-size:0.9rem;color:#222;line-height:1.8;'>
    The COCOMO-81 model was trained on 1980s defense software. The GitHub model was trained on open-source volunteers.
    Neither knows your team's velocity, your tech stack, or how your sprints actually run.
    <br><br>
    <strong>A model trained on your historical projects will outperform any generic benchmark</strong> —
    because it learns your specific patterns: which features predict overruns for you, which teams ship consistently, which project types are chronically underestimated.
  </div>
</div>
""", unsafe_allow_html=True)

# ── MMRE trajectory ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Expected MMRE Trajectory</div>", unsafe_allow_html=True)
st.markdown("""<table class='data-table'>
  <thead><tr><th>Model</th><th>Training Data</th><th>LOO-MMRE</th><th>PRED(25)</th></tr></thead>
  <tbody>
    <tr>
      <td>COCOMO-81 (academic baseline)</td>
      <td>63 projects, 1981</td>
      <td style='font-family:monospace;color:#555;'>0.3132</td>
      <td style='font-family:monospace;color:#555;'>47.6%</td>
    </tr>
    <tr>
      <td>GitHub open-source (proof of concept)</td>
      <td>91 repos, 2021–present</td>
      <td style='font-family:monospace;color:#2563EB;font-weight:600;'>0.1760</td>
      <td style='font-family:monospace;color:#2563EB;'>76.9%</td>
    </tr>
    <tr style='background:#f0fdf4;'>
      <td><strong>Your company model (Mitacs deliverable)</strong></td>
      <td>Your completed projects + Jira hours</td>
      <td style='font-family:monospace;color:#15803d;font-weight:700;'>&lt; 0.15 projected</td>
      <td style='font-family:monospace;color:#15803d;font-weight:700;'>&gt; 85% projected</td>
    </tr>
  </tbody>
</table>""", unsafe_allow_html=True)

st.markdown("""<div class='wi-callout' style='margin-top:0.6rem;'>
    Each row represents the same methodology (FAHP + weighted-kernel SVR + SHAP) on better data.
    The pattern is clear: domain-specific data produces domain-specific accuracy.
    Your projects are not open-source repos — they deserve a model trained on your reality.
</div>""", unsafe_allow_html=True)

# ── What you provide ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>What You Would Provide</div>", unsafe_allow_html=True)
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""**From Jira (per completed project)**
- Total story points delivered
- Sprint velocity (avg points per sprint)
- Cycle time (ticket open → done, days)
- Bug reopen rate
- Number of sprints
- Team size (unique assignees)
- Planned vs actual delivery date""")
with col_b:
    st.markdown("""**From GitHub (per completed project)**
- Total PRs merged
- Avg PR review time (days)
- Test coverage %
- Number of languages
- Copilot / AI tool adoption rate
- Commit frequency
- Code review turnaround""")

# ── What you get ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>What You Would Get</div>", unsafe_allow_html=True)
r1, r2, r3 = st.columns(3)
r1.markdown("""<div class='metric-card'>
    <div class='metric-label'>Company-Specific Model</div>
    <div class='metric-value' style='color:#2563EB;font-size:1.5rem;'>MMRE &lt;0.15</div>
    <div class='metric-unit'>projected</div>
    <div class='metric-range'>Trained on your projects, not 1981 benchmarks</div>
</div>""", unsafe_allow_html=True)
r2.markdown("""<div class='metric-card'>
    <div class='metric-label'>SHAP Explanations</div>
    <div class='metric-value' style='font-size:1.5rem;'>Per estimate</div>
    <div class='metric-unit'>in person-months</div>
    <div class='metric-range'>"Complexity added +18 PM to this project"</div>
</div>""", unsafe_allow_html=True)
r3.markdown("""<div class='metric-card'>
    <div class='metric-label'>Live What-If Tool</div>
    <div class='metric-value' style='font-size:1.5rem;'>Interactive</div>
    <div class='metric-unit'>deployed dashboard</div>
    <div class='metric-range'>"Add 1 senior dev → saves 12 PM"</div>
</div>""", unsafe_allow_html=True)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Mitacs Accelerate Timeline</div>", unsafe_allow_html=True)

for step_num, title, desc in [
    ("1", "Data Collection (Month 1–2)",
     "Extract completed project records from Jira + GitHub. Define effort ground truth (actual person-hours billed). Feature engineering to map your workflow signals to model inputs."),
    ("2", "Model Training & Validation (Month 3–4)",
     "Apply FAHP weighting to your feature set. Train weighted-kernel SVR. LOO cross-validation to tune hyperparameters. SHAP integration for per-project explanations."),
    ("3", "Deployment & Documentation (Month 5–6)",
     "Deploy as a live Streamlit dashboard. What-if analysis configured to your team's levers. Documentation and handoff so your team can maintain it."),
]:
    st.markdown(f"""<div class='step-box' style='display:flex;align-items:flex-start;'>
        <span class='step-num'>{step_num}</span>
        <div>
            <div style='font-size:0.85rem;font-weight:600;color:#111;margin-bottom:0.2rem;'>{title}</div>
            <div style='font-size:0.8rem;color:#555;line-height:1.6;'>{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Mitacs cost ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Mitacs Accelerate — Cost Structure</div>",
            unsafe_allow_html=True)
st.markdown("""<table class='data-table'>
  <thead><tr><th>Contributor</th><th>Amount</th><th>What it covers</th></tr></thead>
  <tbody>
    <tr><td><strong>Industry partner (you)</strong></td><td style='font-family:monospace;font-weight:600;'>$7,500</td><td>4–6 months of focused research + development</td></tr>
    <tr><td>Mitacs (federal funding)</td><td style='font-family:monospace;'>$15,000</td><td>Matched funding — Mitacs contributes 2× industry share</td></tr>
    <tr style='background:#f0fdf4;'><td><strong>Total research value</strong></td><td style='font-family:monospace;font-weight:600;color:#15803d;'>$22,500</td><td>Custom AI tool at 1/3 the cost of a typical engagement</td></tr>
  </tbody>
</table>""", unsafe_allow_html=True)

st.markdown("""<div class='wi-callout' style='margin-top:1rem;'>
    <strong>Next step:</strong> Share 10–20 completed project records (anonymised if preferred) from Jira.
    We can run a feasibility check — compute preliminary feature correlations and estimate
    what MMRE is achievable with your data — before any commitment is made.
</div>""", unsafe_allow_html=True)
