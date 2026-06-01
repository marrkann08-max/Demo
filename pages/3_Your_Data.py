import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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

# ── AI in Modern Development ──────────────────────────────────────────────────
st.markdown("<div class='section-header'>AI Coding Tools — The Gap COCOMO Can't See</div>",
            unsafe_allow_html=True)

st.markdown("""<div style='background:#fff;border:1px solid #e2e2e2;border-left:3px solid #7c3aed;
border-radius:0 8px 8px 0;padding:0.9rem 1.1rem;margin-bottom:1rem;'>
  <div style='font-size:0.64rem;letter-spacing:0.1em;text-transform:uppercase;font-weight:600;
  color:#7c3aed;margin-bottom:0.4rem;'>The core research gap</div>
  <div style='font-size:0.85rem;color:#222;line-height:1.8;'>
    COCOMO-81's TOOL driver assumes the best tools save at most <strong>17% effort</strong>.
    GitHub Copilot was released in 2022. Cursor, Claude Code, and Devin followed.
    Multiple peer-reviewed studies now show <strong>22–55% productivity gains</strong>.
    No existing effort estimation model — including Sehra et al. (2018) — accounts for this.
    <strong>That's the Mitacs research contribution.</strong>
  </div>
</div>""", unsafe_allow_html=True)

# Research data chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
fig.patch.set_facecolor("white")

# Left: COCOMO TOOL cap vs actual AI savings
tools    = ["COCOMO-81\nTOOL (max)", "GitHub\nCopilot", "Cursor /\nClaude Code", "Full AI-first\nteam"]
savings  = [17, 26, 38, 50]
lo_err   = [0,  4,  8, 10]
hi_err   = [0,  9, 12, 10]
colors   = ["#94a3b8", "#3b82f6", "#8b5cf6", "#6d28d9"]

bars = ax1.bar(tools, savings, color=colors, width=0.55, edgecolor="none", alpha=0.88)
ax1.errorbar(range(len(tools)), savings,
             yerr=[lo_err, hi_err], fmt="none",
             color="#374151", capsize=5, linewidth=1.2, capthick=1.2)
ax1.axhline(17, color="#ef4444", linewidth=1.2, linestyle="--", alpha=0.7, label="COCOMO TOOL ceiling (17%)")
for bar, val in zip(bars, savings):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val}%",
             ha="center", va="bottom", fontsize=9, fontweight="600",
             fontfamily="monospace", color="#111")
ax1.set_ylabel("Effort reduction (%)", fontsize=8.5, color="#6b7280")
ax1.set_title("COCOMO TOOL Ceiling vs AI Tools", fontsize=9.5, fontweight="600", color="#111", pad=10)
ax1.legend(fontsize=7.5, frameon=False)
ax1.set_ylim(0, 68)
ax1.set_facecolor("#fafafa")
for sp in ["top","right","left"]: ax1.spines[sp].set_visible(False)
ax1.spines["bottom"].set_color("#e5e7eb")
ax1.tick_params(labelsize=8, colors="#6b7280")
ax1.grid(axis="y", color="#f3f4f6", linewidth=0.8)

# Right: AI adoption growth
years     = [2021, 2022, 2023, 2024, 2025]
adoption  = [8, 22, 45, 65, 78]   # % developers using AI coding tools (Stack Overflow / GitHub surveys)
ax2.fill_between(years, adoption, alpha=0.15, color="#7c3aed")
ax2.plot(years, adoption, color="#7c3aed", linewidth=2.2, marker="o", markersize=6)
for x, y in zip(years, adoption):
    ax2.text(x, y + 2.5, f"{y}%", ha="center", fontsize=8.5,
             fontfamily="monospace", color="#5b21b6", fontweight="600")
ax2.axvline(2022.5, color="#ef4444", linewidth=1, linestyle=":", alpha=0.6)
ax2.text(2022.6, 15, "Copilot\nGA", fontsize=7.5, color="#ef4444", va="bottom")
ax2.set_title("AI Coding Tool Adoption (% developers)", fontsize=9.5, fontweight="600", color="#111", pad=10)
ax2.set_ylabel("Developers using AI tools (%)", fontsize=8.5, color="#6b7280")
ax2.set_ylim(0, 95)
ax2.set_facecolor("#fafafa")
for sp in ["top","right","left"]: ax2.spines[sp].set_visible(False)
ax2.spines["bottom"].set_color("#e5e7eb")
ax2.tick_params(labelsize=8, colors="#6b7280")
ax2.grid(axis="y", color="#f3f4f6", linewidth=0.8)

plt.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Research evidence table
st.markdown("""<table class='data-table' style='margin-top:0.8rem;'>
  <thead><tr><th>Study</th><th>Sample</th><th>Finding</th><th>Productivity Gain</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Peng et al. (2023)</strong> — GitHub</td>
      <td>95 developers, controlled experiment</td>
      <td>Copilot users completed tasks 55% faster (isolated tasks), 26% on full projects</td>
      <td style='font-family:monospace;font-weight:600;color:#15803d;'>+22–55%</td>
    </tr>
    <tr>
      <td><strong>McKinsey (2023)</strong></td>
      <td>850 enterprise projects</td>
      <td>20–45% productivity gain depending on AI adoption depth and task type</td>
      <td style='font-family:monospace;font-weight:600;color:#15803d;'>+20–45%</td>
    </tr>
    <tr>
      <td><strong>Kalliamvakou (2022)</strong> — GitHub</td>
      <td>Copilot users vs non-users</td>
      <td>Copilot users merged PRs 26% faster, reported higher satisfaction</td>
      <td style='font-family:monospace;font-weight:600;color:#15803d;'>+26%</td>
    </tr>
    <tr>
      <td><strong>Stack Overflow Survey (2024)</strong></td>
      <td>65,000 developers globally</td>
      <td>82% have used AI tools; 62% use them regularly in development workflow</td>
      <td style='font-family:monospace;color:#555;'>62% adoption</td>
    </tr>
    <tr>
      <td><strong>GitHub Octoverse (2024)</strong></td>
      <td>1.3M+ Copilot users</td>
      <td>Copilot accounts for 30–40% of code in repos where it's enabled</td>
      <td style='font-family:monospace;color:#555;'>30–40% of code</td>
    </tr>
    <tr style='background:#fdf4ff;'>
      <td><strong>COCOMO-81 TOOL driver</strong></td>
      <td>1981 — design-era tools</td>
      <td>Best possible tools (Very High) reduce effort by 17% maximum</td>
      <td style='font-family:monospace;font-weight:600;color:#dc2626;'>max 17%</td>
    </tr>
  </tbody>
</table>""", unsafe_allow_html=True)

st.markdown("""<div class='wi-callout' style='margin-top:0.8rem;background:#fdf4ff;border-color:#c4b5fd;color:#4c1d95;'>
    <strong>The Mitacs research question:</strong> How much does AI coding tool adoption
    reduce effort for <em>your specific team</em> on <em>your specific stack</em>?
    COCOMO says 17%. Studies say 22–55%. The real number depends on your codebase,
    your team's Copilot acceptance rate, and how much of your work is automatable.
    Measuring this gap — empirically, on your data — is a publishable finding.
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
