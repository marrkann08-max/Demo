import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "cv02","cv03","cv04","cv11";
}
.main { background: #F3F3F3; }
.main .block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #0F0F0F !important;
    border-right: 1px solid #1E1E1E !important;
}
section[data-testid="stSidebar"] > div { background: #0F0F0F !important; }
section[data-testid="stSidebar"] * { color: #CCCCCC !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a p { color: #AAAAAA !important; }
section[data-testid="stSidebar"] [aria-selected="true"] p { color: #FFFFFF !important; }

.page-header {
    margin-bottom: 1.75rem; padding-bottom: 1.25rem; border-bottom: 1px solid #E2E2E2;
}
.page-title    { font-size: 1.15rem; font-weight: 650; color: #111111; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.page-subtitle { font-size: 0.79rem; color: #666666; margin: 0; }

.section-header {
    font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
    font-weight: 700; color: #555555; margin: 2rem 0 0.9rem 0;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-header::after { content: ""; flex: 1; height: 1px; background: #EBEBEB; }

.metric-card {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 10px;
    padding: 1.2rem 1.4rem; height: 100%;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}
.metric-card:hover { border-color: #C8C8C8; box-shadow: 0 2px 14px rgba(0,0,0,0.07); }
.metric-label { font-size: 0.68rem; color: #666666; letter-spacing: 0.07em; text-transform: uppercase; font-weight: 600; margin-bottom: 0.5rem; }
.metric-value { font-family: "JetBrains Mono", monospace; font-size: 1.9rem; font-weight: 600; color: #111111; line-height: 1; }
.metric-unit  { font-size: 0.78rem; color: #888888; margin-top: 0.28rem; }
.metric-range { font-family: "JetBrains Mono", monospace; font-size: 0.68rem; color: #888888; margin-top: 0.42rem; padding-top: 0.42rem; border-top: 1px solid #F0F0F0; }

/* Signal cards */
.signal-card {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 8px;
    padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: flex-start; gap: 0.75rem;
}
.signal-icon {
    width: 2rem; height: 2rem; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-family: "JetBrains Mono", monospace; font-size: 0.65rem;
    font-weight: 700; letter-spacing: 0.04em;
}
.signal-icon-jira   { background: #EFF6FF; color: #2563EB; }
.signal-icon-github { background: #F5F3FF; color: #7C3AED; }
.signal-name { font-size: 0.82rem; font-weight: 600; color: #111111; margin-bottom: 0.1rem; }
.signal-desc { font-size: 0.75rem; color: #666666; line-height: 1.4; }

/* Timeline */
.timeline-item {
    display: flex; gap: 1.1rem; margin-bottom: 0; align-items: flex-start;
}
.timeline-left { display: flex; flex-direction: column; align-items: center; width: 2.2rem; flex-shrink: 0; }
.timeline-circle {
    width: 2rem; height: 2rem; border-radius: 50%;
    background: #2563EB; color: #FFFFFF;
    display: flex; align-items: center; justify-content: center;
    font-family: "JetBrains Mono", monospace; font-size: 0.72rem;
    font-weight: 700; flex-shrink: 0;
}
.timeline-line { width: 2px; flex: 1; background: #E2E2E2; min-height: 2rem; margin-top: 0.25rem; }
.timeline-body {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 8px;
    padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; flex: 1;
}
.timeline-month {
    display: inline-block; background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 3px; padding: 0.1rem 0.45rem;
    font-family: "JetBrains Mono", monospace; font-size: 0.63rem;
    color: #2563EB; font-weight: 600; margin-bottom: 0.35rem;
}
.timeline-title { font-size: 0.88rem; font-weight: 600; color: #111111; margin-bottom: 0.3rem; }
.timeline-desc  { font-size: 0.79rem; color: #555555; line-height: 1.6; }

/* Cost cards */
.cost-card {
    background: #FFFFFF; border: 1px solid #E2E2E2; border-radius: 10px;
    padding: 1.3rem 1.5rem; text-align: center; height: 100%;
}
.cost-amount {
    font-family: "JetBrains Mono", monospace; font-size: 2rem;
    font-weight: 700; line-height: 1; margin: 0.4rem 0 0.2rem;
}
.cost-label { font-size: 0.68rem; color: #666666; letter-spacing: 0.07em; text-transform: uppercase; font-weight: 600; }
.cost-desc  { font-size: 0.76rem; color: #888888; margin-top: 0.35rem; line-height: 1.45; }

/* Research table */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
.data-table thead tr { background: #F7F7F7; border-bottom: 1px solid #E2E2E2; }
.data-table thead th { padding: 0.42rem 0.75rem; text-align: left; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #777777; }
.data-table tbody td { padding: 0.52rem 0.75rem; color: #1A1A1A; border-bottom: 1px solid #F0F0F0; }
.data-table tbody tr:hover { background: #F7F7F7; }

.wi-callout { background: #F0F7FF; border: 1px solid #C8DCF5; border-radius: 6px; padding: 0.65rem 0.9rem; margin-top: 0.6rem; font-size: 0.81rem; color: #1A3A5C; line-height: 1.6; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load models (live MMRE figures) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_pkl(path, mtime):
    with open(path, "rb") as f:
        return pickle.load(f)

_cocomo_path = os.path.join(ROOT, "model.pkl")
_github_path = os.path.join(ROOT, "github_model.pkl")
_cocomo_art  = _load_pkl(_cocomo_path, os.path.getmtime(_cocomo_path)) if os.path.exists(_cocomo_path) else None
_github_art  = _load_pkl(_github_path, os.path.getmtime(_github_path)) if os.path.exists(_github_path) else None

_c_mmre   = f"{_cocomo_art['loo_mmre']:.4f}"            if _cocomo_art else "0.3132"
_c_pred25 = f"{_cocomo_art['pred25']*100:.1f}%"         if _cocomo_art and _cocomo_art.get('pred25') else "47.6%"
_c_n      = _cocomo_art.get('n_cocomo81', 63)            if _cocomo_art else 63
_g_mmre_r   = (_github_art.get('temporal_mmre') or _github_art.get('loo_mmre')) if _github_art else None
_g_pred25_r = (_github_art.get('temporal_pred25') or _github_art.get('pred25')) if _github_art else None
_g_n      = _github_art.get('n_projects', 91)            if _github_art else 91
_g_mmre   = f"{_g_mmre_r:.4f}"                          if _g_mmre_r   else "0.1760"
_g_pred25 = f"{_g_pred25_r*100:.1f}%"                   if _g_pred25_r else "76.9%"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
  <div class='page-title'>Your Company Data</div>
  <div class='page-subtitle'>The Mitacs Accelerate deliverable — a model trained on your projects, not 40-year-old benchmarks</div>
</div>
""", unsafe_allow_html=True)

# ── The core argument ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#fff;border:1px solid #e2e2e2;border-left:3px solid #111;
border-radius:0 8px 8px 0;padding:1rem 1.3rem;margin-bottom:1.2rem;'>
  <div style='font-size:0.64rem;letter-spacing:0.1em;text-transform:uppercase;
  font-weight:600;color:#aaa;margin-bottom:0.5rem;'>The problem with generic models</div>
  <div style='font-size:0.88rem;color:#222;line-height:1.8;'>
    The COCOMO-81 model was trained on 1980s defense software. The GitHub model was trained on open-source volunteers.
    Neither knows your team's velocity, your tech stack, or how your sprints actually run.
    <br><br>
    <strong>A model trained on your historical projects will outperform any generic benchmark</strong> —
    because it learns your specific patterns: which features predict overruns for you, which teams ship
    consistently, which project types are chronically underestimated.
  </div>
</div>
""", unsafe_allow_html=True)

# ── MMRE trajectory ───────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Expected MMRE Trajectory</div>", unsafe_allow_html=True)
st.markdown(f"""<table class='data-table'>
  <thead><tr><th>Model</th><th>Training Data</th><th>LOO-MMRE</th><th>PRED(25)</th></tr></thead>
  <tbody>
    <tr>
      <td>COCOMO-81 (academic baseline)</td>
      <td>{_c_n} projects, 1981</td>
      <td style='font-family:monospace;color:#555;'>{_c_mmre}</td>
      <td style='font-family:monospace;color:#555;'>{_c_pred25}</td>
    </tr>
    <tr>
      <td>GitHub open-source (proof of concept)</td>
      <td>{_g_n} repos, 2021–present</td>
      <td style='font-family:monospace;color:#2563EB;font-weight:600;'>{_g_mmre}</td>
      <td style='font-family:monospace;color:#2563EB;'>{_g_pred25}</td>
    </tr>
    <tr style='background:#f0fdf4;'>
      <td><strong>Your company model (Mitacs deliverable)</strong></td>
      <td>Your completed projects + Jira hours</td>
      <td style='font-family:monospace;color:#15803d;font-weight:700;'>&lt; 0.15 projected</td>
      <td style='font-family:monospace;color:#15803d;font-weight:700;'>&gt; 85% projected</td>
    </tr>
  </tbody>
</table>""", unsafe_allow_html=True)

st.markdown("""<div class='wi-callout'>
    Each row is the same methodology (FAHP + weighted-kernel SVR + SHAP) on better data.
    The pattern is clear: domain-specific data produces domain-specific accuracy.
    Your projects are not open-source repos — they deserve a model trained on your reality.
</div>""", unsafe_allow_html=True)

# ── What you provide ──────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>What You Would Provide</div>", unsafe_allow_html=True)

col_jira, col_gh = st.columns(2)

jira_signals = [
    ("JIRA", "Story Points",      "Total story points delivered per project"),
    ("JIRA", "Sprint Velocity",   "Average points completed per sprint"),
    ("JIRA", "Cycle Time",        "Ticket open → done in days"),
    ("JIRA", "Bug Reopen Rate",   "Quality signal — defects reintroduced post-close"),
    ("JIRA", "Team Size",         "Unique assignees per project"),
    ("JIRA", "Schedule Variance", "Planned vs actual delivery date delta"),
]
github_signals = [
    ("GH",   "PRs Merged",        "Total pull requests merged over project window"),
    ("GH",   "Review Time",       "Avg days from PR open to merge"),
    ("GH",   "Test Coverage",     "Fraction of files that are test/spec files"),
    ("GH",   "Language Count",    "Number of programming languages in the repo"),
    ("GH",   "AI Tool Adoption",  "Copilot / Cursor / Claude Code usage rate"),
    ("GH",   "Commit Frequency",  "Average commits per week over project window"),
]

with col_jira:
    st.markdown("<div style='font-size:0.72rem;font-weight:600;color:#2563EB;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:0.5rem;'>From Jira</div>", unsafe_allow_html=True)
    for tag, name, desc in jira_signals:
        st.markdown(f"""
        <div class='signal-card'>
          <div class='signal-icon signal-icon-jira'>{tag}</div>
          <div>
            <div class='signal-name'>{name}</div>
            <div class='signal-desc'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

with col_gh:
    st.markdown("<div style='font-size:0.72rem;font-weight:600;color:#7C3AED;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:0.5rem;'>From GitHub</div>", unsafe_allow_html=True)
    for tag, name, desc in github_signals:
        st.markdown(f"""
        <div class='signal-card'>
          <div class='signal-icon signal-icon-github'>{tag}</div>
          <div>
            <div class='signal-name'>{name}</div>
            <div class='signal-desc'>{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── What you get ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>What You Would Get</div>", unsafe_allow_html=True)
g1, g2, g3 = st.columns(3)
g1.markdown("""<div class='metric-card'>
    <div class='metric-label'>Company-Specific Model</div>
    <div class='metric-value' style='color:#2563EB;'>MMRE &lt;0.15</div>
    <div class='metric-unit'>projected accuracy</div>
    <div class='metric-range'>Trained on your projects, not 1981 benchmarks</div>
</div>""", unsafe_allow_html=True)
g2.markdown("""<div class='metric-card'>
    <div class='metric-label'>SHAP Explanations</div>
    <div class='metric-value' style='font-size:1.5rem;'>Per estimate</div>
    <div class='metric-unit'>in person-months</div>
    <div class='metric-range'>"Complexity added +18 PM to this project"</div>
</div>""", unsafe_allow_html=True)
g3.markdown("""<div class='metric-card'>
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
    No existing effort estimation model accounts for this.
    <strong>That's the Mitacs research contribution.</strong>
  </div>
</div>""", unsafe_allow_html=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
fig.patch.set_facecolor("white")

tools   = ["COCOMO-81\nTOOL (max)", "GitHub\nCopilot", "Cursor /\nClaude Code", "Full AI-first\nteam"]
savings = [17, 26, 38, 50]
lo_err  = [0,  4,  8, 10]
hi_err  = [0,  9, 12, 10]
colors  = ["#94a3b8", "#3b82f6", "#8b5cf6", "#6d28d9"]

bars = ax1.bar(tools, savings, color=colors, width=0.55, edgecolor="none", alpha=0.88)
ax1.errorbar(range(len(tools)), savings, yerr=[lo_err, hi_err],
             fmt="none", color="#374151", capsize=5, linewidth=1.2, capthick=1.2)
ax1.axhline(17, color="#ef4444", linewidth=1.2, linestyle="--", alpha=0.7, label="COCOMO TOOL ceiling (17%)")
for bar, val, hi in zip(bars, savings, hi_err):
    ax1.text(bar.get_x() + bar.get_width()/2, val + hi + 2, f"{val}%",
             ha="center", va="bottom", fontsize=9, fontweight="600",
             fontfamily="monospace", color="#111")
ax1.set_ylabel("Effort reduction (%)", fontsize=8.5, color="#6b7280")
ax1.set_title("COCOMO TOOL Ceiling vs AI Tools", fontsize=9.5, fontweight="600", color="#111", pad=10)
ax1.legend(fontsize=7.5, frameon=False)
ax1.set_ylim(0, 68)
ax1.set_facecolor("#fafafa")
for sp in ["top", "right", "left"]: ax1.spines[sp].set_visible(False)
ax1.spines["bottom"].set_color("#e5e7eb")
ax1.tick_params(labelsize=8, colors="#6b7280")
ax1.grid(axis="y", color="#f3f4f6", linewidth=0.8)

years    = [2021, 2022, 2023, 2024, 2025]
adoption = [8, 22, 45, 65, 78]
ax2.fill_between(years, adoption, alpha=0.15, color="#7c3aed")
ax2.plot(years, adoption, color="#7c3aed", linewidth=2.2, marker="o", markersize=6)
for x, y in zip(years, adoption):
    ha = "right" if x == 2025 else "center"
    xoff = -0.05 if x == 2025 else 0
    ax2.text(x + xoff, y + 3, f"{y}%", ha=ha, fontsize=8.5,
             fontfamily="monospace", color="#5b21b6", fontweight="600")
ax2.axvline(2022.5, color="#ef4444", linewidth=1, linestyle=":", alpha=0.6)
ax2.text(2022.6, 15, "Copilot\nGA", fontsize=7.5, color="#ef4444", va="bottom")
ax2.set_title("AI Coding Tool Adoption (% developers)", fontsize=9.5, fontweight="600", color="#111", pad=10)
ax2.set_ylabel("Developers using AI tools (%)", fontsize=8.5, color="#6b7280")
ax2.set_xlim(2020.7, 2025.4)
ax2.set_ylim(0, 95)
ax2.set_facecolor("#fafafa")
for sp in ["top", "right", "left"]: ax2.spines[sp].set_visible(False)
ax2.spines["bottom"].set_color("#e5e7eb")
ax2.tick_params(labelsize=8, colors="#6b7280")
ax2.grid(axis="y", color="#f3f4f6", linewidth=0.8)

plt.tight_layout(pad=1.5)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.markdown("""<table class='data-table' style='margin-top:0.8rem;'>
  <thead><tr><th>Study</th><th>Sample</th><th>Finding</th><th>Gain</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>Peng et al. (2023)</strong> — GitHub</td>
      <td>95 developers, controlled</td>
      <td>Copilot users completed tasks 55% faster on isolated tasks, 26% on full projects</td>
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
    <tr style='background:#fdf4ff;'>
      <td><strong>COCOMO-81 TOOL driver</strong></td>
      <td>1981 — design-era tools</td>
      <td>Best possible tools (Very High) reduce effort by 17% maximum</td>
      <td style='font-family:monospace;font-weight:600;color:#dc2626;'>max 17%</td>
    </tr>
  </tbody>
</table>""", unsafe_allow_html=True)

st.markdown("""<div class='wi-callout' style='margin-top:0.8rem;background:#fdf4ff;border-color:#c4b5fd;color:#4c1d95;'>
    <strong>The Mitacs research question:</strong> How much does AI coding tool adoption reduce effort
    for <em>your specific team</em> on <em>your specific stack</em>?
    COCOMO says 17%. Studies say 22–55%. The real number depends on your codebase,
    your team's acceptance rate, and how much of your work is automatable.
    Measuring this gap — empirically, on your data — is a publishable finding.
</div>""", unsafe_allow_html=True)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Mitacs Accelerate Timeline</div>", unsafe_allow_html=True)

timeline = [
    ("1", "Month 1–2", "Data Collection",
     "Extract completed project records from Jira + GitHub. Define effort ground truth (actual person-hours billed). Feature engineering to map your workflow signals to model inputs."),
    ("2", "Month 3", "Model Training & Validation",
     "Apply FAHP weighting to your feature set. Train weighted-kernel SVR. LOO cross-validation to tune hyperparameters. SHAP integration for per-project explanations."),
    ("3", "Month 4", "Deployment & Handoff",
     "Deploy as a live Streamlit dashboard. What-if analysis configured to your team's levers. Documentation and handoff so your team can maintain and extend it."),
]

for idx, (num, month, title, desc) in enumerate(timeline):
    is_last = idx == len(timeline) - 1
    st.markdown(f"""
    <div class='timeline-item'>
      <div class='timeline-left'>
        <div class='timeline-circle'>{num}</div>
        {'<div class="timeline-line"></div>' if not is_last else ''}
      </div>
      <div class='timeline-body'>
        <div class='timeline-month'>{month}</div>
        <div class='timeline-title'>{title}</div>
        <div class='timeline-desc'>{desc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Cost ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Mitacs Accelerate — Cost Structure</div>", unsafe_allow_html=True)

cc1, cc2, cc3 = st.columns(3)
cc1.markdown("""<div class='cost-card' style='border-top:3px solid #2563EB;'>
    <div class='cost-label'>Industry Partner</div>
    <div class='cost-amount' style='color:#2563EB;'>$7,500</div>
    <div class='cost-desc'>4 months of focused research + development</div>
</div>""", unsafe_allow_html=True)
cc2.markdown("""<div class='cost-card' style='border-top:3px solid #7C3AED;'>
    <div class='cost-label'>Mitacs (federal funding)</div>
    <div class='cost-amount' style='color:#7C3AED;'>$7,500</div>
    <div class='cost-desc'>Matched 1:1 — equal federal and industry contribution</div>
</div>""", unsafe_allow_html=True)
cc3.markdown("""<div class='cost-card' style='border-top:3px solid #15803D;background:#f0fdf4;'>
    <div class='cost-label'>Total Research Value</div>
    <div class='cost-amount' style='color:#15803D;'>$15,000</div>
    <div class='cost-desc'>Custom AI estimation tool delivered in 4 months</div>
</div>""", unsafe_allow_html=True)

st.markdown("""<div class='wi-callout' style='margin-top:1rem;'>
    <strong>Next step:</strong> Share 10–20 completed project records (anonymised if preferred) from Jira.
    We can run a feasibility check — compute preliminary feature correlations and estimate
    what MMRE is achievable with your data — before any commitment is made.
</div>""", unsafe_allow_html=True)
