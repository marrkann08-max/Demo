"""
patch_ai_adoption.py — detect AI coding tool adoption in GitHub repos
Checks for config files: .cursorrules, copilot-instructions.md, CLAUDE.md,
AGENTS.md, .windsurfrules, devin.md — via the git tree API (1 call per repo).

Run: python patch_ai_adoption.py
Updates: github_projects.csv (adds ai_tools_count, ai_tools_list columns)
"""
import requests, time, os, ast
import pandas as pd

TOKEN   = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
BASE    = "https://api.github.com"

# Files that signal deliberate AI tool adoption
AI_SIGNALS = {
    ".cursorrules":                     "Cursor",
    "copilot-instructions.md":          "Copilot",
    ".github/copilot-instructions.md":  "Copilot",
    "claude.md":                        "Claude",
    "CLAUDE.md":                        "Claude",
    ".claude/settings.json":            "Claude",
    "agents.md":                        "Agents",
    "AGENTS.md":                        "Agents",
    ".windsurfrules":                   "Windsurf",
    "devin.md":                         "Devin",
    ".aider.conf.yml":                  "Aider",
    ".aider.model.settings.yml":        "Aider",
}

def get_api(url, params=None):
    for _ in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        except requests.RequestException:
            time.sleep(4); continue
        if r.status_code == 403:
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 65))
            time.sleep(max(reset - time.time() + 5, 5)); continue
        return r
    return None

def detect_ai_tools(owner, repo):
    r = get_api(f"{BASE}/repos/{owner}/{repo}/git/trees/HEAD", params={"recursive": "1"})
    if r is None or r.status_code != 200:
        return 0, []
    files = {item["path"].lower() for item in r.json().get("tree", []) if item["type"] == "blob"}
    found = set()
    for pattern, tool in AI_SIGNALS.items():
        if pattern.lower() in files:
            found.add(tool)
    return len(found), sorted(found)

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "github_projects.csv")
df = pd.read_csv(csv_path)
print(f"Detecting AI tool adoption for {len(df)} repos...")

counts, tools_lists = [], []
for _, row in df.iterrows():
    owner, repo = row["repo"].split("/")
    count, tools = detect_ai_tools(owner, repo)
    counts.append(count)
    tools_lists.append(",".join(tools) if tools else "")
    status = f"({','.join(tools)})" if tools else "(none)"
    print(f"  {row['repo']:45s}  {status}")
    time.sleep(0.5)

df["ai_tools_count"] = counts
df["ai_tools_list"]  = tools_lists
df.to_csv(csv_path, index=False)

n_using = sum(1 for c in counts if c > 0)
print(f"\nDone. {n_using}/{len(df)} repos have AI tool config files ({n_using/len(df)*100:.0f}%)")
print(df[df["ai_tools_count"] > 0][["repo", "ai_tools_list"]].to_string(index=False))
