"""
patch_test_ratio.py  —  recompute test_file_ratio from the git tree API
Fixes the broken heuristic (almost all repos got 0.22) with real counts.

Run after collection: python patch_test_ratio.py
"""
import requests, time, os, pandas as pd

TOKEN   = os.environ.get("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}
BASE    = "https://api.github.com"

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

def compute_test_ratio(owner, repo):
    """Count test files vs total files using the git tree API (1 API call)."""
    r = get_api(f"{BASE}/repos/{owner}/{repo}/git/trees/HEAD",
                params={"recursive": "1"})
    if r is None or r.status_code != 200:
        return None
    data  = r.json()
    files = [item["path"] for item in data.get("tree", []) if item["type"] == "blob"]
    if not files:
        return None
    total = len(files)
    test_files = sum(
        1 for p in files if
        "/test"   in p.lower() or
        "/tests"  in p.lower() or
        "/spec"   in p.lower() or
        "test_"   in p.lower() or
        "_test."  in p.lower() or
        "_spec."  in p.lower() or
        p.lower().endswith("_test.go") or
        p.lower().endswith("_spec.rb")
    )
    return round(test_files / total, 4)

csv_path = os.path.join(os.path.dirname(__file__), "github_projects.csv")
df = pd.read_csv(csv_path)
print(f"Patching test_file_ratio for {len(df)} repos...")

for i, row in df.iterrows():
    owner, repo = row["repo"].split("/")
    ratio = compute_test_ratio(owner, repo)
    if ratio is not None:
        df.at[i, "test_file_ratio"] = ratio
        print(f"  {row['repo']:45s} {ratio:.3f}")
    else:
        print(f"  {row['repo']:45s} SKIP (API error)")
    time.sleep(0.4)

df.to_csv(csv_path, index=False)
print(f"\nSaved. test_file_ratio range: {df['test_file_ratio'].min():.3f} – {df['test_file_ratio'].max():.3f}")
print(df[["repo","test_file_ratio"]].to_string(index=False))
