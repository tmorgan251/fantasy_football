"""
Probe ESPN Fantasy Football league endpoints for a set of KNOWN (private) league IDs
across multiple seasons.

Goal:
- For each (league_id, season) pair, record HTTP status + quick metadata
- Find which seasons each league_id actually exists for (200/401) vs 404
- Because leagues are private, expect a lot of 401 (exists but unauthorized) unless you add cookies

Notes:
- 200 = exists + readable without auth (rare for private leagues)
- 401 = exists but private (expected)
- 404 = not found for that season (common if season mismatch)
- 403 = forbidden / blocked
"""

import time
import requests
import pandas as pd
from collections import Counter

BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"

# --- You will fill these in ---
KNOWN_LEAGUE_IDS = [
    # 751465,
    # 937734951,
]
SEASONS_TO_CHECK = list(range(2018, 2026))  # inclusive start, exclusive end -> 2018..2025
# -----------------------------

REQUEST_DELAY = 0.15  # be polite; you can increase if you see issues
TIMEOUT = 10

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://fantasy.espn.com/",
    "Origin": "https://fantasy.espn.com",
    "Connection": "keep-alive",
})

# Optional: if you later want to test with cookies, uncomment and set values safely
# ESPN_S2 = os.environ.get("ESPN_S2", "")
# SWID = os.environ.get("ESPN_SWID", "")
# if ESPN_S2 and SWID:
#     SESSION.cookies.set("espn_s2", ESPN_S2)
#     SESSION.cookies.set("SWID", SWID)


def fetch_probe(league_id: int, season: int, timeout: int = TIMEOUT, allow_redirects: bool = False):
    """
    Returns a dict with status code, final URL, content-type, and a tiny body snippet.
    Does NOT attempt JSON parsing (since 401/404 responses are still JSON-ish sometimes).
    """
    url = f"{BASE_URL}/seasons/{season}/segments/0/leagues/{league_id}"
    params = [("view", "mSettings"), ("view", "mTeam")]

    try:
        r = SESSION.get(url, params=params, timeout=timeout, allow_redirects=allow_redirects)
        ctype = r.headers.get("Content-Type", "")
        text_head = (r.text or "")[:120].replace("\n", " ").replace("\r", " ").strip()

        return {
            "league_id": league_id,
            "season": season,
            "status": r.status_code,
            "content_type": ctype,
            "final_url": r.url,
            "redirected": bool(r.history),
            "history": " | ".join([f"{h.status_code}->{h.headers.get('Location','')}" for h in r.history]) if r.history else "",
            "body_head": text_head,
        }
    except requests.exceptions.Timeout:
        return {
            "league_id": league_id,
            "season": season,
            "status": "timeout",
            "content_type": "",
            "final_url": "",
            "redirected": False,
            "history": "",
            "body_head": "",
        }
    except requests.exceptions.RequestException as e:
        return {
            "league_id": league_id,
            "season": season,
            "status": "request_error",
            "content_type": "",
            "final_url": "",
            "redirected": False,
            "history": str(e)[:120],
            "body_head": "",
        }


def probe_leagues_across_seasons(
    league_ids,
    seasons,
    delay=REQUEST_DELAY,
    status_every=25,
):
    """
    Probe all league_ids across all seasons.
    Prints compact progress + status-code counts occasionally.

    Returns: DataFrame of results.
    """
    league_ids = [int(x) for x in league_ids]
    seasons = [int(y) for y in seasons]

    total = len(league_ids) * len(seasons)
    done = 0
    status_counts = Counter()
    start = time.time()
    rows = []

    print(f"🔎 Probing {len(league_ids)} league IDs across {len(seasons)} seasons ({total} requests)\n", flush=True)

    for lid in league_ids:
        for yr in seasons:
            row = fetch_probe(lid, yr)
            rows.append(row)
            status_counts[row["status"]] += 1
            done += 1

            if done % status_every == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                top = ", ".join(f"{k}:{v}" for k, v in status_counts.most_common(5))
                print(f"📊 {done}/{total} | {rate:.2f} req/s | statuses: [{top}]", flush=True)

            time.sleep(delay)

    df = pd.DataFrame(rows)
    return df


def summarize_probe_results(df: pd.DataFrame):
    """
    Prints a useful summary per league_id:
    - seasons with 401 (exists but private)
    - seasons with 200 (exists and readable)
    - seasons with 404 (not found)
    - other statuses
    """
    print("\n===== SUMMARY (per league_id) =====", flush=True)

    for lid, g in df.groupby("league_id"):
        # normalize statuses to strings for easy filtering
        statuses = g[["season", "status"]].sort_values("season")

        seasons_200 = statuses.loc[statuses["status"] == 200, "season"].tolist()
        seasons_401 = statuses.loc[statuses["status"] == 401, "season"].tolist()
        seasons_404 = statuses.loc[statuses["status"] == 404, "season"].tolist()

        other = statuses.loc[~statuses["status"].isin([200, 401, 404]), ["season", "status"]].values.tolist()

        print(f"\nLeague {lid}:", flush=True)
        print(f"  200 (readable): {seasons_200 if seasons_200 else '—'}", flush=True)
        print(f"  401 (private/exists): {seasons_401 if seasons_401 else '—'}", flush=True)
        print(f"  404 (not in season): {seasons_404 if seasons_404 else '—'}", flush=True)
        if other:
            print(f"  Other: {other}", flush=True)

    print("\n===== OVERALL STATUS COUNTS =====", flush=True)
    print(df["status"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    # 1) Put your known IDs here (uncomment / add as ints)
    KNOWN_LEAGUE_IDS = [
        751465,
        937734951,
    ]

    # 2) Choose seasons to check
    SEASONS_TO_CHECK = list(range(2018, 2026))  # 2018..2025

    # 3) Run the probe
    probe_df = probe_leagues_across_seasons(
        league_ids=KNOWN_LEAGUE_IDS,
        seasons=SEASONS_TO_CHECK,
        delay=REQUEST_DELAY,
        status_every=10,   # print every 10 requests (adjust as you like)
    )

    # 4) Print summary
    summarize_probe_results(probe_df)

    # 5) Optional: save results
    probe_df.to_csv("espn_probe_results.csv", index=False)
    print("\n✅ Saved: espn_probe_results.csv", flush=True)
