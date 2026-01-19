"""
One-time run.

Goal:
- Discover and RECORD "seed" league IDs that return HTTP 200 (readable without auth).
- As soon as we have a few seeds, switch most sampling to "near seed" (seeded sampling),
  while keeping some global exploration to find new clusters.

Outputs:
- seeds_200_season_{SEASON}.csv   (HTTP 200 leagues + metadata)
- exists_401_season_{SEASON}.csv  (optional: leagues that exist but are private, useful for diagnostics)

Notes:
- We do NOT over-filter during discovery. You can filter to 10-team + PPR later.
"""

import os
import random
import time
import requests
import pandas as pd
from collections import Counter

BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"
SEASON = 2024

# --- Discovery settings ---
TARGET_SEEDS_200 = 50
ID_RANGE_LOW = 1_000_000
ID_RANGE_HIGH = 20_000_000

REQUEST_DELAY = 0.1
TIMEOUT = 5
STATUS_EVERY_ATTEMPTS = 250
STATUS_EVERY_SECONDS = 20
SAVE_EVERY_NEW_SEEDS = 10

# --- Seeded sampling settings (new) ---
SEEDING_START_AT = 5          # once we have >= this many 200 seeds, start sampling near them
SEED_PROBABILITY = 0.85       # % of time to sample near a seed once seeding starts
SEED_WINDOW = 200_000         # sample uniformly from [seed - window, seed + window]
SEED_WINDOW_GROWTH = 1.25     # if seeded hit-rate is bad, window grows up to max
SEED_WINDOW_MAX = 2_000_000

# --- Optional: record 401 "exists but private" IDs for troubleshooting/analysis ---
RECORD_401 = True
MAX_401_TO_KEEP = 20_000

OUT_DIR = "data/raw/ESPN/Reduced_Request_Delay"
OUT_SEEDS_CSV = os.path.join(OUT_DIR, f"seeds_200_season_{SEASON}.csv")
OUT_EXISTS_401_CSV = os.path.join(OUT_DIR, f"exists_401_season_{SEASON}.csv")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://fantasy.espn.com/",
    "Origin": "https://fantasy.espn.com",
    "Connection": "keep-alive",
})


def fetch_league_json(league_id: int, timeout: int = TIMEOUT):
    """
    Fetch league JSON. We only treat HTTP 200 + JSON content-type as success.

    Returns:
      (league_json_or_None, status_tag)
        status_tag is one of: 200, 401, 403, 404, 'timeout', 'request_error', 'non_json', 'bad_json'
    """
    url = f"{BASE_URL}/seasons/{SEASON}/segments/0/leagues/{league_id}"
    params = [("view", "mSettings"), ("view", "mTeam")]

    try:
        r = SESSION.get(url, params=params, timeout=timeout, allow_redirects=False)
        code = r.status_code

        if code != 200:
            return None, code

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "json" not in ctype:
            return None, "non_json"

        try:
            return r.json(), 200
        except ValueError:
            return None, "bad_json"

    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.RequestException:
        return None, "request_error"


def extract_seed_metadata(league_json: dict) -> dict:
    """
    Extract minimal fields for later filtering + analysis.
    """
    settings = league_json.get("settings", {}) or {}
    scoring = (settings.get("scoringSettings", {}) or {}).get("scoringItems", []) or []
    teams = league_json.get("teams", []) or []

    # ESPN statId 53 is receptions. Store the points for that (PPR value) if present.
    ppr_points = None
    for item in scoring:
        if item.get("statId") == 53:
            ppr_points = item.get("points")
            break

    # Sum of team points-for (when available)
    total_pf = None
    try:
        total_pf = sum(
            (t.get("record", {}) or {}).get("overall", {}).get("pointsFor", 0)
            for t in teams
            if isinstance(t, dict)
        )
    except Exception:
        total_pf = None

    return {
        "league_id": league_json.get("id"),
        "season": SEASON,
        "num_teams": len(teams) if teams is not None else None,
        "ppr_points": ppr_points,
        "total_pf": total_pf,
    }


def save_csv(rows, path, dedupe_cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty and dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols)
    df.to_csv(path, index=False)
    return df


def choose_league_id(seeds_200_ids, current_window):
    """
    Choose next league_id:
    - If we have enough seeds, sample near a seed with probability SEED_PROBABILITY.
    - Otherwise sample globally.
    """
    if len(seeds_200_ids) >= SEEDING_START_AT and random.random() < SEED_PROBABILITY:
        s = random.choice(seeds_200_ids)
        lo = max(ID_RANGE_LOW, s - current_window)
        hi = min(ID_RANGE_HIGH, s + current_window)
        return random.randint(lo, hi), True  # seeded pick
    else:
        return random.randint(ID_RANGE_LOW, ID_RANGE_HIGH), False  # global pick

def load_existing_seeds(path=OUT_SEEDS_CSV):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    # defensive: keep only expected columns if present
    rows = df.to_dict("records")
    return rows


def discover_seeds_200(target_seeds_200: int = TARGET_SEEDS_200,
                       request_delay: float = REQUEST_DELAY):
    """
    Discover ESPN public (HTTP 200) seed leagues for SEASON, saving progress to disk.

    Key behaviors:
    - Resumes from OUT_SEEDS_CSV (deduped by league_id+season)
    - Does NOT re-fetch already-saved seed league_ids
    - Optionally resumes / records 401 (exists but private) to OUT_EXISTS_401_CSV
    - Prints periodic status (time- and attempt-gated) without spamming
    - Uses seeded sampling once enough 200 seeds exist, and can auto-expand the window
    """

    # ----------------------------
    # Resume from disk (200 seeds)
    # ----------------------------
    seeds = load_existing_seeds(OUT_SEEDS_CSV)
    # de-dupe by (league_id, season)
    seeds = list({(r.get("league_id"), r.get("season")): r for r in seeds}.values())

    # Build seed ID trackers from loaded seeds (normalize to int)
    seeds_200_ids: list[int] = []
    seeds_200_id_set: set[int] = set()
    for r in seeds:
        lid = r.get("league_id")
        if pd.notna(lid):
            try:
                lid_int = int(lid)
            except Exception:
                continue
            seeds_200_id_set.add(lid_int)
            seeds_200_ids.append(lid_int)

    # Seen IDs should start with the already-known seeds, so we don't refetch them
    seen_ids: set[int] = set(seeds_200_id_set)

    # We'll track "session added" separately so status hit-rate isn't misleading after resume
    initial_seed_count = len(seeds)

    # ----------------------------
    # Resume from disk (401 list) - optional
    # ----------------------------
    exists_401: list[dict] = []
    exists_401_set: set[int] = set()
    if RECORD_401 and os.path.exists(OUT_EXISTS_401_CSV):
        try:
            df401 = pd.read_csv(OUT_EXISTS_401_CSV)
            # keep it simple: only track IDs for this season in memory
            # (CSV may contain multiple seasons; dedupe on save handles it)
            for lid in df401.get("league_id", pd.Series(dtype="float")).dropna().tolist():
                try:
                    exists_401_set.add(int(lid))
                except Exception:
                    pass

            # If the file had rows for this season already, keep a lightweight list too
            # (We don't strictly need the list contents to dedupe, but it's used for periodic saves)
            if "season" in df401.columns:
                df401_season = df401[df401["season"] == SEASON]
                for lid in df401_season.get("league_id", pd.Series(dtype="float")).dropna().tolist():
                    try:
                        exists_401.append({"league_id": int(lid), "season": int(SEASON)})
                    except Exception:
                        pass
            else:
                for lid in list(exists_401_set):
                    exists_401.append({"league_id": int(lid), "season": int(SEASON)})
        except Exception:
            # If the 401 CSV is malformed, just start fresh in-memory (we still won't crash)
            exists_401 = []
            exists_401_set = set()

    status_counts = Counter()

    attempts = 0
    start = time.time()
    last_status_time = start
    last_status_attempt = 0
    last_saved_seed_count = len(seeds)      # IMPORTANT: don't immediately re-save
    last_saved_401_count = len(exists_401)

    # Seeded sampling metrics (for auto-tuning window)
    current_window = SEED_WINDOW
    seeded_attempts = 0
    seeded_200 = 0
    last_window_tune_attempt = 0

    print(f"↩️ Resuming with {len(seeds)} saved 200-seeds (file: {OUT_SEEDS_CSV})", flush=True)
    if RECORD_401:
        print(f"↩️ Resuming with {len(exists_401_set)} known 401 IDs (file: {OUT_EXISTS_401_CSV})", flush=True)

    print(f"🚀 Discovering HTTP 200 seed leagues | Season: {SEASON}", flush=True)
    print(f"🔢 Global ID range: {ID_RANGE_LOW:,} – {ID_RANGE_HIGH:,}", flush=True)
    print(f"🎯 Target 200 seeds: {target_seeds_200}", flush=True)
    print(f"⚙️ Delay: {request_delay}s | Timeout: {TIMEOUT}s", flush=True)
    print(f"🌱 Seeded sampling: start_at={SEEDING_START_AT}, prob={SEED_PROBABILITY}, window=±{SEED_WINDOW:,}\n", flush=True)

    def maybe_status(force: bool = False):
        nonlocal last_status_time, last_status_attempt
        now = time.time()

        if not force:
            # Gate by BOTH attempts and seconds to avoid spam during slow/hiccup periods
            if (attempts - last_status_attempt) < STATUS_EVERY_ATTEMPTS and (now - last_status_time) < STATUS_EVERY_SECONDS:
                return

        elapsed = now - start
        rate = attempts / elapsed if elapsed > 0 else 0.0

        top = ", ".join(f"{k}:{v}" for k, v in status_counts.most_common(6)) or "—"

        # Session hit: new seeds found during this run
        session_new = len(seeds) - initial_seed_count
        session_hit = (session_new / attempts) if attempts else 0.0

        seeded_hit = (seeded_200 / seeded_attempts) if seeded_attempts else 0.0

        mode = "GLOBAL"
        if len(seeds_200_ids) >= SEEDING_START_AT:
            mode = f"SEEDED({SEED_PROBABILITY:.0%} @ ±{current_window:,})"

        print(
            f"📊 Attempts: {attempts:,} | 200 seeds: {len(seeds)}/{target_seeds_200} "
            f"| New this run: {session_new} | Session hit: {session_hit:.3%} "
            f"| Rate: {rate:.2f} req/s | Mode: {mode} | Seed-hit: {seeded_hit:.2%} "
            f"| Status: [{top}]",
            flush=True
        )
        last_status_time = now
        last_status_attempt = attempts

    def maybe_tune_window():
        """
        If seeded sampling is active but seed-hit is low, gradually expand window.
        (Tune no more often than every 2,000 attempts to avoid thrash.)
        """
        nonlocal current_window, last_window_tune_attempt

        if len(seeds_200_ids) < SEEDING_START_AT:
            return
        if attempts - last_window_tune_attempt < 2000:
            return

        last_window_tune_attempt = attempts

        if seeded_attempts < 200:  # not enough seeded data
            return

        seeded_hit = seeded_200 / seeded_attempts if seeded_attempts else 0.0

        # If seeded mode isn't doing well, expand the window (up to max)
        if seeded_hit < 0.003 and current_window < SEED_WINDOW_MAX:  # <0.3% seeded hit
            new_window = int(min(SEED_WINDOW_MAX, current_window * SEED_WINDOW_GROWTH))
            if new_window != current_window:
                current_window = new_window
                print(f"🛠️ Tuning: seeded hit={seeded_hit:.3%} -> expanding window to ±{current_window:,}", flush=True)

    while len(seeds) < target_seeds_200:
        league_id, was_seeded_pick = choose_league_id(seeds_200_ids, current_window)

        # Normalize candidate league_id to int (defensive)
        try:
            league_id = int(league_id)
        except Exception:
            continue

        if league_id in seen_ids:
            continue
        seen_ids.add(league_id)

        attempts += 1
        maybe_status()
        maybe_tune_window()

        league_json, status = fetch_league_json(league_id)
        status_counts[status] += 1

        if was_seeded_pick:
            seeded_attempts += 1

        # Respect delay regardless of outcome
        time.sleep(request_delay)

        # Record 401 exists-but-private (optional)
        if RECORD_401 and status == 401 and league_id not in exists_401_set and len(exists_401) < MAX_401_TO_KEEP:
            exists_401_set.add(league_id)
            exists_401.append({"league_id": league_id, "season": int(SEASON)})

            # checkpoint 401 sometimes
            if len(exists_401) - last_saved_401_count >= 250:
                save_csv(exists_401, OUT_EXISTS_401_CSV, dedupe_cols=["league_id", "season"])
                last_saved_401_count = len(exists_401)

        if status != 200 or not league_json:
            continue

        if was_seeded_pick:
            seeded_200 += 1

        meta = extract_seed_metadata(league_json) or {}

        # Normalize / enforce required fields
        meta["league_id"] = int(meta.get("league_id") or league_id)
        meta["season"] = int(meta.get("season") or SEASON)

        # avoid duplicates (int-safe)
        if meta["league_id"] in seeds_200_id_set:
            continue

        seeds_200_id_set.add(meta["league_id"])
        seeds_200_ids.append(meta["league_id"])
        seeds.append(meta)

        print(
            f"✅ Seed found (200): {meta['league_id']} | teams={meta.get('num_teams')} | "
            f"ppr={meta.get('ppr_points')} | total_pf={meta.get('total_pf')}",
            flush=True
        )

        # checkpoint save seeds
        if (len(seeds) - last_saved_seed_count) >= SAVE_EVERY_NEW_SEEDS:
            save_csv(seeds, OUT_SEEDS_CSV, dedupe_cols=["league_id", "season"])
            last_saved_seed_count = len(seeds)
            print(f"💾 Saved checkpoint: {OUT_SEEDS_CSV} (rows={last_saved_seed_count})", flush=True)

    # final save + summary
    maybe_status(force=True)
    seeds_df = save_csv(seeds, OUT_SEEDS_CSV, dedupe_cols=["league_id", "season"])
    if RECORD_401:
        save_csv(exists_401, OUT_EXISTS_401_CSV, dedupe_cols=["league_id", "season"])

    print(f"\n🎯 Done. Saved 200-seeds to: {OUT_SEEDS_CSV} (rows={len(seeds_df)})", flush=True)
    if RECORD_401:
        print(f"🗂️ Saved 401-exists list to: {OUT_EXISTS_401_CSV} (rows={len(exists_401)})", flush=True)

    return seeds_df



if __name__ == "__main__":
    seeds_df = discover_seeds_200()
    try:
        display(seeds_df.head(10))
    except NameError:
        print(seeds_df.head(10))
