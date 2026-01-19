# league_harvest.py
"""
Harvest EXACTLY N ESPN fantasy leagues that match:
- 10 teams
- 1.0 PPR (receptions statId == 53 -> points == 1.0)

Uses previously discovered 200-seeds for fast yield.
Resumes from disk and checkpoints progress.
"""

from __future__ import annotations

import csv
import os
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import requests


# ----------------------------
# Config
# ----------------------------

SEASON = 2024
BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"

ID_RANGE_LOW = 1_000_000
ID_RANGE_HIGH = 20_000_000

SEED_WINDOW = 250_000
GLOBAL_EXPLORATION_PROB = 0.015

REQUEST_DELAY_START = 0.10
REQUEST_DELAY_MIN = 0.05
REQUEST_DELAY_MAX = 1.50
TIMEOUT = 6.0

MAX_RETRIES = 2
BACKOFF_BASE = 0.50
JITTER = 0.10

SAVE_EVERY_NEW_TARGETS = 25
SAVE_EVERY_NEW_401 = 200
STATUS_EVERY_ATTEMPTS = 200

DEFAULT_SEEDS_CSV = r"data/raw/ESPN/seeds_200_season_2024.csv"
DEFAULT_EXISTS_401_CSV = r"data/raw/ESPN/exists_401_season_2024.csv"
DEFAULT_OUT_TARGETS_CSV = r"data/raw/ESPN/targets_10team_ppr1_season_2024.csv"


# ----------------------------
# CSV helpers
# ----------------------------

def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_rows(path: str, rows: List[Dict[str, Any]], dedupe_cols: List[str]) -> None:
    _ensure_dir_for_file(path)
    if not rows:
        return

    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple(str(r.get(c, "")) for c in dedupe_cols)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    cols: List[str] = []
    for r in out:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)


# ----------------------------
# ESPN fetch
# ----------------------------

def fetch_league_json(
    league_id: int,
    season: int,
    timeout: float,
    session: requests.Session,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Return (json, status).
    status is ALWAYS int:
      - 200 success
      - 401/404/etc passthrough
      - -1 timeout
      - 0 other error/decode failure
    """
    url = f"{BASE_URL}/seasons/{season}/segments/0/leagues/{league_id}"
    params = [("view", "mSettings"), ("view", "mTeam")]

    try:
        r = session.get(url, params=params, timeout=timeout, allow_redirects=False)
    except requests.Timeout:
        return None, -1
    except requests.RequestException:
        return None, 0

    if r.status_code != 200:
        return None, int(r.status_code)

    try:
        return r.json(), 200
    except ValueError:
        return None, 0


def fetch_with_retries(
    league_id: int,
    season: int,
    timeout: float,
    session: requests.Session,
) -> Tuple[Optional[Dict[str, Any]], int]:
    retryable = {-1, 0, 429, 500, 502, 503, 504}
    last_status = 0

    for attempt in range(MAX_RETRIES + 1):
        j, status = fetch_league_json(league_id, season, timeout, session)
        last_status = status

        if status == 200:
            return j, 200
        if status not in retryable:
            return None, status

        if attempt < MAX_RETRIES:
            time.sleep((BACKOFF_BASE * (2 ** attempt)) + random.random() * JITTER)

    return None, last_status


# ----------------------------
# Target logic
# ----------------------------

def extract_meta(league_json: Dict[str, Any], league_id: int, season: int) -> Dict[str, Any]:
    settings = league_json.get("settings") or {}
    scoring = settings.get("scoringSettings") or {}
    num_teams = settings.get("size")

    ppr_points = None
    items = scoring.get("scoringItems") or []
    for it in items:
        if it.get("statId") in (53, "53"):  # receptions
            ppr_points = it.get("points")
            break

    return {
        "league_id": int(league_id),
        "season": int(season),
        "num_teams": num_teams,
        "ppr_points": ppr_points,
    }


def is_target(meta: Dict[str, Any]) -> bool:
    try:
        n = meta.get("num_teams")
        p = meta.get("ppr_points")
        if n is None or p is None:
            return False
        return (int(n) == 10) and (abs(float(p) - 1.0) < 1e-9)
    except Exception:
        return False


# ----------------------------
# Sampling + state
# ----------------------------

@dataclass
class State:
    attempts: int = 0
    targets: int = 0
    delay: float = REQUEST_DELAY_START


def choose_id(seed_ids: List[int], rng: random.Random) -> int:
    if (not seed_ids) or (rng.random() < GLOBAL_EXPLORATION_PROB):
        return rng.randint(ID_RANGE_LOW, ID_RANGE_HIGH)

    seed = rng.choice(seed_ids)
    offset = rng.randint(-SEED_WINDOW, SEED_WINDOW)
    return max(1, seed + offset)


def adapt_delay(state: State, recent: Deque[int]) -> None:
    if len(recent) < recent.maxlen:
        return

    c = Counter(recent)
    throttled = c.get(429, 0)
    server_err = sum(c.get(x, 0) for x in (500, 502, 503, 504))
    timeouts = c.get(-1, 0)

    if throttled > 0 or server_err >= 2 or timeouts >= 2:
        state.delay = min(REQUEST_DELAY_MAX, max(state.delay * 1.25, state.delay + 0.05))
    else:
        state.delay = max(REQUEST_DELAY_MIN, state.delay * 0.97)


# ----------------------------
# Main
# ----------------------------

def harvest_target_leagues(
    target_n: int = 500,
    season: int = SEASON,
    seeds_csv: str = DEFAULT_SEEDS_CSV,
    out_targets_csv: str = DEFAULT_OUT_TARGETS_CSV,
    exists_401_csv: str = DEFAULT_EXISTS_401_CSV,
    timeout: float = TIMEOUT,
    rng_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(rng_seed)

    # Load seed IDs
    seed_rows = read_csv_rows(seeds_csv)
    seed_ids: List[int] = []
    for r in seed_rows:
        lid = r.get("league_id")
        if not lid:
            continue
        try:
            seed_ids.append(int(float(lid)))
        except Exception:
            pass
    seed_ids = sorted(set(seed_ids))
    if not seed_ids:
        raise RuntimeError(f"No seed IDs found in {seeds_csv}")

    # Resume targets
    targets = read_csv_rows(out_targets_csv)
    targets_by_id: Dict[int, Dict[str, Any]] = {}
    for r in targets:
        try:
            lid = int(float(r.get("league_id")))
            targets_by_id[lid] = r
        except Exception:
            pass
    targets = list(targets_by_id.values())
    target_ids = set(targets_by_id.keys())

    # Resume 401s
    exists_401_rows = read_csv_rows(exists_401_csv)
    exists_401_set = set()
    exists_401: List[Dict[str, Any]] = []
    for r in exists_401_rows:
        try:
            lid = int(float(r.get("league_id")))
            exists_401_set.add(lid)
            exists_401.append({"league_id": lid, "season": season})
        except Exception:
            pass

    seen_ids = set(target_ids) | exists_401_set

    state = State(targets=len(targets), delay=REQUEST_DELAY_START)
    status_counts = Counter()
    recent: Deque[int] = deque(maxlen=250)

    print(f"↩️ Resuming targets: {len(targets)} (file: {out_targets_csv})", flush=True)
    print(f"↩️ Resuming 401s: {len(exists_401)} (file: {exists_401_csv})", flush=True)
    print(f"🌱 Seeds loaded: {len(seed_ids)} (file: {seeds_csv})", flush=True)
    print(f"🎯 Target leagues: {target_n} | Season: {season}", flush=True)
    print(f"⚙️ Delay={state.delay}s | Timeout={timeout}s | Window=±{SEED_WINDOW:,} | GlobalExplore={GLOBAL_EXPLORATION_PROB:.1%}\n", flush=True)

    session_headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fantasy.espn.com/",
        "Origin": "https://fantasy.espn.com",
        "Connection": "keep-alive",
    }

    saved_targets = len(targets)
    saved_401 = len(exists_401)

    with requests.Session() as session:
        session.headers.update(session_headers)

        while state.targets < target_n:
            league_id = choose_id(seed_ids, rng)

            if league_id in seen_ids:
                continue
            seen_ids.add(league_id)

            state.attempts += 1

            league_json, status = fetch_with_retries(league_id, season, timeout, session)

            status_counts[status] += 1
            recent.append(status)
            adapt_delay(state, recent)

            # record 401s
            if status == 401 and league_id not in exists_401_set:
                exists_401_set.add(league_id)
                exists_401.append({"league_id": league_id, "season": season})

                if len(exists_401) - saved_401 >= SAVE_EVERY_NEW_401:
                    write_csv_rows(exists_401_csv, exists_401, dedupe_cols=["league_id", "season"])
                    saved_401 = len(exists_401)
                    print(f"💾 Saved 401s: {saved_401}", flush=True)

            # only keep targets
            if status == 200 and league_json:
                meta = extract_meta(league_json, league_id, season)
                if is_target(meta):
                    lid = meta["league_id"]
                    if lid not in target_ids:
                        target_ids.add(lid)
                        targets.append({"league_id": lid, "season": season})
                        state.targets += 1
                        print(f"✅ Target {state.targets}/{target_n}: {lid}", flush=True)

                        if len(targets) - saved_targets >= SAVE_EVERY_NEW_TARGETS:
                            write_csv_rows(out_targets_csv, targets, dedupe_cols=["league_id", "season"])
                            saved_targets = len(targets)
                            print(f"💾 Saved targets: {saved_targets}", flush=True)

            if state.attempts % STATUS_EVERY_ATTEMPTS == 0:
                top = ", ".join(f"{k}:{v}" for k, v in status_counts.most_common(5))
                hit = (state.targets / state.attempts) if state.attempts else 0.0
                print(
                    f"📊 Attempts={state.attempts:,} | Targets={state.targets}/{target_n} | Target-hit={hit:.3%} "
                    f"| Delay={state.delay:.2f}s | Status=[{top}]",
                    flush=True,
                )

            time.sleep(state.delay)

    write_csv_rows(out_targets_csv, targets, dedupe_cols=["league_id", "season"])
    write_csv_rows(exists_401_csv, exists_401, dedupe_cols=["league_id", "season"])

    print(f"\n🎯 Done. Saved targets: {out_targets_csv} (rows={len(targets)})", flush=True)
    print(f"🗂️ Saved/updated 401s: {exists_401_csv} (rows={len(exists_401)})", flush=True)

    return targets


if __name__ == "__main__":
    harvest_target_leagues(target_n=500, season=SEASON)
