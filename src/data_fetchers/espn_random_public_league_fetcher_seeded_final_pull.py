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

# Return type annotation (-> None): Indicates this function doesn't return a value.
# It performs a side effect (creates directory) but returns nothing. This helps
# catch bugs if someone tries to use the return value, and makes the function's
# purpose clearer.
def _ensure_dir_for_file(path: str) -> None:
    """
    Ensure the directory for a file path exists, creating it if necessary.
    
    Args:
        path: File path whose directory should be created
    """
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# Return type annotation (-> List[Dict[str, Any]]): Documents that this returns a list
# of dictionaries. The 'Any' means dictionary values can be any type (strings, numbers, etc).
# This annotation helps IDEs provide autocomplete when you use the return value, and
# type checkers can verify you're using it correctly.
def read_csv_rows(path: str) -> List[Dict[str, Any]]:
    """
    Read all rows from a CSV file into a list of dictionaries.
    
    Args:
        path: Path to CSV file
        
    Returns:
        List of dictionaries, one per row. Returns empty list if file doesn't exist.
    """
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        # List comprehension: Convert each CSV row (OrderedDict) to regular dict
        # DictReader returns OrderedDict objects, converting to dict makes them easier to work with
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_rows(path: str, rows: List[Dict[str, Any]], dedupe_cols: List[str]) -> None:
    """
    Write rows to CSV file, removing duplicates based on specified columns.
    
    Deduplication works by creating a tuple of values from the specified columns
    for each row. If a row's tuple has been seen before, it's skipped.
    
    Args:
        path: Output CSV file path
        rows: List of dictionaries to write
        dedupe_cols: Column names to use for deduplication
    """
    _ensure_dir_for_file(path)
    if not rows:
        return

    # Deduplicate rows based on specified columns
    # Create a tuple key from the dedupe columns for each row
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = tuple(str(r.get(c, "")) for c in dedupe_cols)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    # Collect all unique column names from the deduplicated rows
    cols_set = set()
    for r in out:
        cols_set.update(r.keys())
    cols = list(cols_set)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)


# ----------------------------
# ESPN fetch
# ----------------------------

# Return type annotation (-> Tuple[Optional[Dict[str, Any]], int]): Documents that this
# returns a tuple with two elements: (1) an optional dictionary (could be None on error),
# and (2) an integer status code. This is crucial because callers need to know the function
# returns a tuple, not just a dict, and they need to handle the None case.
def fetch_league_json(
    league_id: int,
    season: int,
    timeout: float,
    session: requests.Session,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """
    Fetch league JSON data from ESPN API.
    
    Args:
        league_id: ESPN fantasy league ID
        season: Season year
        timeout: Request timeout in seconds
        session: Requests session object with headers configured
        
    Returns:
        Tuple of (json_data, status_code):
        - json_data: League JSON dict if successful, None otherwise
        - status_code: HTTP status code (200 for success, 401/404/etc for errors),
                      -1 for timeout, 0 for other errors/decode failures
    """
    url = f"{BASE_URL}/seasons/{season}/segments/0/leagues/{league_id}"
    # Request both settings and team views in a single API call
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
    """
    Fetch league JSON with automatic retries for transient errors.
    
    Uses exponential backoff with jitter for retries. Only retries on transient
    errors (timeouts, rate limits, server errors). Non-retryable errors (401, 404)
    are returned immediately.
    
    Args:
        league_id: ESPN fantasy league ID
        season: Season year
        timeout: Request timeout in seconds
        session: Requests session object
        
    Returns:
        Tuple of (json_data, status_code) - same format as fetch_league_json
    """
    # Status codes that should trigger a retry
    retryable = {-1, 0, 429, 500, 502, 503, 504}
    last_status = 0

    for attempt in range(MAX_RETRIES + 1):
        j, status = fetch_league_json(league_id, season, timeout, session)
        last_status = status

        if status == 200:
            return j, 200
        # Non-retryable errors (401, 404, etc.) return immediately
        if status not in retryable:
            return None, status

        # Exponential backoff: wait longer on each retry attempt
        # Add random jitter to prevent thundering herd
        if attempt < MAX_RETRIES:
            time.sleep((BACKOFF_BASE * (2 ** attempt)) + random.random() * JITTER)

    return None, last_status


# ----------------------------
# Target logic
# ----------------------------

def extract_meta(league_json: Dict[str, Any], league_id: int, season: int) -> Dict[str, Any]:
    """
    Extract key metadata from league JSON response.
    
    Parses the nested JSON structure to find league size and PPR scoring settings.
    PPR is identified by statId 53 (receptions).
    
    Args:
        league_json: League JSON response from ESPN API
        league_id: League ID (for output)
        season: Season year (for output)
        
    Returns:
        Dictionary with league_id, season, num_teams, and ppr_points
    """
    settings = league_json.get("settings") or {}
    scoring = settings.get("scoringSettings") or {}
    num_teams = settings.get("size")

    # Find PPR points by searching for statId 53 (receptions)
    # Handle both int and string versions of statId
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


# Return type annotation (-> bool): Documents that this returns True or False.
# This is especially important for boolean functions because the return value
# is often used in if statements - the annotation makes it clear this is a predicate function.
def is_target(meta: Dict[str, Any]) -> bool:
    """
    Check if league matches target criteria: 10 teams and 1.0 PPR.
    
    Uses floating point comparison with small epsilon for PPR to handle
    potential rounding differences in API responses.
    
    Args:
        meta: Metadata dictionary from extract_meta()
        
    Returns:
        True if league is 10 teams with exactly 1.0 PPR, False otherwise
    """
    try:
        n = meta.get("num_teams")
        p = meta.get("ppr_points")
        if n is None or p is None:
            return False
        # Use epsilon comparison for floating point PPR value
        return (int(n) == 10) and (abs(float(p) - 1.0) < 1e-9)
    except Exception:
        return False


# ----------------------------
# Sampling + state
# ----------------------------

# @dataclass decorator: Automatically generates __init__, __repr__, __eq__, and other
# special methods for this class. This saves boilerplate code - instead of writing
# __init__(self, attempts=0, targets=0, delay=...) manually, dataclass does it for us.
@dataclass
class State:
    """
    Tracks harvesting state: attempt count, target count, and adaptive delay.
    """
    attempts: int = 0
    targets: int = 0
    delay: float = REQUEST_DELAY_START


def choose_id(seed_ids: List[int], rng: random.Random) -> int:
    """
    Choose next league ID to probe using seeded sampling strategy.
    
    With probability GLOBAL_EXPLORATION_PROB, explores randomly in the full ID range.
    Otherwise, picks a random seed and samples within SEED_WINDOW of it.
    This balances exploitation (near known good IDs) with exploration (finding new regions).
    
    Args:
        seed_ids: List of known good league IDs to use as seeds
        rng: Random number generator instance
        
    Returns:
        League ID to probe next
    """
    # Small chance to explore globally instead of near seeds
    if (not seed_ids) or (rng.random() < GLOBAL_EXPLORATION_PROB):
        return rng.randint(ID_RANGE_LOW, ID_RANGE_HIGH)

    # Pick a random seed and sample within window around it
    seed = rng.choice(seed_ids)
    offset = rng.randint(-SEED_WINDOW, SEED_WINDOW)
    return max(1, seed + offset)


def adapt_delay(state: State, recent: Deque[int]) -> None:
    """
    Adaptively adjust request delay based on recent error patterns.
    
    Analyzes the last 250 request statuses. If there are rate limits (429),
    server errors (5xx), or timeouts, increases delay. Otherwise, gradually
    decreases delay to speed up when things are working well.
    
    Args:
        state: State object to update delay in
        recent: Deque of recent status codes (last 250 requests)
    """
    # Need full window of data before adapting
    if len(recent) < recent.maxlen:
        return

    # Count error types in recent window
    c = Counter(recent)
    throttled = c.get(429, 0)
    server_err = sum(c.get(x, 0) for x in (500, 502, 503, 504))
    timeouts = c.get(-1, 0)

    # Increase delay if seeing problems
    if throttled > 0 or server_err >= 2 or timeouts >= 2:
        # Increase by 25% or at least 0.05s, capped at max
        state.delay = min(REQUEST_DELAY_MAX, max(state.delay * 1.25, state.delay + 0.05))
    else:
        # Gradually decrease delay when things are working (3% reduction)
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
    """
    Harvest target leagues from ESPN API matching criteria (10 teams, 1.0 PPR).
    
    Uses a seeded sampling strategy: starts with known good league IDs (seeds) and
    samples nearby IDs. Resumes from checkpoint files, so can be interrupted and
    restarted. Saves progress periodically to avoid data loss.
    
    Args:
        target_n: Number of target leagues to find
        season: Season year to search
        seeds_csv: Path to CSV with seed league IDs
        out_targets_csv: Path to save target leagues
        exists_401_csv: Path to save leagues that returned 401 (private)
        timeout: Request timeout in seconds
        rng_seed: Random seed for reproducibility
        
    Returns:
        List of target league dictionaries with league_id and season
        
    Raises:
        RuntimeError: If no seed IDs found in seeds_csv
    """
    rng = random.Random(rng_seed)

    # Load seed IDs from CSV (previously discovered good league IDs)
    seed_rows = read_csv_rows(seeds_csv)
    seed_ids: List[int] = []
    for r in seed_rows:
        lid = r.get("league_id")
        if not lid:
            continue
        # Handle both string and numeric league IDs
        try:
            seed_ids.append(int(float(lid)))
        except Exception:
            pass
    seed_ids = sorted(set(seed_ids))
    if not seed_ids:
        raise RuntimeError(f"No seed IDs found in {seeds_csv}")

    # Resume previously found targets (allows interruption/resume)
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

    # Resume previously seen 401s (private leagues we've already checked)
    # Track these to avoid re-checking them
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

    # Combine all seen IDs to avoid duplicate checks
    seen_ids = set(target_ids) | exists_401_set

    # Initialize state tracking
    state = State(targets=len(targets), delay=REQUEST_DELAY_START)
    status_counts = Counter()
    # Track last 250 status codes for adaptive delay calculation
    recent: Deque[int] = deque(maxlen=250)

    print(f"Resuming targets: {len(targets)} (file: {out_targets_csv})", flush=True)
    print(f"Resuming 401s: {len(exists_401)} (file: {exists_401_csv})", flush=True)
    print(f"Seeds loaded: {len(seed_ids)} (file: {seeds_csv})", flush=True)
    print(f"Target leagues: {target_n} | Season: {season}", flush=True)
    print(f"Delay={state.delay}s | Timeout={timeout}s | Window=±{SEED_WINDOW:,} | GlobalExplore={GLOBAL_EXPLORATION_PROB:.1%}\n", flush=True)

    # Configure session headers to mimic browser requests
    session_headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fantasy.espn.com/",
        "Origin": "https://fantasy.espn.com",
        "Connection": "keep-alive",
    }

    # Track last save point for periodic checkpointing
    saved_targets = len(targets)
    saved_401 = len(exists_401)

    with requests.Session() as session:
        session.headers.update(session_headers)

        # Main harvesting loop
        while state.targets < target_n:
            # Choose next league ID using seeded sampling
            league_id = choose_id(seed_ids, rng)

            # Skip if we've already checked this ID
            if league_id in seen_ids:
                continue
            seen_ids.add(league_id)

            state.attempts += 1

            # Fetch league data with retries
            league_json, status = fetch_with_retries(league_id, season, timeout, session)

            # Track status for adaptive delay and statistics
            status_counts[status] += 1
            recent.append(status)
            adapt_delay(state, recent)

            # Record 401s (private leagues) to avoid re-checking
            if status == 401 and league_id not in exists_401_set:
                exists_401_set.add(league_id)
                exists_401.append({"league_id": league_id, "season": season})

                # Periodic checkpoint: save 401s every N new entries
                if len(exists_401) - saved_401 >= SAVE_EVERY_NEW_401:
                    write_csv_rows(exists_401_csv, exists_401, dedupe_cols=["league_id", "season"])
                    saved_401 = len(exists_401)
                    print(f"Saved 401s: {saved_401}", flush=True)

            # Check if successful response matches target criteria
            if status == 200 and league_json:
                meta = extract_meta(league_json, league_id, season)
                if is_target(meta):
                    lid = meta["league_id"]
                    if lid not in target_ids:
                        target_ids.add(lid)
                        targets.append({"league_id": lid, "season": season})
                        state.targets += 1
                        print(f"Target {state.targets}/{target_n}: {lid}", flush=True)

                        # Periodic checkpoint: save targets every N new entries
                        if len(targets) - saved_targets >= SAVE_EVERY_NEW_TARGETS:
                            write_csv_rows(out_targets_csv, targets, dedupe_cols=["league_id", "season"])
                            saved_targets = len(targets)
                            print(f"Saved targets: {saved_targets}", flush=True)

            # Periodic status update
            if state.attempts % STATUS_EVERY_ATTEMPTS == 0:
                top = ", ".join(f"{k}:{v}" for k, v in status_counts.most_common(5))
                hit = (state.targets / state.attempts) if state.attempts else 0.0
                print(
                    f"Attempts={state.attempts:,} | Targets={state.targets}/{target_n} | Target-hit={hit:.3%} "
                    f"| Delay={state.delay:.2f}s | Status=[{top}]",
                    flush=True,
                )

            # Adaptive delay between requests
            time.sleep(state.delay)

    write_csv_rows(out_targets_csv, targets, dedupe_cols=["league_id", "season"])
    write_csv_rows(exists_401_csv, exists_401, dedupe_cols=["league_id", "season"])

    print(f"\nDone. Saved targets: {out_targets_csv} (rows={len(targets)})", flush=True)
    print(f"Saved/updated 401s: {exists_401_csv} (rows={len(exists_401)})", flush=True)

    return targets


if __name__ == "__main__":
    harvest_target_leagues(target_n=500, season=SEASON)
