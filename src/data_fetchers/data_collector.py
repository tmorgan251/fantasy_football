import sys, subprocess, importlib, os
import pandas as pd
import requests

class FantasyDataCollector:
    """
    A robust collector for historical ESPN Fantasy Football data.
    Uses a hybrid approach: espn-api for structure and raw REST for draft details.
    """
    def __init__(self, swid=None, espn_s2=None, verbose=False):
        # Not used in this script because the data we are pulling is public
        self.swid = swid
        self.espn_s2 = espn_s2

        # Used Variables
        self.verbose = verbose
        self._ensure_dependencies()
        self._debugged_types = set()
        
        # Set up data directory path (relative to this script)
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build path: go up 2 levels (from src/data_fetchers to project root), then into data/raw/espn
        self.data_dir = os.path.join(script_dir, '../../data/raw/espn')
        # Normalize the path (removes .. and . components, converts to absolute if needed)
        self.data_dir = os.path.normpath(self.data_dir)
        # Ensure directory exists (creates it if it doesn't, does nothing if it does)
        os.makedirs(self.data_dir, exist_ok=True)

    def _ensure_dependencies(self):
        """Install dependencies if they are not already installed."""
        for pkg in ['espn_api', 'pandas', 'requests']:
            try:
                importlib.import_module(pkg)
            except ImportError:
                print(f"! {pkg} not found. Installing...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                importlib.invalidate_caches()

        global League
        from espn_api.football import League

    def _get_csv_row_count(self, filename):
        """
        Fast row counting for final progress reporting.
        Uses generator expression to count lines without loading entire file into memory.
        """
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath): return 0
        with open(filepath, 'r', encoding='utf-8') as f:
            # Generator expression: sum(1 for line in f) counts lines efficiently
            # Returns count - 1 to exclude header row
            count = sum(1 for line in f)
            return max(0, count - 1)

    def _save_to_csv(self, data, filename):
        """
        Streaming save to prevent memory bloat during 100+ league runs.
        Uses append mode to add data incrementally rather than loading entire file.
        """
        if not data: return
        df = pd.DataFrame(data)
        filepath = os.path.join(self.data_dir, filename)
        file_exists = os.path.isfile(filepath)
        # Append mode ('a'): adds to existing file
        # header=not file_exists: only write header if file is new (first write)
        # This prevents duplicate headers when appending
        df.to_csv(filepath, mode='a', index=False, header=not file_exists)

    def _print_object_info(self, obj, obj_name):
        """Print all methods and attributes of an object."""
        if not self.verbose:
            return
        
        obj_type = type(obj).__name__
        print(f"\n{'='*60}")
        print(f"Object: {obj_name} (Type: {obj_type})")
        print(f"{'='*60}")
        
        # Get all attributes and methods
        all_attrs = dir(obj)
        
        # Filter out private attributes (starting with __)
        public_attrs = [attr for attr in all_attrs if not attr.startswith('__')]
        
        # Separate methods from attributes
        methods = []
        attributes = []
        
        for attr in public_attrs:
            try:
                value = getattr(obj, attr)
                if callable(value):
                    methods.append(attr)
                else:
                    # Try to get the value, but truncate if too long
                    try:
                        str_value = str(value)
                        if len(str_value) > 100:
                            str_value = str_value[:100] + "..."
                        attributes.append((attr, str_value))
                    except:
                        attributes.append((attr, "<unprintable>"))
            except:
                pass
        
        print(f"\nMethods ({len(methods)}):")
        for method in sorted(methods):
            print(f"  - {method}")
        
        print(f"\nAttributes ({len(attributes)}):")
        for attr, value in sorted(attributes):
            print(f"  - {attr}: {value}")
        print()

    def extract_league(self, league_id, year):
        """Main entry point for processing a league."""
        try:
            # Initialize the espn-api League object
            league = League(league_id=league_id, year=year, swid=self.swid, espn_s2=self.espn_s2)
            if self.verbose: 
                print(f"\n{'='*60}\nLEAGUE: {league.settings.name} ({year})\n{'='*60}")
                self._print_object_info(league, "league")

            # Step 1: Draft Data (via Raw API)
            draft_rows = self._get_draft_data(league, league_id, year)
            self._save_to_csv(draft_rows, "draft_data.csv")

            # Step 2: Seasonal Data (Lineups & Transactions)
            self._get_seasonal_data(league, league_id, year)

            # Audit counts
            print(f"\n✓ {league_id} Saved: Draft({len(draft_rows)})")
            print(f"  Current DB Size: Transactions({self._get_csv_row_count('transaction_data.csv')}) | Lineups({self._get_csv_row_count('lineup_data.csv')})")

        except Exception as e:
            print(f"✗ Failed League {league_id}: {e}")

    def _get_draft_data(self, league, lid, year):
        """
        Fetches draft data using the 'lm-api-reads' endpoint and 'seasons' path.
        This combination has proven successful for retrieving autoDraftTypeId without cookies.
        """
        rows = []
        if self.verbose: print("\n[SECTION 1: DRAFT DATA - RAW API HIT]")

        # API URL for draft data
        url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{lid}?view=mDraftDetail"
        
        # Can send cookies if you have them, but this URL often works without them for public leagues
        # cookies = {"swid": self.swid, "espn_s2": self.espn_s2}
        
        try:
            #r = requests.get(url, cookies=cookies, timeout=30)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()

            # The 'seasons' endpoint returns a direct dictionary, not a list
            # Safe dictionary access: if 'draftDetail' doesn't exist, use empty dict, then get 'picks' or empty list
            picks = data.get('draftDetail', {}).get('picks', [])
            
            # Map IDs to names using the league object (which is still useful for metadata)
            if self.verbose and league.draft:
                self._print_object_info(league.draft[0], "league.draft[0] (draft player)")
            # Dictionary comprehension: Create mapping from player ID to player name
            # This allows us to look up player names when we only have IDs from the raw API
            player_map = {p.playerId: p.playerName for p in league.draft}
            
            if self.verbose and league.teams:
                self._print_object_info(league.teams[0], "league.teams[0] (team)")

            if not picks and self.verbose:
                print("! API connected but 'picks' list is empty.")

            for p in picks:
                p_id = p.get('playerId')
                t_id = p.get('teamId')
                
                # THE GOLDEN TICKET: The raw autoDraftTypeId from ESPN API
                # This field indicates if a pick was autodrafted (non-zero) or manual (0)
                auto_id = p.get('autoDraftTypeId', 0)
                # Convert to binary: 1 if autodrafted (auto_id > 0), 0 if manual
                is_auto = 1 if auto_id > 0 else 0
                
                overall = p.get('overallPickNumber')

                if self.verbose and overall <= 5:
                    status = f"[AUTO (ID:{auto_id})]" if is_auto else "[MANUAL]"
                    # Fallback to ID if name mapping fails
                    p_name = player_map.get(p_id, f"Player {p_id}")
                    print(f"Pick {overall:03} {status:15} | {p_name:20} -> Team {t_id}")

                rows.append({
                    'League_ID': lid, 
                    'Year': year, 
                    'Player': player_map.get(p_id, f"ID:{p_id}"),
                    'Team': t_id,
                    'Round': p.get('roundId'), 
                    'Pick': p.get('roundPickNumber'), 
                    'Overall': overall, 
                    'Is_Autodrafted': is_auto, 
                    'Auto_Draft_Type_ID': auto_id
                })
        except Exception as e:
            if self.verbose: print(f"! Raw Draft API Error: {e}")
            
        return rows

    def _get_seasonal_data(self, league, lid, year):
        """Processes weekly data and identifies reciprocal trades."""
        master_rosters_prev = {}
        reg_weeks = getattr(league.settings, 'reg_season_count', 14)

        if self.verbose: print("\n[SECTION 4: SEASONAL TRANSACTION LOG]")

        # Process regular season weeks + 4 (to include playoffs/championship)
        for week in range(1, reg_weeks + 4):
            try:
                boxes = league.box_scores(week=week)
                # If no box scores returned, we've reached the end of available data
                if not boxes: break
                
                if self.verbose: 
                    print(f"\n--- Analyzing Week {week} ---")
                    if boxes:
                        self._print_object_info(boxes[0], f"league.box_scores(week={week})[0] (box score)")

                curr_week_trans, curr_week_lineups, curr_rosters = [], [], {}
                first_box_printed = False
                first_team_printed = False
                first_player_printed = False

                for box in boxes:
                    if self.verbose and not first_box_printed:
                        self._print_object_info(box, "box (from box_scores)")
                        first_box_printed = True
                    
                    # Process both teams in each matchup (home and away)
                    # Tuple unpacking in loop: iterate over list of (team, lineup) pairs
                    for team, lineup in [(box.home_team, box.home_lineup), (box.away_team, box.away_lineup)]:
                        if self.verbose and not first_team_printed:
                            self._print_object_info(team, "team (from box.home_team/box.away_team)")
                            first_team_printed = True
                        
                        t_id = team.team_id
                        # Set comprehension: Create a set of player names for this team's current roster
                        # Using a set allows fast lookup for roster comparisons later
                        curr_rosters[t_id] = {p.name for p in lineup}

                        for p in lineup:
                            if self.verbose and not first_player_printed:
                                self._print_object_info(p, "player (from lineup)")
                                first_player_printed = True
                            # Get player's lineup slot (QB, RB, WR, TE, FLEX, BE=bench, IR=injured reserve)
                            # getattr with default 'BE' handles cases where attribute might not exist
                            slot = getattr(p, 'lineupSlot', 'BE')
                            # Binary indicator: 1 if starter (not bench or IR), 0 otherwise
                            is_starter = 1 if slot not in ['BE', 'IR'] else 0
                            curr_week_lineups.append({
                                'League_ID': lid, 'Week': week, 'Team': t_id,
                                'Player': p.name, 'Slot': slot, 'Points': p.points, 
                                'Projected_Points': p.projected_points, 'Position': p.position,
                                'Is_Starter': is_starter
                            })

                if master_rosters_prev:
                    # TRANSACTION DETECTION: Compare current week rosters to previous week to identify adds/drops/trades
                    
                    # Create a set of ALL players that were on ANY team in the previous week
                    # This is used later to detect trades (if a player was on another team before)
                    all_prev = set().union(*master_rosters_prev.values())
                    
                    # Create a set of ALL players that are on ANY team in the current week
                    # This is used later to detect trades (if a departing player ends up on another team)
                    all_curr = set().union(*curr_rosters.values())
                    
                    # For each team, find players that ARRIVED (new this week)
                    # Set difference: current_roster - previous_roster = new players
                    arrivals = {tid: names - master_rosters_prev.get(tid, set()) for tid, names in curr_rosters.items()}
                    
                    # For each team, find players that DEPARTED (left this week)
                    # Set difference: previous_roster - current_roster = players who left
                    departures = {tid: master_rosters_prev.get(tid, set()) - names for tid, names in curr_rosters.items()}

                    for tid, joined_players in arrivals.items():
                        for p in joined_players:
                            # TRADE DETECTION LOGIC:
                            # A trade occurs when:
                            # 1. The arriving player was on SOME team last week (p in all_prev)
                            # 2. AND this team had a player leave who ended up on another team
                            #    (any departing player from this team is now in all_curr)
                            # This detects reciprocal player movement = trade
                            # Otherwise, it's a waiver add (player wasn't on any roster before)
                            is_trade = (p in all_prev) and any(d in all_curr for d in departures.get(tid, []))
                            action = 'TRADE_JOIN' if is_trade else 'WAIVER_ADD'
                            if self.verbose: print(f"  [{action:10}] {p:20} | To 'Team {tid}'")
                            curr_week_trans.append({'League_ID': lid, 'Week': week, 'Team': tid, 'Player': p, 'Action': action})

                        for p in departures.get(tid, []):
                            # Only record as DROP if player isn't on any team now
                            # If player is in all_curr, they were traded (handled above as TRADE_JOIN for receiving team)
                            if p not in all_curr:
                                if self.verbose: print(f"  [DROP      ] {p:20} | Left 'Team {tid}'")
                                curr_week_trans.append({'League_ID': lid, 'Week': week, 'Team': tid, 'Player': p, 'Action': 'DROP'})

                self._save_to_csv(curr_week_lineups, "lineup_data.csv")
                self._save_to_csv(curr_week_trans, "transaction_data.csv")
                master_rosters_prev = curr_rosters
            except: continue

# --- EXECUTION ---
if __name__ == "__main__":
#     # Add your Cookies here for private leagues
#     collector = FantasyDataCollector(swid=None, espn_s2=None, verbose=True)
#     collector.extract_league(1409356, 2024)
    import os
    import pandas as pd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "../.."))

    league_csv = os.path.join(project_root, "data", "raw", "espn", "targets_10team_ppr1_season_2024.csv")
    league_ids = pd.read_csv(league_csv)["league_id"].astype(int).tolist()

    collector = FantasyDataCollector(swid=None, espn_s2=None, verbose=False)

    base_dir = collector.data_dir  # this already points to <project_root>/data/raw/espn

    for year in range(2021, 2025):
        collector.data_dir = os.path.join(base_dir, str(year))
        os.makedirs(collector.data_dir, exist_ok=True)

        print(f"\n=== YEAR {year} -> writing to {collector.data_dir} ===")
        for lid in league_ids:
            print(f"Running league={lid} year={year}")
            collector.extract_league(lid, year)

    collector.data_dir = base_dir

