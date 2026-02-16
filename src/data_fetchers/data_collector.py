import sys, subprocess, importlib, os
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class FantasyDataCollector:
    """
    A robust collector for historical ESPN Fantasy Football data.
    Uses a hybrid approach: espn-api for structure and raw REST for draft details.
    """
    def __init__(self, swid=None, espn_s2=None, verbose=False):
        """Initialize data collector. Credentials optional for public leagues."""
        self.swid = swid
        self.espn_s2 = espn_s2
        self.verbose = verbose
        self._ensure_dependencies()
        
        # Set up data directory: project_root/data/raw/espn
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.normpath(os.path.join(script_dir, '../../data/raw/espn'))
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Thread lock ensures only one thread writes to CSV at a time
        self._write_lock = Lock()

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
        """Count rows in CSV file (excluding header) without loading into memory."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath): 
            return 0
        with open(filepath, 'r', encoding='utf-8') as f:
            return max(0, sum(1 for line in f) - 1)  # Subtract 1 for header

    def _save_to_csv(self, data, filename):
        """
        Append data to CSV file in thread-safe manner.
        
        Uses append mode to avoid loading entire file. Lock prevents race conditions
        where multiple threads could write headers simultaneously.
        """
        if not data: 
            return
        df = pd.DataFrame(data)
        filepath = os.path.join(self.data_dir, filename)
        
        with self._write_lock:
            # Check file existence inside lock to prevent race condition
            file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0
            df.to_csv(filepath, mode='a', index=False, header=not file_exists)

    def _print_object_info(self, obj, obj_name):
        """Print all public methods and attributes of an object (debugging helper)."""
        if not self.verbose:
            return
        
        obj_type = type(obj).__name__
        print(f"\n{'='*60}")
        print(f"Object: {obj_name} (Type: {obj_type})")
        print(f"{'='*60}")
        
        all_attrs = dir(obj)
        public_attrs = [attr for attr in all_attrs if not attr.startswith('__')]
        
        methods = []
        attributes = []
        
        for attr in public_attrs:
            try:
                value = getattr(obj, attr)
                if callable(value):
                    methods.append(attr)
                else:
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

    def _league_already_processed(self, league_id, year):
        """Check if league/year combination exists in draft_data.csv."""
        draft_path = os.path.join(self.data_dir, 'draft_data.csv')
        if not os.path.exists(draft_path):
            return False
        
        try:
            df = pd.read_csv(draft_path, usecols=['League_ID', 'Year'])
            return ((df['League_ID'] == league_id) & (df['Year'] == year)).any()
        except:
            return False
    
    def extract_league(self, league_id, year, skip_existing=True):
        """
        Main entry point for processing a league.
        
        Args:
            league_id: ESPN league ID
            year: Season year
            skip_existing: If True, skip if league/year already processed
        """
        # Skip if already processed
        if skip_existing and self._league_already_processed(league_id, year):
            print(f"⊘ Skipping {league_id}/{year} (already processed)")
            return
        
        try:
            league = League(league_id=league_id, year=year, swid=self.swid, espn_s2=self.espn_s2)
            if self.verbose: 
                print(f"\n{'='*60}\nLEAGUE: {league.settings.name} ({year})\n{'='*60}")
                self._print_object_info(league, "league")

            # Extract draft data via raw API (needed for autoDraftTypeId)
            draft_rows = self._get_draft_data(league, league_id, year)
            self._save_to_csv(draft_rows, "draft_data.csv")

            # Extract weekly lineup and transaction data (batched for performance)
            self._get_seasonal_data(league, league_id, year, batch_writes=True)

            print(f"{league_id}/{year} Saved: Draft({len(draft_rows)})")

        except Exception as e:
            print(f"Failed League {league_id}/{year}: {e}")

    def _get_draft_data(self, league, lid, year):
        """
        Fetch draft data via raw ESPN API to get autoDraftTypeId.
        
        The espn-api library doesn't expose autoDraftTypeId, so we use the raw
        REST API endpoint which provides this critical field for identifying autodrafted picks.
        """
        rows = []
        if self.verbose: 
            print("\n[SECTION 1: DRAFT DATA - RAW API HIT]")

        url = f"https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{year}/segments/0/leagues/{lid}?view=mDraftDetail"
        
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            picks = data.get('draftDetail', {}).get('picks', [])
            
            # Map player IDs to names using espn-api league object
            if self.verbose and league.draft:
                self._print_object_info(league.draft[0], "league.draft[0] (draft player)")
            player_map = {p.playerId: p.playerName for p in league.draft}
            
            if self.verbose and league.teams:
                self._print_object_info(league.teams[0], "league.teams[0] (team)")

            if not picks and self.verbose:
                print("! API connected but 'picks' list is empty.")

            for p in picks:
                p_id = p.get('playerId')
                t_id = p.get('teamId')
                
                # autoDraftTypeId: non-zero = autodrafted, 0 = manual pick
                auto_id = p.get('autoDraftTypeId', 0)
                is_auto = 1 if auto_id > 0 else 0
                overall = p.get('overallPickNumber')

                if self.verbose and overall <= 5:
                    status = f"[AUTO (ID:{auto_id})]" if is_auto else "[MANUAL]"
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
            if self.verbose: 
                print(f"! Raw Draft API Error: {e}")
            
        return rows

    def _get_seasonal_data(self, league, lid, year, batch_writes=False):
        """
        Processes weekly data and identifies reciprocal trades.
        
        Args:
            league: ESPN League object
            lid: League ID
            year: Season year
            batch_writes: If True, collect all data and write once at end (faster)
        """
        master_rosters_prev = {}
        reg_weeks = getattr(league.settings, 'reg_season_count', 14)

        if self.verbose: 
            print("\n[SECTION 4: SEASONAL TRANSACTION LOG]")
        
        # Collect all data first for batch writing
        all_lineups = []
        all_transactions = []

        # Process regular season + playoffs (reg_weeks + 4 weeks)
        for week in range(1, reg_weeks + 4):
            try:
                boxes = league.box_scores(week=week)
                if not boxes: 
                    break  # End of available data
                
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
                    
                    # Process both teams in matchup (home and away)
                    for team, lineup in [(box.home_team, box.home_lineup), (box.away_team, box.away_lineup)]:
                        if self.verbose and not first_team_printed:
                            self._print_object_info(team, "team (from box.home_team/box.away_team)")
                            first_team_printed = True
                        
                        t_id = team.team_id
                        # Store roster as set for fast lookup during transaction detection
                        curr_rosters[t_id] = {p.name for p in lineup}

                        for p in lineup:
                            if self.verbose and not first_player_printed:
                                self._print_object_info(p, "player (from lineup)")
                                first_player_printed = True
                            
                            slot = getattr(p, 'lineupSlot', 'BE')
                            is_starter = 1 if slot not in ['BE', 'IR'] else 0
                            curr_week_lineups.append({
                                'League_ID': lid, 'Week': week, 'Team': t_id,
                                'Player': p.name, 'Slot': slot, 'Points': p.points, 
                                'Projected_Points': p.projected_points, 'Position': p.position,
                                'Is_Starter': is_starter
                            })

                if master_rosters_prev:
                    # Detect transactions by comparing current vs previous week rosters
                    # set().union(*dict.values()) combines all sets from dictionary values into one set
                    # The * unpacks the dictionary values (which are sets) as arguments to union()
                    all_prev = set().union(*master_rosters_prev.values())  # All players from last week
                    all_curr = set().union(*curr_rosters.values())  # All players this week
                    
                    # Dictionary comprehensions: Find players who joined/left each team
                    # arrivals: set difference (current - previous) = new players this week
                    # departures: set difference (previous - current) = players who left
                    arrivals = {tid: names - master_rosters_prev.get(tid, set()) 
                               for tid, names in curr_rosters.items()}
                    departures = {tid: master_rosters_prev.get(tid, set()) - names 
                                 for tid, names in curr_rosters.items()}

                    for tid, joined_players in arrivals.items():
                        for p in joined_players:
                            # Trade detection logic:
                            # 1. Player was on another team last week (p in all_prev)
                            # 2. AND this team had a player leave (departures.get(tid, []))
                            # 3. AND that departed player is now on another team (d in all_curr)
                            # If all true, it's a trade; otherwise it's a waiver add
                            # any() returns True if any departed player is found on another team
                            is_trade = (p in all_prev) and any(d in all_curr for d in departures.get(tid, []))
                            action = 'TRADE_JOIN' if is_trade else 'WAIVER_ADD'
                            if self.verbose: 
                                print(f"  [{action:10}] {p:20} | To 'Team {tid}'")
                            curr_week_trans.append({
                                'League_ID': lid, 'Week': week, 'Team': tid, 
                                'Player': p, 'Action': action
                            })

                    for tid, left_players in departures.items():
                        for p in left_players:
                            # Only record as DROP if player isn't on any team now (wasn't traded)
                            if p not in all_curr:
                                if self.verbose: 
                                    print(f"  [DROP      ] {p:20} | Left 'Team {tid}'")
                                curr_week_trans.append({
                                    'League_ID': lid, 'Week': week, 'Team': tid, 
                                    'Player': p, 'Action': 'DROP'
                                })

                # Batch or write immediately
                if batch_writes:
                    all_lineups.extend(curr_week_lineups)
                    all_transactions.extend(curr_week_trans)
                else:
                    self._save_to_csv(curr_week_lineups, "lineup_data.csv")
                    self._save_to_csv(curr_week_trans, "transaction_data.csv")
                
                master_rosters_prev = curr_rosters
            except: 
                continue
        
        # Write all batched data at once
        if batch_writes:
            self._save_to_csv(all_lineups, "lineup_data.csv")
            self._save_to_csv(all_transactions, "transaction_data.csv")


def collect_leagues_parallel(
    league_ids,
    years,
    max_workers=10,
    skip_existing=True,
    swid=None,
    espn_s2=None,
    verbose=False,
    data_dir=None
):
    """
    High-level function to collect data from multiple leagues and years in parallel.
    
    This function handles all the complexity of parallel processing, progress tracking,
    and error handling. Users just need to provide league IDs and years.
    
    Args:
        league_ids: List of league IDs (integers) or path to CSV file with 'league_id' column
        years: List of years (integers) or single year, or range like range(2021, 2025)
        max_workers: Number of parallel workers (default: 10)
        skip_existing: If True, skip already-processed league/year combinations (default: True)
        swid: ESPN SWID cookie (optional, for private leagues)
        espn_s2: ESPN S2 cookie (optional, for private leagues)
        verbose: If True, print detailed progress (default: False)
        data_dir: Custom data directory (optional, defaults to data/raw/espn)
    
    Returns:
        dict with summary statistics:
        {
            'total_processed': int,
            'total_succeeded': int,
            'total_failed': int,
            'total_time_minutes': float,
            'results_by_year': {year: {'succeeded': int, 'failed': int}}
        }
    
    Example:
        # Simple usage - from CSV file
        results = collect_leagues_parallel(
            league_ids='data/raw/espn/targets_10team_ppr1_season_2024.csv',
            years=range(2021, 2025)
        )
        
        # Or with list of IDs
        results = collect_leagues_parallel(
            league_ids=[1374603, 1382426, 995098],
            years=[2024]
        )
    """
    import time
    
    # Normalize league_ids input (CSV path, list, or iterable)
    if isinstance(league_ids, str):
        if not os.path.exists(league_ids):
            raise FileNotFoundError(f"League IDs file not found: {league_ids}")
        df = pd.read_csv(league_ids)
        if 'league_id' not in df.columns:
            raise ValueError(f"CSV file must have 'league_id' column. Found columns: {df.columns.tolist()}")
        league_ids = df['league_id'].astype(int).tolist()
    elif hasattr(league_ids, '__iter__') and not isinstance(league_ids, str):
        league_ids = list(league_ids)
    else:
        raise TypeError(f"league_ids must be a list, CSV file path, or DataFrame. Got: {type(league_ids)}")
    
    # Normalize years input (int, list, or range)
    if isinstance(years, int):
        years = [years]
    elif isinstance(years, range):
        years = list(years)
    elif not isinstance(years, (list, tuple)):
        raise TypeError(f"years must be an int, list, or range. Got: {type(years)}")
    
    # Create base collector to get default data directory
    base_collector = FantasyDataCollector(swid=swid, espn_s2=espn_s2, verbose=verbose)
    if data_dir:
        base_dir = data_dir
    else:
        base_dir = base_collector.data_dir
    
    # Thread-safe wrapper for parallel processing
    def process_league_year(args):
        """Process single league/year combination (thread-safe)."""
        lid, year, year_data_dir = args
        try:
            collector = FantasyDataCollector(swid=swid, espn_s2=espn_s2, verbose=verbose)
            collector.data_dir = year_data_dir
            collector.extract_league(lid, year, skip_existing=skip_existing)
            return (lid, year, True, None)
        except Exception as e:
            return (lid, year, False, str(e))
    
    start_time = time.time()
    results_by_year = {}
    total_succeeded = 0
    total_failed = 0
    
    # Process each year
    for year in years:
        year_dir = os.path.join(base_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"YEAR {year} -> writing to {year_dir}")
            print(f"{'='*70}")
        
        tasks = [(lid, year, year_dir) for lid in league_ids]
        completed = 0
        succeeded = 0
        failed = 0
        
        # Had runtime issues with taking forever to grab data, so I parallelized the process.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks to the executor
            future_to_task = {executor.submit(process_league_year, task): task for task in tasks}
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                lid, yr, success, error = future.result()
                completed += 1
                if success:
                    succeeded += 1
                    if verbose:
                        print(f"  [{completed}/{len(tasks)}] {lid}/{yr}")
                else:
                    failed += 1
                    if verbose:
                        print(f"  [{completed}/{len(tasks)}] {lid}/{yr}: {error}")
        
        results_by_year[year] = {'succeeded': succeeded, 'failed': failed}
        total_succeeded += succeeded
        total_failed += failed
        
        elapsed = time.time() - start_time
        if verbose:
            print(f"\nYear {year} complete: {succeeded} succeeded, {failed} failed")
            print(f"Total elapsed time: {elapsed/60:.1f} minutes")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ALL YEARS COMPLETE!")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        print(f"Total succeeded: {total_succeeded}, Total failed: {total_failed}")
        print(f"{'='*70}")
    
    return {
        'total_processed': total_succeeded + total_failed,
        'total_succeeded': total_succeeded,
        'total_failed': total_failed,
        'total_time_minutes': total_time / 60,
        'results_by_year': results_by_year
    }


def collect_injury_data(years, output_dir=None):
    """
    Collect NFL injury data for specified years using nfl_data_py.
    
    Args:
        years: List of years (integers) or single year, or range like range(2021, 2025)
        output_dir: Directory to save injury CSVs (default: data/raw/injuries)
    
    Returns:
        dict with summary: {'total_years': int, 'files_saved': list, 'total_records': int}
    
    Example:
        from src.data_fetchers.data_collector import collect_injury_data
        results = collect_injury_data(years=range(2021, 2025))
    """
    import nfl_data_py as nfl
    import time
    
    # Normalize years input
    if isinstance(years, int):
        years = [years]
    elif isinstance(years, range):
        years = list(years)
    elif not isinstance(years, (list, tuple)):
        raise TypeError(f"years must be an int, list, or range. Got: {type(years)}")
    
    # Set up output directory
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.normpath(os.path.join(script_dir, '../../data/raw/injuries'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure nfl_data_py is installed
    try:
        importlib.import_module('nfl_data_py')
    except ImportError:
        print("! nfl_data_py not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nfl_data_py"])
        importlib.invalidate_caches()
    
    total_records = 0
    files_saved = []
    
    print(f"\n{'='*70}")
    print(f"Collecting injury data for years: {years}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    for year in years:
        try:
            print(f"\nFetching injury data for {year}...")
            start_time = time.time()
            
            # Fetch injury data
            injuries_df = nfl.import_injuries([year])
            
            if injuries_df.empty:
                print(f"  Warning: No injury data found for {year}")
                continue
            
            # Save to CSV
            output_path = os.path.join(output_dir, f'nfl_injuries_{year}.csv')
            injuries_df.to_csv(output_path, index=False)
            
            elapsed = time.time() - start_time
            records = len(injuries_df)
            total_records += records
            
            print(f"  Saved {records:,} records to {output_path} ({elapsed:.1f}s)")
            files_saved.append(output_path)
            
        except Exception as e:
            print(f"  Failed to fetch injury data for {year}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"INJURY DATA COLLECTION COMPLETE!")
    print(f"Total years processed: {len(files_saved)}")
    print(f"Total records: {total_records:,}")
    print(f"{'='*70}")
    
    return {
        'total_years': len(files_saved),
        'files_saved': files_saved,
        'total_records': total_records
    }

# Main execution code has been moved to Fantasy_Football_Analysis.ipynb
# For standalone usage, see function docstrings above.


