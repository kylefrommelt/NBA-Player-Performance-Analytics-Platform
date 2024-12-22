from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats, leaguedashteamstats, teamgamelogs, scoreboardv2, leaguegamefinder, leaguedashptdefend, leaguedashlineups, leaguedashplayershotlocations, leaguedashplayerbiostats, leaguedashteamclutch, boxscoretraditionalv2, boxscoresummaryv2, playercareerstats
from nba_api.live.nba.endpoints import scoreboard
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
from geopy.distance import geodesic
from zoneinfo import ZoneInfo
import json
import logging
import os
import traceback

class RateLimiter:
    def __init__(self, calls_per_second=1):
        self.calls_per_second = calls_per_second
        self.last_call = datetime.now()

    def wait(self):
        now = datetime.now()
        time_since_last = now - self.last_call
        if time_since_last < timedelta(seconds=1/self.calls_per_second):
            sleep_time = (timedelta(seconds=1/self.calls_per_second) - time_since_last).total_seconds()
            time.sleep(sleep_time)
        self.last_call = datetime.now()

class NBADataCollector:
    def __init__(self, seasons=None):
        # Default to last 3 seasons if none specified
        if seasons is None:
            self.seasons = ['2023-24', '2022-23', '2021-22']
        else:
            self.seasons = seasons if isinstance(seasons, list) else [seasons]
        
        # Set current season
        self.season = self.seasons[0]
        
        self.output_dir = 'data/raw/'
        self.historical_dir = 'data/historical/'
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.historical_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load arena info
        with open('data/reference/arena_info.json', 'r') as f:
            self.arena_info = json.load(f)
            
        # Set up logging
        self._setup_logging()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(calls_per_second=1)

    def _setup_logging(self):
        log_file = f'logs/data_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing NBA Data Collector for seasons {self.seasons}")
        
    def _make_api_call(self, endpoint, max_attempts=3, delay=2, **params):
        """Helper method to make API calls with retries"""
        for attempt in range(max_attempts):
            try:
                self.logger.info(f"Making API call to {endpoint.__name__} with params: {params}")
                self.rate_limiter.wait()  # Rate limiting
                response = endpoint(**params)
                
                if hasattr(response, 'get_data_frames'):
                    df = response.get_data_frames()[0]
                    if df is not None and not df.empty:
                        self.logger.info(f"Successfully retrieved DataFrame with shape: {df.shape}")
                        return df
                else:
                    self.logger.warning(f"Response doesn't have get_data_frames method")
                    return response
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(delay)
        
        self.logger.error(f"Failed to fetch data after {max_attempts} attempts")
        return None
        
    def _validate_dataframe(self, df, required_columns):
        """Helper method to validate DataFrame contents"""
        if df is None or df.empty:
            return False
        return all(col in df.columns for col in required_columns)
        
    def get_team_stats(self):
        """Collect team statistics"""
        self.logger.info("Starting team stats collection")
        try:
            # Basic team stats
            basic_team_stats = self._make_api_call(
                leaguedashteamstats.LeagueDashTeamStats,
                season=self.season,
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame'
            )
            self.logger.info(f"Collected basic team stats: {len(basic_team_stats)} teams")
            
            # Advanced team stats
            advanced_team_stats = self._make_api_call(
                leaguedashteamstats.LeagueDashTeamStats,
                season=self.season,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            self.logger.info(f"Collected advanced team stats: {len(advanced_team_stats)} teams")
            
            # Merge basic and advanced stats
            team_stats = pd.merge(basic_team_stats, advanced_team_stats, on='TEAM_ID')
            team_stats.to_csv(f'{self.output_dir}team_stats.csv', index=False)
            
            time.sleep(2)  # Rate limiting
            self.logger.info("Team stats collection completed successfully")
            return team_stats
        except Exception as e:
            self.logger.error(f"Error collecting team stats: {str(e)}")
            raise

    def get_player_stats(self, season=None):
        """Collect player statistics for a specific season"""
        seasons_to_process = [season] if season else self.seasons
        
        for season in seasons_to_process:
            self.logger.info(f"Starting player stats collection for season {season}")
            try:
                # Basic player stats
                basic_player_stats = self._make_api_call(
                    leaguedashplayerstats.LeagueDashPlayerStats,
                    season=season,
                    measure_type_detailed_defense='Base',
                    per_mode_detailed='PerGame'
                )
                
                # Advanced player stats
                advanced_player_stats = self._make_api_call(
                    leaguedashplayerstats.LeagueDashPlayerStats,
                    season=season,
                    measure_type_detailed_defense='Advanced',
                    per_mode_detailed='PerGame'
                )
                
                # Bio stats
                bio_stats = self._make_api_call(
                    leaguedashplayerbiostats.LeagueDashPlayerBioStats,
                    season=season
                )
                self.logger.info(f"Collected player bio stats: {len(bio_stats)} players")
                
                # Merge all stats
                player_stats = pd.merge(basic_player_stats, advanced_player_stats, on='PLAYER_ID')
                player_stats = pd.merge(player_stats, bio_stats, on='PLAYER_ID')
                player_stats.to_csv(f'{self.output_dir}player_stats.csv', index=False)
                
                # Merge and save with season info
                player_stats['SEASON'] = season
                
                # Save to historical directory
                output_path = f'{self.historical_dir}player_stats_{season}.csv'
                player_stats.to_csv(output_path, index=False)
                self.logger.info(f"Saved {season} player stats to {output_path}")
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error collecting {season} player stats: {str(e)}")
                continue
                
    def collect_historical_props(self):
        """Collect historical player props data"""
        self.logger.info("Starting historical props collection")
        
        try:
            all_props = []
            
            for season in self.seasons:
                self.logger.info(f"Collecting props for season {season}")
                
                # Get game IDs for the season
                games_df = self._make_api_call(
                    leaguegamefinder.LeagueGameFinder,
                    season_nullable=season
                ).get_data_frames()[0]
                
                # Process in smaller batches
                game_ids = games_df['GAME_ID'].unique()
                batch_size = 50  # Process 50 games at a time
                
                for i in range(0, len(game_ids), batch_size):
                    batch = game_ids[i:i + batch_size]
                    self.logger.info(f"Processing batch {i//batch_size + 1} of {len(game_ids)//batch_size + 1}")
                    
                    for game_id in batch:
                        try:
                            # Get box score for actual performance
                            box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(
                                game_id=game_id
                            ).get_data_frames()[0]
                            
                            # Process and store player performance
                            game_props = self._process_game_props(box_score, game_id, season)
                            all_props.extend(game_props)
                            
                            time.sleep(1)  # Rate limiting between games
                            
                        except Exception as e:
                            self.logger.warning(f"Error processing game {game_id}: {str(e)}")
                            continue
                    
                    # Save intermediate results after each batch
                    if all_props:
                        props_df = pd.DataFrame(all_props)
                        props_df.to_csv(f'{self.historical_dir}player_props_{season}_partial.csv', index=False)
                        self.logger.info(f"Saved partial results for {season}")
                    
                    time.sleep(5)  # Rate limiting between batches
                
                # Save complete season props
                if all_props:
                    props_df = pd.DataFrame(all_props)
                    props_df.to_csv(f'{self.historical_dir}player_props_{season}.csv', index=False)
                    self.logger.info(f"Completed {season} collection")
                
            self.logger.info("Historical props collection completed")
            return pd.DataFrame(all_props) if all_props else None
            
        except Exception as e:
            self.logger.error(f"Error in historical props collection: {str(e)}")
            return None

    def _process_game_props(self, box_score, game_id, season):
        """Process box score into props format"""
        props = []
        
        for _, row in box_score.iterrows():
            props.extend([
                {
                    'game_id': game_id,
                    'season': season,
                    'player': row['PLAYER_NAME'],
                    'market': 'player_points',
                    'actual_value': row['PTS']
                },
                {
                    'game_id': game_id,
                    'season': season,
                    'player': row['PLAYER_NAME'],
                    'market': 'player_assists',
                    'actual_value': row['AST']
                },
                {
                    'game_id': game_id,
                    'season': season,
                    'player': row['PLAYER_NAME'],
                    'market': 'player_rebounds',
                    'actual_value': row['REB']
                },
                {
                    'game_id': game_id,
                    'season': season,
                    'player': row['PLAYER_NAME'],
                    'market': 'player_threes',
                    'actual_value': row['FG3M']
                }
            ])
            
        return props

    def get_last_n_games_stats(self, n_games=6):
        """Collect last N games stats"""
        self.logger.info(f"Starting collection of last {n_games} games stats")
        try:
            # Last N games for teams
            team_stats = self._make_api_call(
                leaguedashteamstats.LeagueDashTeamStats,
                season=self.season,
                last_n_games=n_games,
                per_mode_detailed='PerGame'
            )
            
            if team_stats is not None:
                team_stats.to_csv(f'{self.output_dir}team_last_{n_games}_games.csv', index=False)
                self.logger.info(f"Saved team stats for last {n_games} games")
            
            # Last N games for players
            player_stats = self._make_api_call(
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=self.season,
                last_n_games=n_games,
                per_mode_detailed='PerGame'
            )
            
            if player_stats is not None:
                player_stats.to_csv(f'{self.output_dir}player_last_{n_games}_games.csv', index=False)
                self.logger.info(f"Saved player stats for last {n_games} games")
            
            time.sleep(2)
            return team_stats, player_stats
            
        except Exception as e:
            self.logger.error(f"Error collecting last {n_games} games stats: {str(e)}")
            return None, None

    def get_matchup_data(self):
        """
        Collecting matchup data including defensive stats and schedules
        """
        self.logger.info("Starting matchup data collection")
        try:
            # Get team defensive stats
            self.logger.info("Collecting team defensive stats")
            defensive_stats = self._make_api_call(
                leaguedashteamstats.LeagueDashTeamStats,
                season=self.season,
                measure_type_detailed_defense='Defense',
                per_mode_detailed='PerGame'
            )
            defensive_stats.to_csv(f'{self.output_dir}team_defensive_stats.csv', index=False)
            self.logger.info("Saved team defensive stats")
            
            # Get schedule data
            self.logger.info("Collecting schedule data")
            schedule_data, today_games = self.get_schedule_data()
            
            # Get defensive matchup stats by position
            self.logger.info("Collecting defense vs position stats")
            defense_vs_position = self.get_defense_vs_position()
            
            time.sleep(2)
            self.logger.info("Matchup data collection completed successfully")
            return {
                'defensive_stats': defensive_stats,
                'schedule': schedule_data,
                'today_games': today_games,
                'defense_vs_position': defense_vs_position
            }
        except Exception as e:
            self.logger.error(f"Error collecting matchup data: {str(e)}")
            raise

    def get_schedule_data(self):
        self.logger.info("Starting schedule data collection")
        try:
            # Get upcoming games and recent results
            self.logger.info("Collecting season schedule data")
            game_finder = self._make_api_call(
                leaguegamefinder.LeagueGameFinder,
                season_nullable=self.season,
                league_id_nullable='00'  # NBA
            )
            
            # Get today's games using scoreboardv2 instead of live scoreboard
            self.logger.info("Collecting today's games")
            today_games = scoreboardv2.ScoreboardV2().get_data_frames()[0]
            
            # Save both datasets
            game_finder.to_csv(f'{self.output_dir}season_schedule.csv', index=False)
            today_games.to_csv(f'{self.output_dir}today_games.csv', index=False)
            
            self.logger.info(f"Saved schedule data: {len(game_finder)} games in season, {len(today_games)} games today")
            time.sleep(2)
            return game_finder, today_games
        except Exception as e:
            self.logger.error(f"Error collecting schedule data: {str(e)}")
            raise

    def get_defense_vs_position(self):
        self.logger.info("Starting defense vs position stats collection")
        try:
            # First, let's try to get the endpoint documentation
            endpoint = leaguedashptdefend.LeagueDashPtDefend
            self.logger.info(f"Endpoint details: {endpoint.__doc__}")
            
            # Try with minimal parameters first
            self.logger.info("Attempting to fetch defense stats with minimal parameters")
            defense_stats = self._make_api_call(
                endpoint,
                season=self.season
            )
            
            if defense_stats is not None:
                self.logger.info(f"Successfully retrieved data with shape: {defense_stats.shape}")
                self.logger.info(f"Columns available: {defense_stats.columns.tolist()}")
                defense_stats.to_csv(f'{self.output_dir}defense_vs_position.csv', index=False)
                return defense_stats
            else:
                # Try alternative parameters
                self.logger.info("First attempt failed, trying with additional parameters")
                defense_stats = self._make_api_call(
                    endpoint,
                    season=self.season,
                    league_id='00',
                    per_mode_simple='PerGame'
                )
                
                if defense_stats is not None:
                    self.logger.info(f"Second attempt successful with shape: {defense_stats.shape}")
                    defense_stats.to_csv(f'{self.output_dir}defense_vs_position.csv', index=False)
                    return defense_stats
                
                self.logger.error("All attempts to collect defense stats failed")
                return None
            
        except Exception as e:
            self.logger.error(f"Error collecting defense vs position stats: {str(e)}")
            raise

    def get_home_away_splits(self):
        self.logger.info("Starting home/away splits collection")
        try:
            # Get team performance splits for home and away games
            self.logger.info("Collecting team game logs")
            team_game_logs = teamgamelogs.TeamGameLogs(
                season_nullable=self.season
            ).get_data_frames()[0]
            
            # Debug column names
            self.logger.info(f"Available columns: {team_game_logs.columns.tolist()}")
            
            # Process home/away splits using MATCHUP column
            self.logger.info("Processing home/away splits")
            home_games = team_game_logs[team_game_logs['MATCHUP'].str.contains(' vs. ', na=False)]
            away_games = team_game_logs[team_game_logs['MATCHUP'].str.contains(' @ ', na=False)]
            
            self.logger.info(f"Found {len(home_games)} home games and {len(away_games)} away games")
            
            home_games.to_csv(f'{self.output_dir}home_games.csv', index=False)
            away_games.to_csv(f'{self.output_dir}away_games.csv', index=False)
            
            time.sleep(2)
            self.logger.info("Home/away splits collection completed successfully")
            return home_games, away_games
        except Exception as e:
            self.logger.error(f"Error collecting home/away splits: {str(e)}")
            raise

    def get_rest_days_impact(self):
        """Analyze impact of rest days on team performance"""
        self.logger.info("Starting rest days impact analysis")
        try:
            # Get team game logs with correct parameter
            self.logger.info("Collecting team game logs")
            team_game_logs = teamgamelogs.TeamGameLogs(
                season_nullable=self.season  # Changed from season to season_nullable
            ).get_data_frames()[0]
            
            rest_impact = []
            for team_id in team_game_logs['TEAM_ID'].unique():
                team_games = team_game_logs[team_game_logs['TEAM_ID'] == team_id]
                team_games = team_games.sort_values('GAME_DATE')
                
                # Calculate days between games
                team_games['GAME_DATE'] = pd.to_datetime(team_games['GAME_DATE'])
                team_games['DAYS_REST'] = team_games['GAME_DATE'].diff().dt.days
                
                # Group by rest days and calculate stats
                rest_stats = team_games.groupby('DAYS_REST').agg({
                    'PTS': 'mean',
                    'FG_PCT': 'mean',
                    'WL': lambda x: (x == 'W').mean(),  # Win percentage
                    'PLUS_MINUS': 'mean'
                }).reset_index()
                
                for _, row in rest_stats.iterrows():
                    rest_impact.append({
                        'team_id': team_id,
                        'days_rest': row['DAYS_REST'],
                        'avg_points': row['PTS'],
                        'avg_fg_pct': row['FG_PCT'],
                        'win_pct': row['WL'],
                        'plus_minus': row['PLUS_MINUS']
                    })
            
            rest_impact_df = pd.DataFrame(rest_impact)
            rest_impact_df.to_csv(f'{self.output_dir}rest_days_impact.csv', index=False)
            self.logger.info("Rest days impact analysis completed successfully")
            return rest_impact_df
            
        except Exception as e:
            self.logger.error(f"Error analyzing rest days impact: {str(e)}")
            return None

    def get_injury_data(self):
        """Enhanced injury data collection"""
        self.logger.info("Starting enhanced injury data collection")
        try:
            # Get current injuries
            current_injuries = self._scrape_injury_data()
            
            # Get historical injury data from game logs
            historical_injuries = {}
            for season in self.seasons:
                game_logs = self.get_player_game_logs(season)
                if game_logs is not None:
                    game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
                    
                    # Detect injury periods (gaps in games)
                    for player_id in game_logs['Player_ID'].unique():
                        player_games = game_logs[game_logs['Player_ID'] == player_id].sort_values('GAME_DATE')
                        date_gaps = player_games['GAME_DATE'].diff()
                        
                        # Find gaps longer than 7 days
                        injury_gaps = date_gaps[date_gaps > pd.Timedelta(days=7)]
                        
                        if not injury_gaps.empty:
                            historical_injuries[player_id] = {
                                'injury_periods': [
                                    {
                                        'start_date': player_games.loc[idx - 1, 'GAME_DATE'] if idx > 0 else None,
                                        'return_date': player_games.loc[idx, 'GAME_DATE'],
                                        'games_missed': len(date_gaps[date_gaps > pd.Timedelta(days=7)]),
                                        'pre_injury_stats': player_games.iloc[max(0, idx-5):idx].mean().to_dict(),
                                        'post_injury_stats': player_games.iloc[idx:min(len(player_games), idx+5)].mean().to_dict()
                                    }
                                    for idx in injury_gaps.index
                                ]
                            }
            
            # Combine current and historical data
            injury_data = {
                'current': current_injuries,
                'historical': historical_injuries,
                'metadata': {
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'seasons_analyzed': self.seasons
                }
            }
            
            # Save to file
            with open(f'{self.output_dir}injury_data.json', 'w') as f:
                json.dump(injury_data, f)
            
            return injury_data

        except Exception as e:
            self.logger.error(f"Error collecting injury data: {str(e)}")
            return None

    def _scrape_injury_data(self):
        """Scrapes current NBA injury data from NBA.com."""
        try:
            # NBA's official injury report URL
            url = "https://www.nba.com/players/injuries"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch NBA.com injury data. Status code: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            current_injuries = []
            
            # Find the injury table
            injury_table = soup.find('table', class_='InjuriesTable_table__qoaa9')
            
            if injury_table:
                for row in injury_table.find_all('tr')[1:]:  # Skip header row
                    cols = row.find_all('td')
                    if len(cols) >= 4:
                        injury_info = {
                            'player_name': cols[0].text.strip(),
                            'team': cols[1].text.strip(),
                            'status': cols[2].text.strip(),
                            'injury_details': cols[3].text.strip(),
                            'source': 'NBA.com',
                            'last_updated': datetime.now().strftime('%Y-%m-%d')
                        }
                        current_injuries.append(injury_info)
            
            if not current_injuries:
                # Backup method: try finding injury cards
                injury_cards = soup.find_all('div', class_='InjuryCard_card__Ykn9F')
                for card in injury_cards:
                    player_name = card.find('div', class_='InjuryCard_name__gFjVh')
                    team = card.find('div', class_='InjuryCard_team__Kd5Rk')
                    status = card.find('div', class_='InjuryCard_status__7QVZC')
                    details = card.find('div', class_='InjuryCard_injury__Dzm8O')
                    
                    if player_name and team:
                        injury_info = {
                            'player_name': player_name.text.strip(),
                            'team': team.text.strip(),
                            'status': status.text.strip() if status else 'Out',
                            'injury_details': details.text.strip() if details else 'Not Specified',
                            'source': 'NBA.com',
                            'last_updated': datetime.now().strftime('%Y-%m-%d')
                        }
                        current_injuries.append(injury_info)
            
            self.logger.info(f"Found {len(current_injuries)} current injuries")
            return current_injuries if current_injuries else None
            
        except Exception as e:
            self.logger.error(f"Error scraping injury data: {str(e)}")
            self.logger.error(f"Full error: {traceback.format_exc()}")
            return None

    def get_clutch_stats(self):
        """Get team performance in clutch situations"""
        self.logger.info("Starting clutch stats collection")
        try:
            # Updated parameters for clutch stats
            clutch_stats = self._make_api_call(
                leaguedashteamclutch.LeagueDashTeamClutch,  # Changed endpoint
                season=self.season,
                per_mode_detailed='PerGame',
                clutch_time='Last 5 Minutes',
                ahead_behind='Ahead or Behind',
                point_diff=5
            )
            
            if clutch_stats is not None:
                clutch_stats.to_csv(f'{self.output_dir}team_clutch_stats.csv', index=False)
                self.logger.info("Clutch stats collection completed successfully")
                return clutch_stats
            return None
        except Exception as e:
            self.logger.error(f"Error collecting clutch stats: {str(e)}")
            return None

    def get_lineup_stats(self):
        self.logger.info("Starting lineup stats collection")
        try:
            self.logger.info("Collecting performance data for different lineup combinations")
            lineup_stats = leaguedashlineups.LeagueDashLineups(
                season=self.season,
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame'
            ).get_data_frames()[0]
            
            lineup_stats.to_csv(f'{self.output_dir}lineup_stats.csv', index=False)
            self.logger.info(f"Saved lineup stats for {len(lineup_stats)} combinations")
            time.sleep(2)
            return lineup_stats
        except Exception as e:
            self.logger.error(f"Error collecting lineup stats: {str(e)}")
            raise

    def get_shooting_stats(self):
        self.logger.info("Starting shooting stats collection")
        try:
            self.logger.info("Collecting detailed shooting stats (shot zones, distances)")
            shooting_stats = leaguedashplayershotlocations.LeagueDashPlayerShotLocations(
                season=self.season,
                per_mode_detailed='PerGame'
            ).get_data_frames()[0]
            
            shooting_stats.to_csv(f'{self.output_dir}shooting_stats.csv', index=False)
            self.logger.info(f"Saved shooting stats for {len(shooting_stats)} players")
            time.sleep(2)
            return shooting_stats
        except Exception as e:
            self.logger.error(f"Error collecting shooting stats: {str(e)}")
            raise

    def get_back_to_back_stats(self):
        """Analyze team performance in back-to-back games"""
        self.logger.info("Starting back-to-back analysis")
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=self.season,
                league_id_nullable='00'
            )
            games_df = game_finder.get_data_frames()[0]
            
            # Sort games by team and date
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values(['TEAM_ID', 'GAME_DATE'])
            
            # Calculate days between games
            games_df['DAYS_REST'] = games_df.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days
            
            # Identify back-to-back games (1 or 0 days rest)
            back_to_back_stats = games_df[games_df['DAYS_REST'].isin([0, 1])]
            
            return back_to_back_stats
        except Exception as e:
            self.logger.error(f"Error analyzing back-to-back stats: {str(e)}")
            return None

    def get_travel_impact_stats(self):
        """Analyze impact of travel on team performance"""
        self.logger.info("Starting travel impact analysis")
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=self.season,
                league_id_nullable='00'
            )
            games_df = game_finder.get_data_frames()[0]
            
            # Sort games by team and date
            games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
            games_df = games_df.sort_values(['TEAM_ID', 'GAME_DATE'])
            
            # Identify home/away games
            travel_impact = games_df.copy()
            travel_impact['IS_HOME'] = travel_impact['MATCHUP'].str.contains('vs')
            
            return travel_impact
        except Exception as e:
            self.logger.error(f"Error analyzing travel impact: {str(e)}")
            return None

    def get_arena_factors(self):
        """Analyze impact of different arenas on team performance"""
        self.logger.info("Starting arena factors analysis")
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=self.season,
                league_id_nullable='00'
            )
            games_df = game_finder.get_data_frames()[0]
            
            # Calculate arena-specific stats
            arena_factors = games_df.groupby(['TEAM_ID', 'MATCHUP']).agg({
                'PTS': 'mean',
                'FG_PCT': 'mean',
                'FG3_PCT': 'mean',
                'FT_PCT': 'mean',
                'PLUS_MINUS': 'mean'
            }).reset_index()
            
            # Add altitude data from arena_info
            arena_factors['ALTITUDE'] = arena_factors['TEAM_ID'].map(
                {team_id: info.get('altitude', 0) for team_id, info in self.arena_info.items()}
            )
            
            # Save to CSV
            arena_factors.to_csv(f'{self.output_dir}arena_factors.csv', index=False)
            self.logger.info("Arena factors analysis completed and saved")
            
            return arena_factors
        except Exception as e:
            self.logger.error(f"Error analyzing arena factors: {str(e)}")
            return None

    def get_head_to_head_stats(self):
        """Collect historical head-to-head matchup data"""
        self.logger.info("Starting head-to-head analysis")
        try:
            game_finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=self.season,
                league_id_nullable='00'
            )
            games_df = game_finder.get_data_frames()[0]
            
            # Extract opponent from MATCHUP
            games_df['OPPONENT_ID'] = games_df.apply(
                lambda x: x['MATCHUP'].split()[-1], axis=1
            )
            
            # Calculate head-to-head stats
            h2h_stats = games_df.groupby(['TEAM_ID', 'OPPONENT_ID']).agg({
                'WL': lambda x: (x == 'W').mean(),  # Win percentage
                'PTS': 'mean',
                'PLUS_MINUS': 'mean',
                'FG_PCT': 'mean',
                'FG3_PCT': 'mean'
            }).reset_index()
            
            h2h_stats.to_csv(f'{self.output_dir}head_to_head_stats.csv', index=False)
            self.logger.info("Head-to-head analysis completed and saved")
            
            return h2h_stats
        except Exception as e:
            self.logger.error(f"Error analyzing head-to-head stats: {str(e)}")
            return None

    def get_referee_stats(self):
        """Collect referee tendencies and impact data"""
        self.logger.info("Starting referee analysis")
        try:
            # Get today's games
            scoreboard = self._make_api_call(
                scoreboardv2.ScoreboardV2,
                game_date=datetime.now().strftime("%Y-%m-%d")
            )
            games = scoreboard.game_header.get_data_frame()
            
            self.logger.info(f"Found {len(games)} games for today")
            
            ref_data = []
            for _, game in games.iterrows():
                game_id = game['GAME_ID']
                self.logger.info(f"Getting officials for game {game_id}")
                
                # Get game summary including officials
                game_summary = self._make_api_call(
                    boxscoresummaryv2.BoxScoreSummaryV2,
                    game_id=game_id
                )
                
                # Correctly access officials data frame
                if hasattr(game_summary, 'officials'):
                    officials_df = game_summary.officials.get_data_frame()
                    for _, official in officials_df.iterrows():
                        ref_data.append({
                            'REFEREE_ID': official['OFFICIAL_ID'],
                            'REFEREE_NAME': f"{official['FIRST_NAME']} {official['LAST_NAME']}",
                            'GAME_ID': game_id
                        })
            
            if ref_data:
                ref_stats = pd.DataFrame(ref_data)
                ref_stats.to_csv(f'{self.output_dir}referee_stats.csv', index=False)
                self.logger.info(f"Saved referee stats for {len(ref_stats)} officials")
                return ref_stats
            else:
                self.logger.warning("No referee data collected")
                pd.DataFrame(columns=['REFEREE_ID', 'REFEREE_NAME', 'GAME_ID']
                ).to_csv(f'{self.output_dir}referee_stats.csv', index=False)
                return None

        except Exception as e:
            self.logger.error(f"Error analyzing referee stats: {str(e)}")
            self.logger.error(f"Error details: {traceback.format_exc()}")
            return None

    def _get_todays_game_ids(self):
        """Helper method to get today's game IDs"""
        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=datetime.now().strftime("%Y-%m-%d")
        )
        games_df = scoreboard.get_data_frames()[0]
        return games_df['GAME_ID'].tolist()

    def get_usage_patterns(self):
        """Analyze player usage patterns"""
        self.logger.info("Starting usage patterns analysis")
        try:
            # Get player stats directly
            player_stats = self.get_player_stats()
            if player_stats is None:
                self.logger.error("Player stats data is missing")
                return None
            
            # Log the columns of the DataFrame
            self.logger.info(f"Player stats DataFrame columns: {player_stats.columns.tolist()}")
            
            # Check for minutes column variations
            min_column = None
            for col in ['MIN', 'MIN_x', 'MIN_y']:
                if col in player_stats.columns:
                    min_column = col
                    self.logger.info(f"Found minutes column: {col}")
                    break
                
            if min_column is None:
                self.logger.error("No minutes column found in player stats")
                return None
            
            usage_patterns = {}
            for team_id in player_stats['TEAM_ID'].unique():
                team_players = player_stats[player_stats['TEAM_ID'] == team_id]
                
                # Calculate usage patterns based on minutes played
                usage_patterns[team_id] = {
                    'high_usage_players': team_players[
                        team_players[min_column] >= 20
                    ]['PLAYER_ID'].tolist(),
                    'rotation_players': team_players[
                        (team_players[min_column] >= 10) & 
                        (team_players[min_column] < 20)
                    ]['PLAYER_ID'].tolist(),
                    'bench_players': team_players[
                        team_players[min_column] < 10
                    ]['PLAYER_ID'].tolist()
                }
            
            return usage_patterns
        except Exception as e:
            self.logger.error(f"Error analyzing usage patterns: {str(e)}")
            return None

    def get_pace_impact_stats(self):
        """Analyze impact of pace on team and player performance"""
        self.logger.info("Starting pace impact analysis")
        try:
            # Get team and player stats
            team_stats = self.get_team_stats()
            player_stats = self.get_player_stats()
            
            if team_stats is None or player_stats is None:
                self.logger.error("Stats data is missing")
                return None
            
            # Create pace impact analysis
            pace_impact = {
                'team_pace': team_stats[['TEAM_ID', 'PACE', 'OFF_RATING', 'DEF_RATING']],
                'player_pace': player_stats[['PLAYER_ID', 'PACE', 'OFF_RATING', 'DEF_RATING']]
            }
            
            return pace_impact
        except Exception as e:
            self.logger.error(f"Error analyzing pace impact: {str(e)}")
            return None

    def get_player_game_logs(self, season=None):
        """Collect detailed game logs for each player"""
        self.logger.info("Starting player game logs collection")
        
        all_logs = []
        seasons_to_process = [season] if season else self.seasons
        
        for season in seasons_to_process:
            try:
                # Get list of active players for the season
                players_df = self._make_api_call(
                    leaguedashplayerstats.LeagueDashPlayerStats,
                    season=season
                )
                
                if players_df is not None:
                    game_logs = []
                    for player_id in players_df['PLAYER_ID'].unique():
                        try:
                            logs = self._make_api_call(
                                playergamelog.PlayerGameLog,
                                player_id=player_id,
                                season=season
                            )
                            if logs is not None:
                                logs['SEASON'] = season
                                game_logs.append(logs)
                            time.sleep(1)  # Rate limiting
                        except Exception as e:
                            self.logger.warning(f"Error getting game logs for player {player_id}: {str(e)}")
                            continue
                            
                    if game_logs:
                        season_logs = pd.concat(game_logs)
                        all_logs.append(season_logs)
                        season_logs.to_csv(f'{self.historical_dir}game_logs_{season}.csv', index=False)
                        self.logger.info(f"Saved game logs for season {season}")
                
            except Exception as e:
                self.logger.error(f"Error collecting game logs for season {season}: {str(e)}")
                continue
                
        return pd.concat(all_logs) if all_logs else None

    def get_matchup_history(self):
        """Collect historical matchup data between players/teams"""
        self.logger.info("Starting matchup history collection")
        
        all_matchups = []
        for season in self.seasons:
            try:
                matchups = self._make_api_call(
                    leaguegamefinder.LeagueGameFinder,
                    season_nullable=season
                )
                
                if matchups is not None:
                    matchups['SEASON'] = season
                    all_matchups.append(matchups)
                    matchups.to_csv(f'{self.historical_dir}matchups_{season}.csv', index=False)
                    self.logger.info(f"Saved matchup history for season {season}")
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Error collecting matchup history: {str(e)}")
                continue
                
        return pd.concat(all_matchups) if all_matchups else None

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting NBA data collection process")
    collector = NBADataCollector(seasons=['2023-24', '2022-23', '2024-25'])
    
    # Initialize results dictionary
    results = {}
    
    # Collect historical data first
    collector.collect_historical_props()
    
    # Then collect current season data
    collection_methods = [
        ('team_stats', collector.get_team_stats),
        ('player_stats', collector.get_player_stats),
        ('game_logs', collector.get_player_game_logs),
        ('matchup_history', collector.get_matchup_history),
        ('last_6_games', collector.get_last_n_games_stats),
        ('matchup_data', collector.get_matchup_data),
        ('defense_vs_position', collector.get_defense_vs_position),
        ('clutch_stats', collector.get_clutch_stats),
        ('home_away_splits', collector.get_home_away_splits),
        ('lineup_stats', collector.get_lineup_stats),
        ('shooting_stats', collector.get_shooting_stats),
        ('rest_impact', collector.get_rest_days_impact),
        ('injury_data', collector.get_injury_data),
        ('back_to_back', collector.get_back_to_back_stats),
        ('travel_impact', collector.get_travel_impact_stats),
        ('arena_factors', collector.get_arena_factors),
        ('head_to_head', collector.get_head_to_head_stats),
        #('referee_stats', collector.get_referee_stats),
        ('usage_patterns', collector.get_usage_patterns),
        ('pace_impact', collector.get_pace_impact_stats)
    ]
    
    # Collect data from each method
    for name, method in collection_methods:
        try:
            logging.info(f"Collecting {name}")
            results[name] = method()
            if results[name] is not None:
                logging.info(f"Successfully collected {name}")
            else:
                logging.warning(f"No data collected for {name}")
        except Exception as e:
            logging.error(f"Error collecting {name}: {str(e)}")
            results[name] = None
            continue  # Continue with next collection even if one fails
    
    # Log final summary
    successful = sum(1 for v in results.values() if v is not None)
    total = len(collection_methods)
    logging.info(f"Data collection completed. Successfully collected {successful}/{total} datasets")
    
    # Log which collections failed
    failed = [k for k, v in results.items() if v is None]
    if failed:
        logging.warning(f"Failed collections: {', '.join(failed)}")

if __name__ == "__main__":
    main()
