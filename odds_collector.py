import requests
import pandas as pd
from datetime import datetime
import os
import logging
import time
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

class OddsCollector:
    def __init__(self):
        self.api_key = '7802fbc4a693bd70d62106c0863fcf47'
        self.base_url = 'https://api.the-odds-api.com/v4/sports'
        self.output_dir = 'data/raw'
        self.sport = 'basketball_nba'
        self.regions = 'us'
        
        # Set up logging
        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/odds_collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        
        # Add date handling
        self.current_date = datetime.now(ZoneInfo('America/New_York')).date()
        self.historical_dir = 'data/historical/odds'
        os.makedirs(self.historical_dir, exist_ok=True)

    def get_sample_games(self) -> List[Dict[Any, Any]]:
        """Get a sample of recent NBA games when no live games are available"""
        url = f'{self.base_url}/{self.sport}/scores/'
        params = {
            'apiKey': self.api_key,
            'daysFrom': 3,  # Get games from last 3 days
            'dateFormat': 'iso'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            games = response.json()
            
            if not games:
                # If no recent games, load from historical data
                return self.load_historical_games()
            
            logging.info(f"Found {len(games)} recent games")
            return games
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching games: {str(e)}")
            # Fallback to historical data
            return self.load_historical_games()

    def load_historical_games(self) -> List[Dict[Any, Any]]:
        """Load historical game data when no live/recent games are available"""
        historical_files = sorted(
            [f for f in os.listdir(self.historical_dir) if f.startswith('props_')],
            reverse=True
        )
        
        if not historical_files:
            logging.error("No historical data available")
            return []
            
        # Load most recent historical file
        latest_file = historical_files[0]
        historical_path = os.path.join(self.historical_dir, latest_file)
        
        try:
            historical_df = pd.read_csv(historical_path)
            
            # Convert DataFrame back to game format
            games = []
            for game_id in historical_df['game_id'].unique():
                game_data = historical_df[historical_df['game_id'] == game_id].iloc[0]
                games.append({
                    'id': game_id,
                    'home_team': game_data['home_team'],
                    'away_team': game_data['away_team'],
                    'commence_time': game_data['commence_time']
                })
            
            logging.info(f"Loaded {len(games)} games from historical data")
            return games
            
        except Exception as e:
            logging.error(f"Error loading historical data: {str(e)}")
            return []

    def get_props(self, game_id: str) -> Dict[Any, Any]:
        """Fetch props for a specific game"""
        url = f'{self.base_url}/{self.sport}/events/{game_id}/odds'
        params = {
            'apiKey': self.api_key,
            'regions': self.regions,
            'markets': ('player_points,player_rebounds,player_assists,player_threes,'
                       'player_blocks,player_steals,player_points_rebounds,'
                       'player_points_assists,player_points_rebounds_assists'),
            'bookmakers': 'fanduel'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            logging.info(f"Successfully fetched props for game {game_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching props for game {game_id}: {str(e)}")
            return None

    def process_props(self, props_data: Dict[Any, Any]) -> pd.DataFrame:
        """Process raw props data into a DataFrame"""
        processed_props = []
        
        if not props_data or 'bookmakers' not in props_data:
            return pd.DataFrame()
            
        game_id = props_data.get('id')
        home_team = props_data.get('home_team')
        away_team = props_data.get('away_team')
        commence_time = props_data.get('commence_time')
        
        for bookmaker in props_data.get('bookmakers', []):
            if bookmaker['key'] != 'fanduel':
                continue
                
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                
                for outcome in market.get('outcomes', []):
                    prop_data = {
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'commence_time': commence_time,
                        'market': market_key,
                        'player': outcome.get('description', ''),
                        'line': outcome.get('point', None),
                        'over_price': outcome.get('price') if outcome.get('name') == 'Over' else None,
                        'under_price': outcome.get('price') if outcome.get('name') == 'Under' else None,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    processed_props.append(prop_data)
        
        return pd.DataFrame(processed_props)

    def collect_all_props(self):
        """Collect all props with fallback to historical data"""
        logging.info("Getting list of games...")
        games = self.get_sample_games()  # Use sample games instead of live games
        
        if not games:
            logging.warning("No games found - live or historical")
            return
            
        all_props = []
        collection_time = datetime.now()
        
        for game in games:
            game_id = game.get('id')
            logging.info(f"Processing props for game {game_id}...")
            
            # Try to get live props first, fall back to historical
            props_data = self.get_props(game_id)
            if not props_data:
                props_data = self.load_historical_props(game_id)
            
            if props_data:
                props_df = self.process_props(props_data)
                if not props_df.empty:
                    all_props.append(props_df)
            
            time.sleep(1)  # Rate limiting
        
        if all_props:
            final_df = pd.concat(all_props, ignore_index=True)
            
            # Save to daily_props.csv
            output_path = os.path.join(self.output_dir, 'daily_props.csv')
            final_df.to_csv(output_path, index=False)
            
            logging.info(f"Successfully processed {len(final_df)} props")
            logging.info(f"Data saved to {output_path}")
        else:
            logging.warning("No props data processed")

    def load_historical_props(self, game_id: str) -> Dict[Any, Any]:
        """Load historical props data for a specific game"""
        historical_file = os.path.join(self.historical_dir, f'{game_id}_odds_history.csv')
        
        if os.path.exists(historical_file):
            try:
                props_df = pd.read_csv(historical_file)
                
                # Convert DataFrame back to API response format
                props_data = {
                    'id': game_id,
                    'home_team': props_df['home_team'].iloc[0],
                    'away_team': props_df['away_team'].iloc[0],
                    'commence_time': props_df['commence_time'].iloc[0],
                    'bookmakers': [{
                        'key': 'fanduel',
                        'markets': self._convert_df_to_markets(props_df)
                    }]
                }
                
                return props_data
                
            except Exception as e:
                logging.error(f"Error loading historical props for game {game_id}: {str(e)}")
                return None
        return None

    def _convert_df_to_markets(self, props_df: pd.DataFrame) -> List[Dict[Any, Any]]:
        """Convert DataFrame format back to API market format"""
        markets = []
        
        for market_type in props_df['market'].unique():
            market_data = props_df[props_df['market'] == market_type]
            
            market = {
                'key': market_type,
                'outcomes': []
            }
            
            for _, row in market_data.iterrows():
                if pd.notna(row['over_price']):
                    market['outcomes'].append({
                        'name': 'Over',
                        'description': row['player'],
                        'point': row['line'],
                        'price': row['over_price']
                    })
                if pd.notna(row['under_price']):
                    market['outcomes'].append({
                        'name': 'Under',
                        'description': row['player'],
                        'point': row['line'],
                        'price': row['under_price']
                    })
            
            markets.append(market)
        
        return markets

def main():
    collector = OddsCollector()
    collector.collect_all_props()

if __name__ == "__main__":
    main()
