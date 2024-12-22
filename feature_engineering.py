import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional
import warnings
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import xgboost as xgb
import gc
from consistency_tracker import PlayerConsistencyTracker
from sklearn.preprocessing import RobustScaler

class NBAFeatureEngineer:
    def __init__(self, log_level='INFO'):
        """Initialize the feature engineer with logging"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=log_level)
        self.logger.setLevel(log_level)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features(self, historical_data, current_data, props_data=None):
        """Main feature engineering pipeline"""
        try:
            self.logger.info("Starting feature creation process")
            all_features = []
            
            # Process each season
            for season in tqdm(historical_data, desc="Processing seasons"):
                try:
                    self.logger.info(f"\nProcessing season {season}")
                    
                    # Get season data
                    game_logs = historical_data[season]['game_logs']
                    matchups = historical_data[season]['matchups']
                    
                    # Create features
                    features = self._create_features(game_logs, matchups)
                    
                    if features is not None and not features.empty:
                        all_features.append(features)
                        self.logger.info(f"Added features with shape: {features.shape}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing season {season}: {str(e)}")
                    continue
            
            # Check if we have any features
            if not all_features:
                raise ValueError("No features were created from any season")
            
            # Combine all seasons
            try:
                # Ensure all feature sets have the same columns
                common_columns = set.intersection(*[set(df.columns) for df in all_features])
                self.logger.info(f"Common columns across seasons: {len(common_columns)}")
                
                # Keep only common columns and concatenate
                aligned_features = [df[list(common_columns)] for df in all_features]
                final_features = pd.concat(aligned_features, axis=0, ignore_index=True)
                
                self.logger.info(f"Final combined features shape: {final_features.shape}")
                
                # Split into train/val/test sets
                return self._split_and_validate(final_features)
                
            except Exception as e:
                self.logger.error(f"Error combining features: {str(e)}")
                raise
            
        except Exception as e:
            self.logger.error(f"Error in feature creation pipeline: {str(e)}")
            raise

    def _create_features(self, game_logs: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:
        """Create all features for a single season"""
        try:
            features = game_logs.copy()
            
            # Calculate FG_PCT and related features first
            if 'FGM' in features.columns and 'FGA' in features.columns:
                features['FG_PCT'] = (features['FGM'] / features['FGA'].replace(0, 1)).astype('float32')
            else:
                features['FG_PCT'] = 0.45  # League average approximation
            
            # Calculate avg_FG_PCT_mean
            features['avg_FG_PCT_mean'] = features.groupby('Player_ID')['FG_PCT'].transform(
                lambda x: x.rolling(10, min_periods=3).mean()
            ).fillna(features['FG_PCT']).astype('float32')
            
            # Calculate player roles first
            player_stats = features.groupby('Player_ID').agg({
                'PTS': 'mean',
                'AST': 'mean',
                'REB': 'mean'
            }).reset_index()
            
            # Initialize roles
            player_stats['player_role'] = 'balanced'
            
            # Assign roles based on statistical thresholds
            pts_threshold = player_stats['PTS'].quantile(0.75)
            ast_threshold = player_stats['AST'].quantile(0.75)
            reb_threshold = player_stats['REB'].quantile(0.75)
            
            player_stats.loc[player_stats['PTS'] > pts_threshold, 'player_role'] = 'scorer'
            player_stats.loc[player_stats['AST'] > ast_threshold, 'player_role'] = 'playmaker'
            player_stats.loc[player_stats['REB'] > reb_threshold, 'player_role'] = 'rebounder'
            
            # Merge roles back to features
            features = features.merge(player_stats[['Player_ID', 'player_role']], on='Player_ID', how='left')
            
            # Define essential columns that must be kept
            essential_cols = [
                'Player_ID', 'PLAYER_NAME',
                'GAME_DATE', 'SEASON', 'MATCHUP',
                'player_role',  # Make sure player_role is included
                'PTS', 'AST', 'REB',
                'PTS_REB', 'PTS_AST', 'PTS_AST_REB'
            ]
            
            # Convert GAME_DATE to datetime
            features['GAME_DATE'] = pd.to_datetime(features['GAME_DATE'])
            
            # Basic game context features
            if 'WL' in features.columns:
                features['game_result'] = (features['WL'] == 'W').astype('float32')
            
            if 'START_POSITION' in features.columns:
                features['is_starter'] = (features['START_POSITION'].notna()).astype('float32')
            
            if 'MATCHUP' in features.columns:
                features['home_away'] = (~features['MATCHUP'].str.contains('@')).astype('float32')
            
            # Calculate rest days
            features['days_rest'] = features.groupby('Player_ID')['GAME_DATE'].diff().dt.days.fillna(2)
            features['optimal_rest'] = ((features['days_rest'] >= 2) & (features['days_rest'] <= 3)).astype('float32')
            features['back_to_back'] = (features['days_rest'] == 1).astype('float32')
            
            # Create recent form and consistency features for key stats
            for stat in ['PTS', 'AST', 'REB']:
                if stat in features.columns:
                    # Sort by date within each player group
                    features = features.sort_values(['Player_ID', 'GAME_DATE'])
                    
                    # Recent form (last 5 games average)
                    features[f'{stat}_recent_form'] = features.groupby('Player_ID')[stat].transform(
                        lambda x: x.rolling(5, min_periods=1).mean()
                    ).fillna(features[stat]).astype('float32')
                    
                    # Consistency score
                    features[f'{stat.lower()}_consistency'] = features.groupby('Player_ID')[stat].transform(
                        lambda x: 1 - (x.rolling(10, min_periods=3).std() / (x.rolling(10, min_periods=3).mean() + 1e-6))
                    ).fillna(0).astype('float32')
                    
                    # Rolling averages
                    features[f'{stat}_rolling_mean'] = features.groupby('Player_ID')[stat].transform(
                        lambda x: x.rolling(10, min_periods=3).mean()
                    ).fillna(features[stat].mean()).astype('float32')
            
            # Convert numeric columns to float32
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in essential_cols:  # Don't convert essential columns
                    features[col] = features[col].astype('float32')
            
            # List of columns to drop
            cols_to_drop = [
                'COMMENT', 'ARENA', 'CITY', 'WL', 'LOCATION', 'START_POSITION',
                'VIDEO_AVAILABLE'
            ]
            
            # Only drop non-essential columns that exist
            cols_to_drop = [col for col in cols_to_drop if col in features.columns]
            if cols_to_drop:
                features = features.drop(columns=cols_to_drop)
            
            # Keep all essential columns plus numeric features
            keep_cols = list(set(essential_cols + numeric_cols.tolist()))
            keep_cols = [col for col in keep_cols if col in features.columns]
            features = features[keep_cols]
            
            # Fill NaN values in numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(0)
            
            self.logger.info(f"Created features with shape: {features.shape}")
            self.logger.debug(f"Columns in final features: {features.columns.tolist()}")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            self.logger.error(f"DataFrame columns: {features.columns.tolist()}")
            return None

    def _process_season_data(self, game_logs, matchups, current_data):
        """Process single season data with memory optimization"""
        try:
            self.logger.info(f"Processing game logs with shape: {game_logs.shape}")
            
            # Validate input data
            if game_logs.empty:
                self.logger.warning("Empty game logs provided")
                return None
            
            # Create features in chunks
            features_list = []
            chunk_size = 1000
            
            total_chunks = (len(game_logs) + chunk_size - 1) // chunk_size
            self.logger.info(f"Processing {total_chunks} chunks...")
            
            for chunk_start in range(0, len(game_logs), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(game_logs))
                chunk = game_logs.iloc[chunk_start:chunk_end].copy()
                
                # Process chunk
                chunk_features = self._create_chunk_features(chunk, matchups)
                
                if chunk_features is not None and not chunk_features.empty:
                    features_list.append(chunk_features)
                    self.logger.debug(f"Processed chunk {len(features_list)}/{total_chunks}")
                else:
                    self.logger.warning(f"No features created for chunk {len(features_list)+1}")
                
                # Clear chunk memory
                del chunk
                gc.collect()
            
            # Check if we have any features
            if not features_list:
                self.logger.warning("No features were created from any chunk")
                return None
            
            # Combine chunks
            self.logger.info(f"Combining {len(features_list)} chunks...")
            season_features = pd.concat(features_list, axis=0)
            self.logger.info(f"Combined season features shape: {season_features.shape}")
            
            del features_list
            gc.collect()
            
            return season_features
            
        except Exception as e:
            self.logger.error(f"Error processing season chunk: {str(e)}")
            return None

    def _split_and_validate(self, features):
        """Split and validate features with memory optimization"""
        try:
            # Split into train/val/test
            train_data, val_data, test_data = self.prepare_train_val_test_split(features)
            
            if train_data is None or val_data is None or test_data is None:
                raise ValueError("Error in train/val/test split")
            
            # Get numeric columns only
            numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
            
            # Create deep copies to prevent data leakage
            train_data = train_data.copy()
            val_data = val_data.copy()
            test_data = test_data.copy()
            
            # Optimize memory for numeric columns only
            for df in [train_data, val_data, test_data]:
                for col in numeric_cols:
                    if col in df.columns:  # Check if column exists
                        df[col] = df[col].astype('float32')
            
            # Validate that each split has the same columns
            all_splits = [train_data, val_data, test_data]
            all_columns = set(train_data.columns)
            for split in all_splits:
                if set(split.columns) != all_columns:
                    missing = all_columns - set(split.columns)
                    self.logger.error(f"Missing columns in split: {missing}")
                    raise ValueError("Inconsistent columns across splits")
            
            # Log split information
            self.logger.info(f"Train shape: {train_data.shape}")
            self.logger.info(f"Validation shape: {val_data.shape}")
            self.logger.info(f"Test shape: {test_data.shape}")
            
            return {
                'train': train_data,
                'validation': val_data,
                'test': test_data,
                'all_features': features
            }
            
        except Exception as e:
            self.logger.error(f"Error in split and validate: {str(e)}")
            raise

    def _create_basic_stats_features(self, game_logs):
        """Create basic statistical features"""
        try:
            features = game_logs.groupby('Player_ID').agg({
                'PTS': ['mean', 'std', 'max'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std'],
                'MIN': ['mean'],
                'FG_PCT': 'mean',
                'FG3_PCT': 'mean',
                'FT_PCT': 'mean'
            })
            
            # Flatten column names
            features.columns = [f"avg_{col[0]}_{col[1]}" for col in features.columns]
            return features.reset_index()
            
        except Exception as e:
            self.logger.error(f"Error creating basic stats features: {str(e)}")
            return pd.DataFrame()

    def _create_matchup_features(self, matchups):
        """Create matchup-based features"""
        try:
            features = matchups.groupby('MATCHUP').agg({
                'WL': lambda x: (x == 'W').mean(),
                'PTS': ['mean', 'std'],
                'PLUS_MINUS': 'mean'
            })
            
            # Flatten column names
            features.columns = [f"matchup_{col[0]}_{col[1]}" for col in features.columns]
            return features.reset_index()
            
        except Exception as e:
            self.logger.error(f"Error creating matchup features: {str(e)}")
            return pd.DataFrame()

    def _create_trend_features(self, game_logs):
        """Create trend-based features"""
        try:
            # Calculate rolling averages for key stats
            features = pd.DataFrame()
            
            # Group by player and sort by date
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player].sort_values('GAME_DATE')
                
                # Calculate 5-game rolling averages
                rolling_stats = {
                    'rolling_pts': player_logs['PTS'].rolling(5, min_periods=1).mean(),
                    'rolling_ast': player_logs['AST'].rolling(5, min_periods=1).mean(),
                    'rolling_reb': player_logs['REB'].rolling(5, min_periods=1).mean(),
                    'rolling_min': player_logs['MIN'].rolling(5, min_periods=1).mean()
                }
                
                player_features = pd.DataFrame(rolling_stats)
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
        except Exception as e:
            self.logger.error(f"Error creating trend features: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame instead of None

    def _create_form_features(self, game_logs):
        """Create current form features"""
        try:
            # Calculate recent performance metrics
            features = pd.DataFrame()
            
            # Group by player and sort by date
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player].sort_values('GAME_DATE')
                
                # Get last 3 games
                last_3 = player_logs.tail(3)
                
                form_stats = {
                    'recent_pts_avg': last_3['PTS'].mean(),
                    'recent_ast_avg': last_3['AST'].mean(),
                    'recent_reb_avg': last_3['REB'].mean(),
                    'recent_min_avg': last_3['MIN'].mean(),
                    'recent_fg_pct': last_3['FG_PCT'].mean()
                }
                
                player_features = pd.DataFrame([form_stats])
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
        except Exception as e:
            self.logger.error(f"Error creating form features: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame instead of None

    def _create_injury_features(self, game_logs, injury_data):
        """Create features based on team injury status"""
        try:
            features = pd.DataFrame()
            
            for team in game_logs['TEAM'].unique():
                team_logs = game_logs[game_logs['TEAM'] == team]
                team_injuries = injury_data[injury_data['team'] == team]
                
                injury_features = {
                    'team_injuries': len(team_injuries),
                    'out_players': len(team_injuries[team_injuries['status'] == 'Out']),
                    'day_to_day_players': len(team_injuries[team_injuries['status'] == 'Day-To-Day']),
                    'position_impact': self._calculate_position_impact(team_injuries)
                }
                
                team_features = pd.DataFrame([injury_features] * len(team_logs))
                team_features['TEAM'] = team
                features = pd.concat([features, team_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating injury features: {str(e)}")
            return pd.DataFrame()

    def _calculate_position_impact(self, team_injuries):
        """Calculate impact score based on injured players' positions"""
        position_weights = {
            'PG': 0.9,  # High impact on assists
            'SG': 0.8,  # High impact on scoring
            'SF': 0.7,  # Balanced impact
            'PF': 0.7,  # Balanced impact
            'C': 0.8,   # High impact on rebounds
        }
        
        impact_score = 0
        for _, player in team_injuries.iterrows():
            if player['status'] == 'Out':
                pos = player['position'].split('/')[0]  # Take primary position
                impact_score += position_weights.get(pos, 0.5)
        
        return impact_score

    def _create_props_features(self, game_logs, props_data):
        """Create features based on props data"""
        try:
            features = pd.DataFrame()
            
            # Group props by player
            for player in game_logs['Player_ID'].unique():
                player_props = props_data[props_data['player'] == player]
                
                if not player_props.empty:
                    props_stats = {
                        'avg_line': player_props['line'].mean(),
                        'line_std': player_props['line'].std(),
                        'over_price_avg': player_props['over_price'].mean(),
                        'under_price_avg': player_props['under_price'].mean(),
                        'line_movement': player_props['line'].diff().mean()
                    }
                    
                    player_features = pd.DataFrame([props_stats])
                    player_features['Player_ID'] = player
                    features = pd.concat([features, player_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating props features: {str(e)}")
            return pd.DataFrame()

    def _create_consistency_features(self, game_logs):
        """Create features for player consistency across seasons"""
        try:
            features = pd.DataFrame()
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player]
                
                consistency_stats = {
                    'pts_consistency': player_logs['PTS'].std() / player_logs['PTS'].mean() if player_logs['PTS'].mean() != 0 else 0,
                    'ast_consistency': player_logs['AST'].std() / player_logs['AST'].mean() if player_logs['AST'].mean() != 0 else 0,
                    'reb_consistency': player_logs['REB'].std() / player_logs['REB'].mean() if player_logs['REB'].mean() != 0 else 0,
                    'min_consistency': player_logs['MIN'].std() / player_logs['MIN'].mean() if player_logs['MIN'].mean() != 0 else 0,
                    'games_played': len(player_logs)
                }
                
                player_features = pd.DataFrame([consistency_stats])
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating consistency features: {str(e)}")
            return pd.DataFrame()

    def _create_home_away_features(self, game_logs):
        """Create features for home/away performance differences"""
        try:
            features = pd.DataFrame()
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player]
                
                # Determine home/away games
                home_games = player_logs[~player_logs['MATCHUP'].str.contains('@')]
                away_games = player_logs[player_logs['MATCHUP'].str.contains('@')]
                
                home_away_stats = {
                    'home_pts_avg': home_games['PTS'].mean() if not home_games.empty else 0,
                    'away_pts_avg': away_games['PTS'].mean() if not away_games.empty else 0,
                    'home_ast_avg': home_games['AST'].mean() if not home_games.empty else 0,
                    'away_ast_avg': away_games['AST'].mean() if not away_games.empty else 0,
                    'home_reb_avg': home_games['REB'].mean() if not home_games.empty else 0,
                    'away_reb_avg': away_games['REB'].mean() if not away_games.empty else 0,
                    'home_away_pts_diff': (home_games['PTS'].mean() - away_games['PTS'].mean()) if not (home_games.empty or away_games.empty) else 0
                }
                
                player_features = pd.DataFrame([home_away_stats])
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating home/away features: {str(e)}")
            return pd.DataFrame()

    def _create_rest_features(self, game_logs):
        """Create features based on days of rest"""
        try:
            features = pd.DataFrame()
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player].sort_values('GAME_DATE')
                
                # Calculate days between games
                player_logs['days_rest'] = player_logs['GAME_DATE'].diff().dt.days
                
                # Group performance by rest days
                rest_stats = {
                    'avg_pts_no_rest': player_logs[player_logs['days_rest'] == 1]['PTS'].mean(),
                    'avg_pts_one_day_rest': player_logs[player_logs['days_rest'] == 2]['PTS'].mean(),
                    'avg_pts_two_plus_rest': player_logs[player_logs['days_rest'] > 2]['PTS'].mean(),
                    'rest_impact_score': (player_logs[player_logs['days_rest'] > 1]['PTS'].mean() - 
                                        player_logs[player_logs['days_rest'] == 1]['PTS'].mean())
                }
                
                player_features = pd.DataFrame([rest_stats])
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating rest features: {str(e)}")
            return pd.DataFrame()

    def _enhance_rest_features(self, game_logs):
        """Create more sophisticated rest features based on findings"""
        try:
            features = pd.DataFrame()
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player].sort_values('GAME_DATE')
                
                # Calculate days between games
                player_logs['days_rest'] = player_logs['GAME_DATE'].diff().dt.days
                
                rest_features = {
                    'optimal_rest': (player_logs['days_rest'] >= 2) & (player_logs['days_rest'] <= 3),
                    'extended_rest': player_logs['days_rest'] > 3,
                    'back_to_back': player_logs['days_rest'] == 1,
                    'rest_performance_ratio': player_logs.groupby('days_rest')['PTS'].transform('mean') / player_logs['PTS'].mean(),
                    'rest_impact_ast': (player_logs[player_logs['days_rest'] > 1]['AST'].mean() - 
                                      player_logs[player_logs['days_rest'] == 1]['AST'].mean()),
                    'rest_impact_reb': (player_logs[player_logs['days_rest'] > 1]['REB'].mean() - 
                                      player_logs[player_logs['days_rest'] == 1]['REB'].mean())
                }
                
                player_df = pd.DataFrame([rest_features])
                player_df['Player_ID'] = player
                features = pd.concat([features, player_df])
            
            return features
        except Exception as e:
            self.logger.error(f"Error creating enhanced rest features: {str(e)}")
            return pd.DataFrame()

    def _create_position_features(self, game_logs, injury_data):
        """Create position-specific features based on injury distribution"""
        try:
            features = pd.DataFrame()
            
            position_weights = {
                'PG': 0.3, 'SG': 0.45,  # Based on injury distribution
                'SF': 0.6, 'PF': 0.6,
                'C': 0.5,
                'G': 0.4, 'F': 0.6      # Generic positions
            }
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player]
                position = player_logs['POS'].iloc[0] if 'POS' in player_logs.columns else None
                
                position_features = {
                    'is_forward': position in ['F', 'SF', 'PF'] if position else False,
                    'is_guard': position in ['G', 'SG', 'PG'] if position else False,
                    'is_center': position == 'C' if position else False,
                    'position_injury_risk': position_weights.get(position, 0.5) if position else 0.5,
                    'position_minutes_impact': self._calculate_position_minutes_impact(position, game_logs)
                }
                
                player_df = pd.DataFrame([position_features])
                player_df['Player_ID'] = player
                features = pd.concat([features, player_df])
            
            return features
        except Exception as e:
            self.logger.error(f"Error creating position features: {str(e)}")
            return pd.DataFrame()

    def _calculate_position_minutes_impact(self, position, game_logs):
        """Helper method to calculate position-specific minutes impact"""
        try:
            if not position:
                return 0
            
            position_games = game_logs[game_logs['POS'] == position]
            all_games = game_logs
            
            if position_games.empty:
                return 0
            
            position_min_avg = position_games['MIN'].mean()
            overall_min_avg = all_games['MIN'].mean()
            
            return (position_min_avg - overall_min_avg) / overall_min_avg if overall_min_avg != 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating position minutes impact: {str(e)}")
            return 0

    def _create_consistency_metrics(self, game_logs):
        """Create enhanced consistency metrics"""
        try:
            features = pd.DataFrame()
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player].sort_values('GAME_DATE')
                
                # Calculate rolling windows
                rolling_5 = player_logs.rolling(5, min_periods=3).agg({
                    'PTS': ['mean', 'std'],
                    'AST': ['mean', 'std'],
                    'REB': ['mean', 'std'],
                    'MIN': ['mean', 'std']
                })
                
                consistency_features = {
                    'pts_consistency': player_logs['PTS'].std() / player_logs['PTS'].mean() if player_logs['PTS'].mean() != 0 else 0,
                    'ast_consistency': player_logs['AST'].std() / player_logs['AST'].mean() if player_logs['AST'].mean() != 0 else 0,
                    'reb_consistency': player_logs['REB'].std() / player_logs['REB'].mean() if player_logs['REB'].mean() != 0 else 0,
                    'recent_form_pts': rolling_5['PTS']['mean'].iloc[-1] if not rolling_5.empty else player_logs['PTS'].mean(),
                    'recent_volatility': rolling_5['PTS']['std'].iloc[-1] if not rolling_5.empty else player_logs['PTS'].std(),
                    'minutes_stability': rolling_5['MIN']['std'].iloc[-1] if not rolling_5.empty else player_logs['MIN'].std(),
                    'performance_trend': self._calculate_trend(player_logs['PTS'].tail(10))
                }
                
                player_df = pd.DataFrame([consistency_features])
                player_df['Player_ID'] = player
                features = pd.concat([features, player_df])
            
            return features
        except Exception as e:
            self.logger.error(f"Error creating consistency metrics: {str(e)}")
            return pd.DataFrame()

    def _calculate_trend(self, series):
        """Helper method to calculate recent trend"""
        try:
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            slope, _ = np.polyfit(x, y, 1)
            return slope
        except Exception as e:
            self.logger.error(f"Error calculating trend: {str(e)}")
            return 0

    def _create_rolling_averages(self, game_logs):
        """Create rolling average features for different time windows"""
        try:
            features = pd.DataFrame()
            windows = [7, 14, 30]
            stats = ['PTS', 'AST', 'REB', 'MIN']
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player].sort_values('GAME_DATE')
                
                rolling_stats = {}
                for window in windows:
                    for stat in stats:
                        rolling_stats[f'{stat}_rolling_{window}d'] = player_logs[stat].rolling(window, min_periods=1).mean()
                        rolling_stats[f'{stat}_rolling_{window}d_std'] = player_logs[stat].rolling(window, min_periods=1).std()
                
                player_features = pd.DataFrame(rolling_stats)
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating rolling averages: {str(e)}")
            return pd.DataFrame()

    def _create_minutes_role_features(self, game_logs):
        """Create features based on minutes and role combinations"""
        try:
            features = pd.DataFrame()
            
            for player in game_logs['Player_ID'].unique():
                player_logs = game_logs[game_logs['Player_ID'] == player]
                
                # Calculate role based on statistical tendencies
                avg_stats = player_logs.mean()
                role = self._determine_player_role(avg_stats)
                
                # Create minutes interaction features
                minutes_features = {
                    'minutes_role_interaction': player_logs['MIN'] * (role == 'scorer'),
                    'role': role
                }
                
                player_features = pd.DataFrame(minutes_features)
                player_features['Player_ID'] = player
                features = pd.concat([features, player_features])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating minutes-role features: {str(e)}")
            return pd.DataFrame()

    def _determine_player_role(self, avg_stats):
        """Determine player role based on statistical profile"""
        if avg_stats['AST'] > 6:
            return 'playmaker'
        elif avg_stats['REB'] > 8:
            return 'rebounder'
        elif avg_stats['PTS'] > 15:
            return 'scorer'
        else:
            return 'balanced'

    def _add_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add target-related features"""
        try:
            df = data.copy()
            
            # Reset index to handle duplicates
            df = df.reset_index()
            
            # List of base stats to create features for
            base_stats = ['PTS', 'AST', 'REB']
            
            for stat in base_stats:
                if stat not in df.columns:
                    continue
                    
                try:
                    # Recent form (last 5 games average normalized)
                    recent_form = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.rolling(window=5, min_periods=1).mean()
                    )
                    
                    # Normalize to -3 to 3 range
                    mean = recent_form.mean()
                    std = recent_form.std()
                    df[f'{stat}_recent_form'] = ((recent_form - mean) / std).clip(-3, 3)
                    
                    # Consistency features
                    rolling_std = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.rolling(window=10, min_periods=1).std()
                    )
                    
                    rolling_mean = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.rolling(window=10, min_periods=1).mean()
                    )
                    
                    # Coefficient of variation (normalized)
                    cv = (rolling_std / rolling_mean).replace([np.inf, -np.inf], np.nan)
                    df[f'{stat.lower()}_consistency'] = (1 - cv).clip(-3, 3)
                    
                    # Average stats
                    df[f'avg_{stat}_mean'] = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.expanding(min_periods=1).mean()
                    )
                    
                    df[f'avg_{stat}_std'] = df.groupby('PLAYER_ID')[stat].transform(
                        lambda x: x.expanding(min_periods=1).std()
                    )
                    
                    # Normalize expanding features
                    for col in [f'avg_{stat}_mean', f'avg_{stat}_std']:
                        mean = df[col].mean()
                        std = df[col].std()
                        df[col] = ((df[col] - mean) / std).clip(-3, 3)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {stat}: {str(e)}")
                    continue
            
            # Set index back
            df = df.set_index('PLAYER_ID')
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            # Log feature creation summary
            created_features = [col for col in df.columns if col not in data.columns]
            self.logger.info(f"Created {len(created_features)} new features: {created_features}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding target variables: {str(e)}")
            raise

    def _scale_features(self, df):
        """Scale numerical features"""
        try:
            # Only scale features, not targets
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_cols = [col for col in numerical_cols if not any(x in col.lower() for x in ['pts', 'ast', 'reb'])]
            
            # Use RobustScaler instead of StandardScaler for better handling of outliers
            scaler = RobustScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            return df
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            return df

    def prepare_train_val_test_split(self, df):
        """Split data chronologically into train/val/test sets"""
        try:
            # Sort by date
            df = df.sort_values('GAME_DATE')
            
            # Calculate split points (70% train, 15% val, 15% test)
            n = len(df)
            train_idx = int(n * 0.7)
            val_idx = int(n * 0.85)
            
            # Split the data
            train_data = df.iloc[:train_idx]
            val_data = df.iloc[train_idx:val_idx]
            test_data = df.iloc[val_idx:]
            
            self.logger.info(f"Train set: {len(train_data)} samples")
            self.logger.info(f"Validation set: {len(val_data)} samples")
            self.logger.info(f"Test set: {len(test_data)} samples")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            return None, None, None

    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        try:
            # Identify categorical columns
            categorical_cols = [
                'TEAM_ABBREVIATION', 
                'MATCHUP',
                'role',
                'POSITION',
                'WL'
            ]
            
            # Initialize encoders if not exists
            for col in categorical_cols:
                if col in df.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    
                    # Fit and transform
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('UNKNOWN'))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {str(e)}")
            return df

    def enhance_form_indicators(self, df):
        """Add enhanced form indicators"""
        try:
            stats = ['PTS', 'AST', 'REB']
            
            for stat in stats:
                # Recent form (last 5 games vs season average)
                df[f'{stat}_recent_form'] = df.groupby('Player_ID')[stat].transform(
                    lambda x: x.rolling(5, min_periods=1).mean() / x.mean()
                )
                
                # Consistency score (inverse of coefficient of variation)
                df[f'{stat}_consistency'] = df.groupby('Player_ID')[stat].transform(
                    lambda x: 1 - (x.rolling(10, min_periods=1).std() / x.rolling(10, min_periods=1).mean())
                )
                
                # Hot/Cold streak indicator
                df[f'{stat}_streak'] = df.groupby('Player_ID')[stat].transform(
                    lambda x: (x > x.rolling(5, min_periods=1).mean()).astype(int).groupby(
                        (x <= x.rolling(5, min_periods=1).mean()).astype(int).cumsum()
                    )
                )
                
                
                
                # Performance trend
                df[f'{stat}_trend'] = df.groupby('Player_ID')[stat].transform(
                    lambda x: x.rolling(5, min_periods=1).mean() - x.rolling(10, min_periods=1).mean())
                
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error enhancing form indicators: {str(e)}")
            return df

    def validate_features(self, df):
        """Validate feature quality before modeling"""
        from tqdm import tqdm
        
        try:
            validation_steps = [
                ("Checking missing values", self._check_missing_values),
                ("Checking correlations", self._check_correlations),
                ("Checking distributions", self._check_distributions),
                ("Checking class balance", self._check_class_balance),
                ("Calculating feature importance", self._calculate_feature_importance)
            ]
            
            for desc, validation_func in tqdm(validation_steps, desc="Validation steps"):
                try:
                    validation_func(df)
                except Exception as e:
                    self.logger.error(f"Error in {desc}: {str(e)}")
            
            return df, True
            
        except Exception as e:
            self.logger.error(f"Error validating features: {str(e)}")
            return df, False
    
    def _check_missing_values(self, df):
        """Check and handle missing values"""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        significant_missing = missing_pct[missing_pct > 5].to_dict()
        
        if significant_missing:
            self.logger.warning(f"Features with >5% missing values: {significant_missing}")
            for col in df.columns:
                if col in ['PLAYER_ID', 'PLAYER_NAME']:
                    continue
                elif df[col].dtype in ['float64', 'int64']:
                    if '_rolling' in col or 'matchup_' in col:
                        df[col] = df.groupby('PLAYER_ID')[col].transform(
                            lambda x: x.fillna(x.mean())
                        )
                    else:
                        df[col] = df[col].fillna(df[col].mean())
    
    def _check_correlations(self, df):
        """Check feature correlations"""
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # Calculate correlations in chunks to save memory
        chunk_size = 50
        high_corr_pairs = []
        
        for i in range(0, len(numerical_cols), chunk_size):
            chunk_cols = numerical_cols[i:i + chunk_size]
            corr_chunk = df[chunk_cols].corr()
            
            # Find high correlations in this chunk
            for col1 in corr_chunk.columns:
                for col2 in corr_chunk.index:
                    if col1 < col2 and abs(corr_chunk.loc[col2, col1]) > 0.95:
                        high_corr_pairs.append((col1, col2, corr_chunk.loc[col2, col1]))
        
        if high_corr_pairs:
            self.logger.warning("Highly correlated feature pairs:")
            for feat1, feat2, corr in high_corr_pairs:
                self.logger.warning(f"{feat1} - {feat2}: {corr:.3f}")
    
    def _check_distributions(self, df):
        """Check feature distributions"""
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        skewed_features = []
        
        for col in numerical_cols:
            if col not in ['target_pts_over', 'target_ast_over', 'target_reb_over']:
                skewness = df[col].skew()
                if abs(skewness) > 3:
                    skewed_features.append((col, skewness))
        
        if skewed_features:
            self.logger.warning("Highly skewed features:")
            for feat, skew in skewed_features:
                self.logger.warning(f"{feat}: {skew:.3f}")
    
    def _check_class_balance(self, df):
        """Check class balance for binary targets"""
        for target in ['target_pts_over', 'target_ast_over', 'target_reb_over']:
            if target in df.columns:
                class_dist = df[target].value_counts(normalize=True)
                if min(class_dist) < 0.3:
                    self.logger.warning(f"Class imbalance in {target}: {dict(class_dist)}")
    
    def _calculate_feature_importance(self, df):
        """Calculate feature importance for each target"""
        try:
            # Sample data to reduce memory usage
            sample_size = min(5000, len(df))  # Reduced from 100000
            sample_idx = np.random.choice(df.index, sample_size, replace=False)
            
            # Select only numeric columns
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            for target in ['PTS', 'AST', 'REB']:
                if target in df.columns:
                    try:
                        # Select features and convert to float32
                        X = df.loc[sample_idx, numerical_cols].drop([target], axis=1, errors='ignore')
                        X = X.astype('float32')
                        y = df.loc[sample_idx, target].astype('float32')
                        
                        # Clean data
                        X = X.fillna(X.mean())
                        y = y.fillna(y.mean())
                        
                        # Calculate feature importance using mutual information
                        # with chunking to save memory
                        chunk_size = 10
                        mi_scores = []
                        
                        for i in range(0, len(X.columns), chunk_size):
                            chunk_cols = X.columns[i:i + chunk_size]
                            chunk_scores = mutual_info_regression(X[chunk_cols], y)
                            mi_scores.extend(chunk_scores)
                        
                        feature_importance = pd.Series(mi_scores, index=X.columns)
                        top_features = feature_importance.nlargest(10)
                        
                        self.logger.info(f"\nTop 10 features for {target}:")
                        for feat, score in top_features.items():
                            self.logger.info(f"{feat}: {score:.3f}")
                            
                        # Clear memory
                        del X, y
                        gc.collect()
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating feature importance for {target}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error in feature importance calculation: {str(e)}")
        finally:
            gc.collect()

    def calculate_feature_importance(self, data: pd.DataFrame, target: str, max_samples: int = 50000) -> pd.Series:
        """Calculate feature importance with memory efficient approach"""
        try:
            # Sample data if it's too large (reduced to 10k samples)
            if len(data) > max_samples:
                self.logger.info(f"Sampling {max_samples} rows for feature importance calculation")
                data = data.sample(n=max_samples, random_state=42)
            
            # Select only numeric features
            features = data.select_dtypes(include=[np.number]).columns
            features = [f for f in features if f != target]
            
            if not features:
                self.logger.warning(f"No numeric features found for {target}")
                return pd.Series()
            
            # Prepare X and y with memory optimization
            X = data[features].astype('float32')  # Use float32 instead of float64
            y = data[target].astype('float32')
            
            # Clean the data
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Create and train a lightweight model for feature importance
            model = xgb.XGBRegressor(
                n_estimators=25,  # Further reduced number of trees
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=2,  # Limit parallel jobs
                tree_method='hist'  # Memory efficient tree method
            )
            
            # Train model
            model.fit(X, y, verbose=False)
            
            # Get feature importance
            importance = pd.Series(
                model.feature_importances_,
                index=features
            ).sort_values(ascending=False)
            
            # Log top features
            self.logger.info(f"\nTop 10 features for {target}:")
            for feature, score in importance.head(10).items():
                self.logger.info(f"{feature}: {score:.3f}")
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance for {target}: {str(e)}")
            return pd.Series()
        finally:
            # Force garbage collection
            gc.collect()

    def _create_chunk_features(self, chunk: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:
        try:
            features = chunk.copy()
            
            # Define all columns that should remain non-numeric
            non_numeric_cols = [
                # Identifiers
                'Player_ID', 'PLAYER_ID', 'PLAYER_NAME', 'TeamID', 'TEAM_ID', 'TEAM_NAME',
                # Date and time
                'GAME_DATE', 'GAME_TIME', 'SEASON',
                # Text fields
                'MATCHUP', 'WL', 'TEAM_ABBREVIATION', 'TEAM_CITY',
                'START_POSITION', 'COMMENT',
                # Location
                'LOCATION', 'ARENA', 'CITY',
                # Other categorical
                'player_role', 'position', 'POS'
            ]
            
            # Convert categorical variables to numeric where appropriate
            if 'WL' in features.columns:
                features['game_result'] = (features['WL'] == 'W').astype('float32')
            
            if 'START_POSITION' in features.columns:
                features['is_starter'] = (features['START_POSITION'].notna()).astype('float32')
            
            if 'LOCATION' in features.columns:
                features['is_home'] = (features['LOCATION'] == 'H').astype('float32')
            
            # Calculate FG_PCT if not present
            if 'FG_PCT' not in features.columns and 'FGM' in features.columns and 'FGA' in features.columns:
                features['FG_PCT'] = (features['FGM'] / features['FGA'].replace(0, 1)).astype('float32')
            
            # Calculate rolling averages for shooting percentages
            if 'FG_PCT' in features.columns:
                features['avg_FG_PCT_mean'] = features.groupby('Player_ID')['FG_PCT'].transform(
                    lambda x: x.rolling(10, min_periods=3).mean()
                ).fillna(features['FG_PCT']).astype('float32')
            
            # Convert only numeric columns to float32
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in non_numeric_cols:
                    try:
                        features[col] = features[col].astype('float32')
                    except Exception as e:
                        self.logger.warning(f"Could not convert {col} to float32: {str(e)}")
            
            # List of columns to drop
            cols_to_drop = [
                'COMMENT', 'ARENA', 'CITY', 'WL', 'LOCATION', 'START_POSITION',
                'VIDEO_AVAILABLE'
            ]
            
            # Drop columns if they exist in the DataFrame's columns
            existing_cols_to_drop = [col for col in cols_to_drop if col in features.columns]
            if existing_cols_to_drop:
                features = features.drop(columns=existing_cols_to_drop)
            
            # Keep only necessary columns
            essential_non_numeric = ['Player_ID', 'GAME_DATE', 'PLAYER_NAME', 'SEASON', 'MATCHUP']
            essential_numeric = ['avg_FG_PCT_mean']  # Add to essential numeric columns
            keep_cols = (numeric_cols.tolist() + 
                        [col for col in essential_non_numeric if col in features.columns] +
                        [col for col in essential_numeric if col in features.columns])
            features = features[keep_cols]
            
            # Fill NaN values only in numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(0)
            
            # Log the final column types
            self.logger.debug("Final column types:")
            for col in features.columns:
                self.logger.debug(f"{col}: {features[col].dtype}")
            
            # Add consistency features
            consistency_tracker = PlayerConsistencyTracker()
            features = consistency_tracker.get_consistency_features(features)
            
            # Add to essential numeric columns
            essential_numeric.extend(['pts_consistency', 'ast_consistency', 'reb_consistency'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error creating chunk features: {str(e)}")
            self.logger.error(f"Chunk head: {chunk.head().to_string()}")
            raise

    def _prepare_matchup_data(self, matchups: pd.DataFrame) -> pd.DataFrame:
        """Prepare matchup data by calculating required statistics"""
        try:
            if matchups.empty:
                raise ValueError("Empty matchups DataFrame")
            
            prepared_matchups = matchups.copy()
            
            # Calculate PTS_AGAINST
            if 'PTS_AGAINST' not in prepared_matchups.columns:
                if 'PTS' in prepared_matchups.columns:
                    prepared_matchups['PTS_AGAINST'] = prepared_matchups.groupby('MATCHUP')['PTS'].transform('mean')
                else:
                    # If no PTS data, use a default value
                    prepared_matchups['PTS_AGAINST'] = 100.0  # League average approximation
                    
            # Calculate DEF_RATING
            if 'DEF_RATING' not in prepared_matchups.columns:
                if 'PTS' in prepared_matchups.columns and 'POSS' in prepared_matchups.columns:
                    prepared_matchups['DEF_RATING'] = (prepared_matchups['PTS'] * 100) / prepared_matchups['POSS'].clip(lower=1)
                else:
                    # If we can't calculate DEF_RATING, use PTS_AGAINST as an approximation
                    prepared_matchups['DEF_RATING'] = prepared_matchups['PTS_AGAINST']
                
            # Ensure columns are float32
            for col in ['PTS_AGAINST', 'DEF_RATING']:
                prepared_matchups[col] = prepared_matchups[col].astype('float32')
            
            self.logger.info(f"Prepared matchup data with shape: {prepared_matchups.shape}")
            
            return prepared_matchups
            
        except Exception as e:
            self.logger.error(f"Error preparing matchup data: {str(e)}")
            raise

    def select_optimal_features(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Select features based on importance scores"""
        try:
            # Train a simple model to get feature importance
            model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1
            )
            
            # Prepare features
            X = df.select_dtypes(include=[np.number]).fillna(0)
            y = df[target] if target in df else None
            
            if y is None:
                return X
            
            # Fit model and get importance scores
            model.fit(X, y)
            importance = pd.Series(model.feature_importances_, index=X.columns)
            
            # Select top features (keep those that explain 95% of variance)
            importance = importance.sort_values(ascending=False)
            cumsum = importance.cumsum()
            n_features = (cumsum <= 0.95).sum()
            
            selected_features = importance.head(n_features).index.tolist()
            return df[selected_features]
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            return df

    def create_features(self, historical_data, current_data, props_data=None):
        try:
            # Add rolling averages with different windows
            for stat in ['PTS', 'AST', 'REB']:
                for window in [3, 5, 10, 20]:
                    historical_data['player_stats'][f'{stat}_rolling_{window}'] = (
                        historical_data['player_stats']
                        .groupby('PLAYER_NAME')[stat]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    )
                    
            # Add opponent strength metrics
            historical_data['player_stats']['opp_def_rating'] = (
                historical_data['player_stats']
                .groupby('OPPONENT')['PTS']
                .transform('mean')
            )
            
            # Add home/away impact
            historical_data['player_stats']['is_home'] = (
                ~historical_data['player_stats']['MATCHUP'].str.contains('@')
            ).astype(int)
            
            # Add rest days impact
            historical_data['player_stats']['days_rest'] = (
                historical_data['player_stats']
                .groupby('PLAYER_NAME')['GAME_DATE']
                .diff()
                .dt.days
                .fillna(2)
            )
            
            # Add recent form (last 5 games vs season average)
            for stat in ['PTS', 'AST', 'REB']:
                season_avg = historical_data['player_stats'].groupby('PLAYER_NAME')[stat].transform('mean')
                recent_avg = historical_data['player_stats'].groupby('PLAYER_NAME')[stat].transform(
                    lambda x: x.rolling(5, min_periods=1).mean()
                )
                historical_data['player_stats'][f'{stat}_form'] = recent_avg - season_avg
                
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error in feature creation: {str(e)}")
            raise

    # Add other feature creation methods as needed