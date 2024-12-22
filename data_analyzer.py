import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
import logging
from typing import List
import xgboost as xgb
from contextlib import contextmanager
import signal
import threading
from functools import wraps

class DataAnalyzer:
    def __init__(self, position_mapping=None):
        self.analysis_results = {}
        self.logger = logging.getLogger(__name__)
        self.position_mapping = position_mapping or {}
        
    def analyze_target_distribution(self, data, market_types):
        """Analyze the distribution of target variables with enhanced metrics"""
        try:
            analysis_results = {}
            for market in market_types:
                market_data = data[data['market'] == market]
                
                # Enhanced statistics
                stats_dict = {
                    'mean': market_data['actual_value'].mean(),
                    'median': market_data['actual_value'].median(),
                    'std': market_data['actual_value'].std(),
                    'skew': scipy_stats.skew(market_data['actual_value']),
                    'kurtosis': scipy_stats.kurtosis(market_data['actual_value']),
                    'samples': len(market_data),
                    'iqr': market_data['actual_value'].quantile(0.75) - market_data['actual_value'].quantile(0.25),
                    'cv': market_data['actual_value'].std() / market_data['actual_value'].mean() if market_data['actual_value'].mean() != 0 else 0
                }
                
                # Add trend analysis
                if 'GAME_DATE' in market_data.columns:
                    market_data = market_data.sort_values('GAME_DATE')
                    stats_dict['trend'] = np.polyfit(range(len(market_data)), market_data['actual_value'], 1)[0]
                
                analysis_results[market] = stats_dict
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing target distribution: {str(e)}")
            return {}
    
    def analyze_feature_correlations(self, data, market_types):
        """Analyze feature correlations with target"""
        for market in market_types:
            market_data = data[data['market'] == market]
            
            # Get numeric columns
            numeric_cols = market_data.select_dtypes(include=[np.number]).columns
            
            # Calculate correlations with target
            correlations = market_data[numeric_cols].corr()['actual_value'].sort_values(ascending=False)
            
            # Save top correlations
            self.analysis_results[f'{market}_correlations'] = correlations
            
            # Plot correlation heatmap for top features
            plt.figure(figsize=(12, 8))
            top_features = correlations.head(10).index
            sns.heatmap(market_data[top_features].corr(), annot=True, cmap='coolwarm')
            plt.title(f'Top Feature Correlations - {market}')
            plt.savefig(f'analysis/{market}_correlations.png')
            plt.close()
    
    def check_data_quality(self, data):
        """Check for data quality issues"""
        quality_report = {
            'missing_values': data.isnull().sum(),
            'sample_count': len(data),
            'outliers': {}
        }
        
        # Check for outliers in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                z_scores = np.abs(scipy_stats.zscore(data[col].dropna()))
                quality_report['outliers'][col] = (z_scores > 3).sum()
            except:
                self.logger.warning(f"Could not calculate outliers for column: {col}")
        
        # Add basic statistics for numeric columns
        quality_report['statistics'] = data[numeric_cols].describe()
        
        self.analysis_results['data_quality'] = quality_report
        
        # Log key findings
        self.logger.info(f"Total samples: {quality_report['sample_count']}")
        self.logger.info(f"Columns with >5% missing values: {data.isnull().mean()[data.isnull().mean() > 0.05]}")
        
        return quality_report
    
    def analyze_feature_importance(self, data: pd.DataFrame, market_types: List[str]) -> None:
        """Analyze feature importance for different market types"""
        try:
            # Automatically sample if data is large
            if len(data) > 50000:
                data = data.sample(n=50000, random_state=42)
                
            for market in market_types:
                if market not in data.columns:
                    continue
                    
                # Select relevant features
                features = data.select_dtypes(include=[np.number]).columns
                features = [f for f in features if f != market]
                
                if not features:
                    self.logger.warning(f"No numeric features found for {market}")
                    continue
                    
                # Clean the data
                X = data[features].copy()
                y = data[market].copy()
                
                # Replace infinities and fill NaN values
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())
                
                # Create and train a simple model
                model = xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                
                # Add timeout for training
                with timeout(seconds=30):
                    model.fit(X, y)
                    
                    # Get feature importance
                    importance = pd.Series(
                        model.feature_importances_,
                        index=features
                    ).sort_values(ascending=False)
                    
                    # Log top 10 features
                    self.logger.info(f"\nTop 10 features for {market}:")
                    for feature, score in importance.head(10).items():
                        self.logger.info(f"{feature}: {score:.3f}")
                        
        except TimeoutError:
            self.logger.warning("Feature importance analysis timed out")
        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {str(e)}")
        finally:
            # Force garbage collection
            import gc
            gc.collect()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = []
        
        for key, value in self.analysis_results.items():
            report.append(f"\n=== {key} ===")
            report.append(str(value))
        
        with open('analysis/data_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
    
    def analyze_player_trends(self, historical_data):
        """Analyze player performance trends across seasons"""
        trends = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            
            # Calculate per-player trends
            player_stats = game_logs.groupby('Player_ID').agg({
                'PTS': ['mean', 'std', 'max'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std'],
                'MIN': ['mean']
            }).reset_index()
            
            trends[season] = player_stats
        
        self.analysis_results['player_trends'] = trends
        
        # Visualize trends
        plt.figure(figsize=(12, 6))
        for season in trends:
            season_stats = trends[season]
            plt.plot(season_stats['PTS']['mean'], label=f'Season {season}')
        plt.title('Player Scoring Trends Across Seasons')
        plt.xlabel('Player Index')
        plt.ylabel('Average Points')
        plt.legend()
        plt.savefig('analysis/player_trends.png')
        plt.close()
    
    def analyze_matchup_history(self, historical_data):
        """Analyze historical matchup data"""
        matchup_stats = {}
        
        for season, data in historical_data.items():
            matchups = data['matchups']
            
            # Calculate head-to-head stats
            h2h_stats = matchups.groupby(['TEAM_ABBREVIATION', 'MATCHUP']).agg({
                'WL': lambda x: (x == 'W').mean(),
                'PTS': 'mean',
                'PLUS_MINUS': 'mean'
            }).reset_index()
            
            matchup_stats[season] = h2h_stats
        
        self.analysis_results['matchup_history'] = matchup_stats
    
    def analyze_injury_impact(self, injury_data, historical_data):
        """Analyze how injuries affect team and player performance"""
        injury_impact = {}
        
        for season in historical_data:
            game_logs = historical_data[season]['game_logs']
            
            # Create team injury counts
            team_injuries = injury_data.groupby('team').agg({
                'player': 'count',
                'status': lambda x: (x == 'Out').sum()
            }).rename(columns={
                'player': 'total_injuries',
                'status': 'out_count'
            })
            
            # Position-specific injury impact
            position_injuries = injury_data.groupby(['team', 'position']).size().unstack(fill_value=0)
            
            # Analyze team and player performance
            team_stats = {}
            for team in team_injuries.index:
                team_games = game_logs[game_logs['TEAM'] == team]
                if not team_games.empty:
                    # Get injured players for this team
                    team_injured = injury_data[injury_data['team'] == team]
                    injured_positions = team_injured['position'].unique()
                    
                    # Calculate position-specific impacts
                    pos_impact = {}
                    for pos in injured_positions:
                        pos_players = team_games[team_games['position'] == pos]
                        if not pos_players.empty:
                            pos_impact[pos] = {
                                'avg_pts': pos_players['PTS'].mean(),
                                'avg_ast': pos_players['AST'].mean(),
                                'avg_reb': pos_players['REB'].mean(),
                                'players_out': len(team_injured[team_injured['position'] == pos])
                            }
                    
                    team_stats[team] = {
                        'avg_pts_with_injuries': team_games['PTS'].mean(),
                        'avg_ast_with_injuries': team_games['AST'].mean(),
                        'avg_reb_with_injuries': team_games['REB'].mean(),
                        'injury_count': team_injuries.loc[team, 'total_injuries'],
                        'players_out': team_injuries.loc[team, 'out_count'],
                        'position_impact': pos_impact,
                        'key_players_out': len(team_injured[team_injured['status'] == 'Out']),
                        'day_to_day_players': len(team_injured[team_injured['status'] == 'Day-To-Day'])
                    }
            
            injury_impact[season] = {
                'team_stats': team_stats,
                'position_distribution': position_injuries.to_dict(),
                'total_injuries': len(injury_data),
                'out_percentage': (injury_data['status'] == 'Out').mean() * 100
            }
        
        # Add visualization
        self._visualize_injury_impact(injury_impact)
        
        self.analysis_results['injury_impact'] = injury_impact
        return injury_impact
    
    def _visualize_injury_impact(self, injury_impact):
        """Visualize injury impact analysis"""
        plt.figure(figsize=(12, 6))
        
        # Plot injury distribution by position
        latest_season = list(injury_impact.keys())[0]
        position_data = pd.DataFrame(injury_impact[latest_season]['position_distribution'])
        
        sns.barplot(data=position_data)
        plt.title('Injury Distribution by Position')
        plt.xlabel('Team')
        plt.ylabel('Number of Injuries')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis/injury_distribution.png')
        plt.close()
    
    def visualize_player_performance(self, historical_data):
        """Create detailed visualizations of player performance trends"""
        plt.style.use('seaborn')
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Performance Trends Across Seasons', fontsize=16)
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            
            # Points Distribution
            sns.kdeplot(data=game_logs['PTS'], ax=axes[0,0], label=season)
            axes[0,0].set_title('Points Distribution')
            
            # Minutes vs Points
            sns.scatterplot(data=game_logs, x='MIN', y='PTS', alpha=0.5, ax=axes[0,1], label=season)
            axes[0,1].set_title('Minutes vs Points')
            
            # Assists Distribution
            sns.kdeplot(data=game_logs['AST'], ax=axes[1,0], label=season)
            axes[1,0].set_title('Assists Distribution')
            
            # Rebounds Distribution
            sns.kdeplot(data=game_logs['REB'], ax=axes[1,1], label=season)
            axes[1,1].set_title('Rebounds Distribution')
        
        plt.tight_layout()
        plt.savefig('analysis/player_performance_trends.png')
        plt.close()
        
        self.analysis_results['performance_visualizations'] = 'Saved to player_performance_trends.png'
    
    def analyze_data_quality_for_props(self, historical_data):
        """Analyze data quality specifically for prop betting predictions"""
        prop_stats = {}
        
        # Props we care about
        props = {
            'PTS': ['PTS'],
            'AST': ['AST'],
            'REB': ['REB'],
            'PTS+AST': ['PTS', 'AST'],
            'PTS+REB': ['PTS', 'REB'],
            'PRA': ['PTS', 'REB', 'AST']
        }
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            season_stats = {}
            
            for prop_name, columns in props.items():
                # Calculate completeness
                data_completeness = (game_logs[columns].notna().all(axis=1).sum() / len(game_logs)) * 100
                
                # Calculate consistency (coefficient of variation)
                if len(columns) > 1:
                    prop_values = game_logs[columns].sum(axis=1)
                else:
                    prop_values = game_logs[columns[0]]
                
                cv = (prop_values.std() / prop_values.mean()) * 100 if prop_values.mean() != 0 else 0
                
                # New metrics for prop betting
                stats = {
                    'completeness_%': round(data_completeness, 2),
                    'consistency_cv_%': round(cv, 2),
                    'sample_size': len(prop_values.dropna()),
                    'mean': round(prop_values.mean(), 2),
                    'std': round(prop_values.std(), 2),
                    'min': round(prop_values.min(), 2),
                    'max': round(prop_values.max(), 2),
                    'null_count': prop_values.isna().sum(),
                    # New metrics
                    'median': round(prop_values.median(), 2),
                    'quartiles': {
                        '25%': round(prop_values.quantile(0.25), 2),
                        '75%': round(prop_values.quantile(0.75), 2)
                    },
                    'volatility': round(prop_values.std() / prop_values.mean() if prop_values.mean() != 0 else 0, 3),
                    'games_above_mean': (prop_values > prop_values.mean()).sum(),
                    'games_below_mean': (prop_values < prop_values.mean()).sum(),
                    'zero_games': (prop_values == 0).sum()
                }
                
                season_stats[prop_name] = stats
            
            prop_stats[season] = season_stats
        
        # Add visualization
        self._visualize_prop_distributions(prop_stats)
        
        self.analysis_results['prop_data_quality'] = prop_stats
        return prop_stats
    
    def _visualize_prop_distributions(self, prop_stats):
        """Create visualizations for prop distributions"""
        plt.figure(figsize=(15, 10))
        
        for i, (prop_name, _) in enumerate(prop_stats[list(prop_stats.keys())[0]].items()):
            plt.subplot(2, 3, i+1)
            for season in prop_stats:
                stats = prop_stats[season][prop_name]
                plt.hist([stats['mean']], alpha=0.5, label=season)
            plt.title(f'{prop_name} Distribution')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('analysis/prop_distributions.png')
        plt.close()
    
    def analyze_minutes_impact(self, historical_data):
        """Analyze how minutes played affects prop performance"""
        minutes_impact = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            
            # Calculate correlation between minutes and props for players with sufficient games
            correlations = {}
            for stat in ['PTS', 'AST', 'REB']:
                player_correlations = []
                for player_id in game_logs['Player_ID'].unique():
                    player_data = game_logs[game_logs['Player_ID'] == player_id]
                    if len(player_data) >= 5:  # Only include players with 5+ games
                        try:
                            if player_data['MIN'].std() != 0 and player_data[stat].std() != 0:
                                corr = player_data['MIN'].corr(player_data[stat])
                                if not np.isnan(corr):
                                    player_correlations.append(corr)
                        except Exception as e:
                            self.logger.warning(f"Error calculating correlation for player {player_id}: {e}")
                
                correlations[stat] = {
                    'mean_correlation': np.mean(player_correlations) if player_correlations else 0,
                    'median_correlation': np.median(player_correlations) if player_correlations else 0,
                    'sample_size': len(player_correlations)
                }
            
            # Add minutes-based features with error handling
            try:
                game_logs['MIN_rolling_3'] = game_logs.groupby('Player_ID')['MIN'].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )
                game_logs['MIN_trend'] = game_logs.groupby('Player_ID')['MIN'].transform(
                    lambda x: x.diff().fillna(0)
                )
            except Exception as e:
                self.logger.error(f"Error calculating minutes features: {e}")
                game_logs['MIN_rolling_3'] = 0
                game_logs['MIN_trend'] = 0
            
            minutes_impact[season] = {
                'correlations': correlations,
                'avg_min_trend': game_logs['MIN_trend'].mean(),
                'min_volatility': game_logs.groupby('Player_ID')['MIN'].std().mean()
            }
        
        self.analysis_results['minutes_impact'] = minutes_impact
        return minutes_impact
    
    def calculate_player_consistency(self, historical_data):
        """Score players based on prop consistency"""
        consistency_scores = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            season_scores = {}
            
            for player_id in game_logs['Player_ID'].unique():
                player_games = game_logs[game_logs['Player_ID'] == player_id]
                if len(player_games) < 5:  # Skip players with too few games
                    continue
                    
                # Calculate coefficient of variation for each prop
                consistency = {
                    'PTS': player_games['PTS'].std() / player_games['PTS'].mean() if player_games['PTS'].mean() != 0 else float('inf'),
                    'AST': player_games['AST'].std() / player_games['AST'].mean() if player_games['AST'].mean() != 0 else float('inf'),
                    'REB': player_games['REB'].std() / player_games['REB'].mean() if player_games['REB'].mean() != 0 else float('inf'),
                    'games_played': len(player_games),
                    'avg_minutes': player_games['MIN'].mean()
                }
                
                season_scores[player_id] = consistency
            
            consistency_scores[season] = season_scores
        
        self.analysis_results['player_consistency'] = consistency_scores
        return consistency_scores
    
    def analyze_opponent_impact(self, historical_data):
        """Analyze how opponent defense affects prop performance"""
        opponent_impact = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            matchups = data['matchups']
            
            # Calculate opponent defensive ratings
            opp_defense = matchups.groupby('TEAM_ABBREVIATION').agg({
                'PTS': 'mean',
                'PLUS_MINUS': 'mean'
            }).reset_index()
            
            # Add opponent defensive rating to game logs
            def get_opponent(matchup):
                parts = matchup.split()
                return parts[-1] if '@' in matchup else parts[-1]
                
            game_logs['opponent'] = game_logs['MATCHUP'].apply(get_opponent)
            game_logs = game_logs.merge(
                opp_defense,
                left_on='opponent',
                right_on='TEAM_ABBREVIATION',
                how='left',
                suffixes=('', '_against')
            )
            
            # Calculate impact metrics
            impact_stats = {
                'pts_vs_good_defense': game_logs[game_logs['PTS_against'] < game_logs['PTS_against'].mean()]['PTS'].mean(),
                'pts_vs_bad_defense': game_logs[game_logs['PTS_against'] > game_logs['PTS_against'].mean()]['PTS'].mean(),
                'ast_vs_good_defense': game_logs[game_logs['PTS_against'] < game_logs['PTS_against'].mean()]['AST'].mean(),
                'ast_vs_bad_defense': game_logs[game_logs['PTS_against'] > game_logs['PTS_against'].mean()]['AST'].mean(),
                'reb_vs_good_defense': game_logs[game_logs['PTS_against'] < game_logs['PTS_against'].mean()]['REB'].mean(),
                'reb_vs_bad_defense': game_logs[game_logs['PTS_against'] > game_logs['PTS_against'].mean()]['REB'].mean(),
            }
            
            opponent_impact[season] = impact_stats
        
        self.analysis_results['opponent_impact'] = opponent_impact
        return opponent_impact
    
    def analyze_props_vs_performance(self, historical_data, daily_props):
        """Analyze historical prop lines vs actual performance"""
        props_analysis = {}
        
        for season in historical_data:
            game_logs = historical_data[season]['game_logs']
            # Extract year from season (e.g., '2024-25' -> 2024)
            year = int(season.split('-')[0])
            # Filter props by year
            season_props = daily_props[daily_props['commence_time'].dt.year == year]
            
            # Calculate hit rates and edge metrics
            props_analysis[season] = {
                'points_hit_rate': self._calculate_hit_rate(game_logs['PTS'], season_props, 'player_points'),
                'assists_hit_rate': self._calculate_hit_rate(game_logs['AST'], season_props, 'player_assists'),
                'rebounds_hit_rate': self._calculate_hit_rate(game_logs['REB'], season_props, 'player_rebounds'),
                'line_movement_impact': self._analyze_line_movement(season_props),
                'odds_value_spots': self._identify_value_opportunities(season_props)
            }
        
        return props_analysis
    
    def _calculate_hit_rate(self, actual_values, props_data, prop_type):
        """Calculate over/under hit rates for a specific prop type"""
        hits = 0
        total = 0
        
        for _, prop in props_data[props_data['market'] == prop_type].iterrows():
            actual = actual_values.get(prop['player'], None)
            if actual is not None:
                hits += 1 if actual > prop['line'] else 0
                total += 1
                
        return hits / total if total > 0 else 0
    
    def _analyze_line_movement(self, props_data):
        """Analyze the impact of line movements on prop outcomes"""
        movement_analysis = {
            'avg_movement': 0,
            'movement_direction': 'none',
            'significant_moves': 0
        }
        
        if props_data.empty:
            return movement_analysis
        
        try:
            # Group by player and market to track line movements
            for (player, market), group in props_data.groupby(['player', 'market']):
                if len(group) > 1:
                    movement = group['line'].iloc[-1] - group['line'].iloc[0]
                    movement_analysis['avg_movement'] += abs(movement)
                    movement_analysis['significant_moves'] += 1 if abs(movement) >= 1 else 0
            
            if len(props_data) > 0:
                movement_analysis['avg_movement'] /= len(props_data)
                movement_analysis['movement_direction'] = 'up' if movement_analysis['avg_movement'] > 0 else 'down'
            
        except Exception as e:
            self.logger.error(f"Error analyzing line movement: {str(e)}")
        
        return movement_analysis
    
    def _identify_value_opportunities(self, props_data):
        """Identify potential value opportunities in props"""
        value_spots = {
            'total_opportunities': 0,
            'avg_edge': 0,
            'best_markets': []
        }
        
        if props_data.empty:
            return value_spots
        
        try:
            # Simple value identification based on odds movement
            for market in props_data['market'].unique():
                market_props = props_data[props_data['market'] == market]
                if not market_props.empty:
                    odds_movement = market_props['over_price'].diff().mean()
                    if abs(odds_movement) > 10:  # Significant odds movement
                        value_spots['total_opportunities'] += 1
                        value_spots['best_markets'].append(market)
                        value_spots['avg_edge'] += abs(odds_movement)
            
            if value_spots['total_opportunities'] > 0:
                value_spots['avg_edge'] /= value_spots['total_opportunities']
            
        except Exception as e:
            self.logger.error(f"Error identifying value opportunities: {str(e)}")
        
        return value_spots
    
    def track_line_movement(self, daily_props):
        """Public method to track line movements for props"""
        if daily_props.empty:
            return {}
        
        movements = {}
        for player in daily_props['player'].unique():
            player_props = daily_props[daily_props['player'] == player]
            for market in player_props['market'].unique():
                market_props = player_props[player_props['market'] == market]
                if len(market_props) > 1:
                    movements[f"{player}_{market}"] = {
                        'initial_line': market_props['line'].iloc[0],
                        'current_line': market_props['line'].iloc[-1],
                        'line_movement': market_props['line'].iloc[-1] - market_props['line'].iloc[0],
                        'initial_odds': (market_props['over_price'].iloc[0], market_props['under_price'].iloc[0]),
                        'current_odds': (market_props['over_price'].iloc[-1], market_props['under_price'].iloc[-1])
                    }
        
        return movements
    
    def analyze_props_features(self, features_df, props_data):
        """Analyze props-specific features and their relationships"""
        props_analysis = {}
        
        try:
            # Analyze line movement patterns
            props_analysis['line_movements'] = {
                'avg_movement': features_df['line_movement'].mean() if 'line_movement' in features_df.columns else 0,
                'std_movement': features_df['line_movement'].std() if 'line_movement' in features_df.columns else 0
            }
            
            # Analyze odds patterns
            if 'over_price_avg' in features_df.columns and 'under_price_avg' in features_df.columns:
                props_analysis['odds_patterns'] = {
                    'avg_over_price': features_df['over_price_avg'].mean(),
                    'avg_under_price': features_df['under_price_avg'].mean(),
                    'implied_probability_diff': (1/features_df['over_price_avg'] - 1/features_df['under_price_avg']).mean()
                }
            
            # Visualize line movement distributions
            if 'line_movement' in features_df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(features_df['line_movement'], kde=True)
                plt.title('Line Movement Distribution')
                plt.savefig('analysis/line_movement_distribution.png')
                plt.close()
            
            self.analysis_results['props_features_analysis'] = props_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing props features: {str(e)}")
        
        return props_analysis
    
    def analyze_home_away_splits(self, historical_data):
        """Analyze home/away performance differences"""
        splits_analysis = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            
            # Calculate home/away splits
            game_logs['is_home'] = ~game_logs['MATCHUP'].str.contains('@')
            
            splits = game_logs.groupby(['Player_ID', 'is_home']).agg({
                'PTS': ['mean', 'std'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std']
            }).reset_index()
            
            splits_analysis[season] = splits
        
        self.analysis_results['home_away_splits'] = splits_analysis
        return splits_analysis
    
    def analyze_rest_impact(self, historical_data):
        """Analyze impact of rest days on performance"""
        rest_analysis = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs'].sort_values(['Player_ID', 'GAME_DATE'])
            game_logs['days_rest'] = game_logs.groupby('Player_ID')['GAME_DATE'].diff().dt.days
            
            rest_stats = game_logs.groupby('days_rest').agg({
                'PTS': ['mean', 'std', 'count'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std']
            })
            
            rest_analysis[season] = rest_stats
        
        self.analysis_results['rest_impact'] = rest_analysis
        return rest_analysis
    
    def analyze_enhanced_rest_patterns(self, historical_data):
        """Analyze the enhanced rest patterns and their impact"""
        rest_patterns = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs'].sort_values(['Player_ID', 'GAME_DATE'])
            
            # Calculate rest days
            game_logs['days_rest'] = game_logs.groupby('Player_ID')['GAME_DATE'].diff().dt.days
            
            # Create rest pattern indicators
            game_logs['optimal_rest'] = (game_logs['days_rest'] >= 2) & (game_logs['days_rest'] <= 3)
            game_logs['back_to_back'] = game_logs['days_rest'] == 1
            game_logs['extended_rest'] = game_logs['days_rest'] > 3
            
            # Calculate rest performance ratio
            avg_pts = game_logs.groupby('Player_ID')['PTS'].transform('mean')
            game_logs['rest_performance_ratio'] = game_logs['PTS'] / avg_pts
            
            # Analyze optimal rest performance
            optimal_rest_stats = game_logs[game_logs['optimal_rest']].agg({
                'PTS': ['mean', 'std'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std']
            })
            
            # Analyze back-to-back performance
            b2b_stats = game_logs[game_logs['back_to_back']].agg({
                'PTS': ['mean', 'std'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std']
            })
            
            rest_patterns[season] = {
                'optimal_rest_stats': optimal_rest_stats,
                'back_to_back_stats': b2b_stats,
                'rest_performance_ratio_avg': game_logs['rest_performance_ratio'].mean()
            }
        
        self.analysis_results['enhanced_rest_patterns'] = rest_patterns
        return rest_patterns
    
    def analyze_position_impact(self, historical_data):
        """Analyze performance patterns based on statistical roles"""
        position_analysis = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            
            # Calculate player averages for role classification
            player_avgs = game_logs.groupby('Player_ID').agg({
                'PTS': 'mean',
                'AST': 'mean',
                'REB': 'mean',
                'MIN': 'mean'
            }).reset_index()
            
            # Define roles based on statistical thresholds
            player_avgs['role'] = 'balanced'
            player_avgs.loc[player_avgs['AST'] > player_avgs['AST'].quantile(0.75), 'role'] = 'playmaker'
            player_avgs.loc[player_avgs['PTS'] > player_avgs['PTS'].quantile(0.75), 'role'] = 'scorer'
            player_avgs.loc[player_avgs['REB'] > player_avgs['REB'].quantile(0.75), 'role'] = 'rebounder'
            
            # Try to add position information if available
            if 'PLAYER_NAME' in game_logs.columns:
                game_logs['POS'] = game_logs['PLAYER_NAME'].map(self.position_mapping)
            
            # Merge roles back to game logs
            game_logs = game_logs.merge(
                player_avgs[['Player_ID', 'role']], 
                on='Player_ID', 
                how='left'
            )
            
            # Calculate role-based stats
            role_stats = game_logs.groupby('role').agg({
                'PTS': ['mean', 'std', 'count'],
                'AST': ['mean', 'std'],
                'REB': ['mean', 'std'],
                'MIN': ['mean', 'std']
            })
            
            position_analysis[season] = {
                'role_stats': role_stats,
                'role_distribution': game_logs['role'].value_counts(normalize=True)
            }
        
        self.analysis_results['role_based_impact'] = position_analysis
        return position_analysis
    
    def analyze_consistency_patterns(self, historical_data):
        """Analyze player consistency and performance trends"""
        consistency_analysis = {}
        
        for season, data in historical_data.items():
            game_logs = data['game_logs']
            
            # Calculate consistency metrics if they don't exist
            if 'pts_consistency' not in game_logs.columns:
                game_logs['pts_consistency'] = game_logs.groupby('Player_ID')['PTS'].transform(
                    lambda x: x.std() / x.mean() if x.mean() != 0 else 0
                )
            
            # Calculate performance trend
            game_logs['performance_trend'] = game_logs.groupby('Player_ID')['PTS'].transform(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
            
            consistency_stats = {
                'high_consistency_players': len(game_logs[game_logs['pts_consistency'] < 0.3]['Player_ID'].unique()),
                'volatile_players': len(game_logs[game_logs['pts_consistency'] > 0.7]['Player_ID'].unique()),
                'avg_performance_trend': game_logs['performance_trend'].mean()
            }
            
            consistency_analysis[season] = consistency_stats
        
        self.analysis_results['consistency_patterns'] = consistency_analysis
        return consistency_analysis

# Cross-platform timeout implementation
class TimeoutError(Exception):
    pass

def timeout_handler(signum=None, frame=None):
    raise TimeoutError("Function timed out")

@contextmanager
def timeout(seconds=30):
    """Cross-platform timeout context manager"""
    timer = None
    
    def handle_timeout():
        raise TimeoutError("Function timed out")
    
    try:
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
        else:  # Windows systems
            timer = threading.Timer(seconds, handle_timeout)
            timer.start()
        yield
    finally:
        if hasattr(signal, 'SIGALRM'):  # Unix systems
            signal.alarm(0)
        else:  # Windows systems
            if timer:
                timer.cancel()
