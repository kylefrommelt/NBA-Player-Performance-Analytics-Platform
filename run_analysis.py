import pandas as pd
import os
from data_analyzer import DataAnalyzer
from feature_engineering import NBAFeatureEngineer
from predictor import PropsPredictor
from tqdm import tqdm
import logging
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

# Set up logging at module level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_predictions(feature_sets: Dict[str, pd.DataFrame], val_predictions: Dict[str, Dict]) -> None:
    """Evaluate predictions with enhanced metrics and logging"""
    try:
        cv_metrics = {}
        
        for target in val_predictions:
            predictions = val_predictions[target]['prediction']
            actual = feature_sets['validation'][target]
            
            # Ensure we have valid predictions and actuals
            if len(predictions) == 0 or len(actual) == 0:
                logger.warning(f"No predictions or actuals for {target}")
                continue
                
            # Basic metrics
            metrics = {
                'all_predictions': {
                    'mae': float(mean_absolute_error(actual, predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(actual, predictions))),
                    'r2': float(r2_score(actual, predictions)),
                    'count': len(predictions)
                }
            }
            
            # Add confidence-based metrics
            confidence = val_predictions[target]['confidence']
            for conf_level in ['High', 'Medium', 'Low']:
                mask = confidence == conf_level
                if any(mask):
                    metrics[conf_level.lower()] = {
                        'mae': float(mean_absolute_error(actual[mask], predictions[mask])),
                        'rmse': float(np.sqrt(mean_squared_error(actual[mask], predictions[mask]))),
                        'r2': float(r2_score(actual[mask], predictions[mask])),
                        'count': int(sum(mask))
                    }
            
            cv_metrics[target] = metrics
            
        return cv_metrics

    except Exception as e:
        logger.error(f"Error in evaluate_predictions: {str(e)}")
        raise

def write_prediction_results(prediction_results: Dict, prediction_file: str, daily_props_df: pd.DataFrame = None) -> None:
    """Write prediction results with detailed betting analysis"""
    try:
        with open(prediction_file, 'w', encoding='utf-8') as f:
            f.write("=== NBA Props Prediction Analysis ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write detailed validation metrics
            f.write("=== Model Validation Metrics ===\n")
            for target, metrics in prediction_results['validation_metrics'].items():
                f.write(f"\n{target} Model Performance:\n")
                f.write("-" * 40 + "\n")
                
                # Overall metrics
                if 'all_predictions' in metrics:
                    all_preds = metrics['all_predictions']
                    f.write("Overall Metrics:\n")
                    f.write(f"MAE: {all_preds['mae']:.2f}\n")
                    f.write(f"RMSE: {all_preds['rmse']:.2f}\n")
                    f.write(f"R²: {all_preds['r2']:.3f}\n")
                    f.write(f"Sample Size: {all_preds['count']}\n")
                
                # Metrics by confidence level
                for conf_level in ['High', 'Medium', 'Low']:
                    if conf_level.lower() in metrics:
                        conf_metrics = metrics[conf_level.lower()]
                        f.write(f"\n{conf_level} Confidence Predictions:\n")
                        
                        count = conf_metrics.get('count')
                        f.write(f"Count: {count if count is not None else 'N/A'}\n")
                        
                        mae = conf_metrics.get('mae')
                        f.write(f"MAE: {f'{mae:.2f}' if mae is not None else 'N/A'}\n")
                        
                        rmse = conf_metrics.get('rmse')
                        f.write(f"RMSE: {f'{rmse:.2f}' if rmse is not None else 'N/A'}\n")
                        
                        r2 = conf_metrics.get('r2')
                        f.write(f"R²: {f'{r2:.3f}' if r2 is not None else 'N/A'}\n")
                        
                        avg_error = conf_metrics.get('avg_error')
                        error_std = conf_metrics.get('error_std')
                        if avg_error is not None and error_std is not None:
                            f.write(f"Average Error: {avg_error:.2f} ± {error_std:.2f}\n")
                        else:
                            f.write("Average Error: N/A\n")
                        
                        # Confidence interval coverage
                        if 'ci_coverage' in conf_metrics:
                            f.write(f"Confidence Interval Coverage: {conf_metrics['ci_coverage']:.1%}\n")
                        
                        # Historical betting performance if available
                        if 'betting_performance' in conf_metrics:
                            bet_perf = conf_metrics['betting_performance']
                            f.write("\nHistorical Betting Performance:\n")
                            f.write(f"Win Rate: {bet_perf['win_rate']:.1%}\n")
                            f.write(f"ROI: {bet_perf['roi']:.1%}\n")
                            f.write(f"Sample Size: {bet_perf['sample_size']} bets\n")
                        
                        # Add win rate metrics
                        if 'all_predictions' in conf_metrics:
                            all_preds = conf_metrics['all_predictions']
                            if 'win_rate_0.5' in all_preds:
                                f.write("\nWin Rates:\n")
                                f.write(f"Within 0.5: {all_preds['win_rate_0.5']:.1%}\n")
                                f.write(f"Within 1.0: {all_preds['win_rate_1.0']:.1%}\n")
                                f.write(f"Within 2.0: {all_preds['win_rate_2.0']:.1%}\n")
                            
                            if 'prediction_bias' in all_preds:
                                f.write(f"\nPrediction Bias: {all_preds['prediction_bias']:.2f}\n")
                            
                            if 'extreme_pred_rate' in all_preds:
                                f.write(f"Extreme Prediction Rate: {all_preds['extreme_pred_rate']:.1%}\n")
                        
                        # Add trend analysis
                        if 'prediction_trends' in conf_metrics:
                            trends = conf_metrics['prediction_trends']
                            f.write("\nPrediction Trends:\n")
                            f.write(f"Recent Accuracy: {trends['recent_accuracy']:.1%}\n")
                            f.write(f"Accuracy Trend: {trends['accuracy_trend']:.1%}\n")
                            f.write(f"Market Efficiency: {trends['market_efficiency']:.2f}\n")
                        
                        # Add player-specific metrics
                        if 'player_metrics' in conf_metrics:
                            f.write("\nPlayer-Specific Performance:\n")
                            for player, player_metrics in conf_metrics['player_metrics'].items():
                                f.write(f"\n{player}:")
                                f.write(f"\n  Accuracy: {player_metrics['accuracy']:.1%}")
                                f.write(f"\n  Sample Size: {player_metrics['sample_size']}")
                                f.write(f"\n  Consistency: {player_metrics['consistency']:.2f}")
                        
                        f.write("\n")

            # Write daily predictions and betting analysis
            if 'daily' in prediction_results['predictions'] and daily_props_df is not None:
                f.write("\n=== Today's Predictions & Betting Analysis ===\n")
                
                # Market type mapping
                market_map = {
                    'PTS': 'player_points',
                    'AST': 'player_assists',
                    'REB': 'player_rebounds',
                    'PTS_AST': 'player_points_assists',
                    'PTS_REB': 'player_points_rebounds',
                    'PTS_AST_REB': 'player_points_rebounds_assists'
                }
                
                # Group predictions by player
                player_predictions = {}
                
                for target, preds in prediction_results['predictions']['daily'].items():
                    market = market_map.get(target)
                    if not market:
                        continue
                    
                    # Safely get prediction values with defaults
                    player_names = preds.get('player_names', [])
                    predictions = preds.get('prediction', [])
                    confidences = preds.get('confidence', [])
                    lower_bounds = preds.get('lower_bound', [np.nan] * len(player_names))
                    upper_bounds = preds.get('upper_bound', [np.nan] * len(player_names))
                    
                    for player_name, pred, conf, lower, upper in zip(
                        player_names,
                        predictions,
                        confidences,
                        lower_bounds,
                        upper_bounds
                    ):
                        if player_name not in player_predictions:
                            player_predictions[player_name] = []
                            
                        # Find matching prop
                        props = daily_props_df[
                            (daily_props_df['market'] == market) & 
                            (daily_props_df['player'] == player_name)
                        ]
                        
                        if not props.empty:
                            prop = props.iloc[0]
                            edge = pred - prop['line']
                            
                            pred_info = {
                                'market': target,
                                'prediction': pred,
                                'confidence': conf,
                                'line': prop['line'],
                                'edge': edge,
                                'over_price': prop.get('over_price'),
                                'under_price': prop.get('under_price')
                            }
                            
                            # Only add bounds if they exist
                            if not np.isnan(lower):
                                pred_info['lower_bound'] = lower
                            if not np.isnan(upper):
                                pred_info['upper_bound'] = upper
                            
                            player_predictions[player_name].append(pred_info)
                
                # Write predictions grouped by player
                for player, predictions in player_predictions.items():
                    f.write(f"\n=== {player} ===\n")
                    for pred in predictions:
                        f.write(f"\n{pred['market']}:")
                        f.write(f"\nPrediction: {pred['prediction']:.1f} ({pred['confidence']} confidence)")
                        
                        # Add prediction context
                        if 'lower_bound' in pred and 'upper_bound' in pred:
                            f.write(f"\nRange: [{pred['lower_bound']:.1f} - {pred['upper_bound']:.1f}]")
                            range_width = pred['upper_bound'] - pred['lower_bound']
                            f.write(f"\nPrediction Range: {range_width:.1f} units")
                        
                        f.write(f"\nLine: {pred['line']}")
                        f.write(f"\nEdge: {pred['edge']:.1f}")
                        
                        # Enhanced bet recommendation without emoji
                        if abs(pred['edge']) >= 1.0:
                            bet_type = "OVER" if pred['edge'] > 0 else "UNDER"
                            price = pred['over_price'] if pred['edge'] > 0 else pred['under_price']
                            confidence_note = "[LOW CONFIDENCE]" if pred['confidence'] == 'Low' else "[HIGH CONFIDENCE]"
                            f.write(f"\n{confidence_note} BET: {bet_type} {pred['line']} ({price})")
                        f.write("\n")
                
                # Summary of best bets
                f.write("\n=== Best Bets Summary ===\n")
                best_bets = []
                for player, predictions in player_predictions.items():
                    for pred in predictions:
                        if pred['confidence'] == 'High' and abs(pred['edge']) >= 1.0:
                            best_bets.append({
                                'player': player,
                                'market': pred['market'],
                                'edge': pred['edge'],
                                'line': pred['line'],
                                'bet_type': "OVER" if pred['edge'] > 0 else "UNDER",
                                'price': pred['over_price'] if pred['edge'] > 0 else pred['under_price'],
                                'confidence': pred['confidence']
                            })
                
                if best_bets:
                    best_bets.sort(key=lambda x: abs(x['edge']), reverse=True)
                    f.write("\n=== Best Bets Summary ===\n")
                    f.write("\nTop Recommendations (sorted by edge):\n")
                    for bet in best_bets:
                        f.write(f"\n{bet['player']} {bet['market']}")
                        f.write(f"\n  {bet['bet_type']} {bet['line']} (Price: {bet['price']})")
                        f.write(f"\n  Edge: {bet['edge']:.1f}")
                        f.write(f"\n  Confidence: {bet['confidence']}")
                        if 'historical_win_rate' in bet:
                            f.write(f"\n  Historical Win Rate: {bet['historical_win_rate']:.1%}")
                        f.write("\n")
                else:
                    f.write("\nNo high-confidence bets recommended for today.")
                    
            print(f"\nPrediction results saved to {prediction_file}")
            
    except Exception as e:
        logger.error(f"Error writing prediction results: {str(e)}")
        raise

def add_combined_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced combined target columns to dataframe"""
    df = df.copy()
    
    # Existing combinations
    if all(col in df.columns for col in ['PTS', 'AST', 'REB']):
        df['PTS_REB'] = df['PTS'] + df['REB']
        df['PTS_AST'] = df['PTS'] + df['AST']
        df['PTS_AST_REB'] = df['PTS'] + df['AST'] + df['REB']
        
        # Add rolling averages
        for col in ['PTS', 'AST', 'REB', 'PTS_REB', 'PTS_AST', 'PTS_AST_REB']:
            df[f'{col}_rolling_3'] = df.groupby('PLAYER_NAME')[col].rolling(3).mean().reset_index(0, drop=True)
            df[f'{col}_rolling_5'] = df.groupby('PLAYER_NAME')[col].rolling(5).mean().reset_index(0, drop=True)
            
        # Add variance metrics
        df['performance_volatility'] = df.groupby('PLAYER_NAME')['PTS'].rolling(5).std().reset_index(0, drop=True)
        
    return df

def preprocess_historical_data(historical_data: Dict) -> Dict:
    """Preprocess historical data with standardized date format"""
    try:
        for season in historical_data:
            if 'game_logs' in historical_data[season]:
                # Specify format explicitly for date parsing
                historical_data[season]['game_logs']['GAME_DATE'] = pd.to_datetime(
                    historical_data[season]['game_logs']['GAME_DATE'],
                    format='%Y-%m-%d',  # Use this if dates are like '2023-01-30'
                    # or use format='%m/%d/%Y' if dates are like '01/30/2023'
                    errors='coerce'  # Handle any unexpected formats gracefully
                )
                
                # Fill any NaT (Not a Time) values with appropriate fallback
                mask = historical_data[season]['game_logs']['GAME_DATE'].isna()
                if mask.any():
                    logger.warning(f"Found {mask.sum()} invalid dates in season {season}")
                    # Use the previous valid date, or first valid date if at start
                    historical_data[season]['game_logs']['GAME_DATE'].fillna(
                        method='ffill', inplace=True
                    )
                    historical_data[season]['game_logs']['GAME_DATE'].fillna(
                        method='bfill', inplace=True
                    )

        return historical_data
        
    except Exception as e:
        logger.error(f"Error preprocessing historical data: {str(e)}")
        raise

def analyze_current_state(feature_sets, historical_data):
    """Analyze current state of data and predictions"""
    print("\n=== Data Analysis ===")
    
    # Sample size analysis
    print(f"Training samples: {len(feature_sets['train'])}")
    print(f"Validation samples: {len(feature_sets['validation'])}")
    print(f"Total historical games: {len(historical_data)}")
    
    # Target distribution
    for target in ['PTS', 'AST', 'REB']:
        data = historical_data[target].dropna()
        print(f"\n{target} Distribution:")
        print(f"Mean: {data.mean():.2f}")
        print(f"Median: {data.median():.2f}")
        print(f"Std: {data.std():.2f}")
        print(f"Missing values: {historical_data[target].isna().sum()}")
    
    # Feature completeness
    missing_stats = historical_data.isna().sum()
    if len(missing_stats[missing_stats > 0]) > 0:
        print("\nMissing Data in Features:")
        print(missing_stats[missing_stats > 0])

def enhance_features(df):
    """Add high-impact features"""
    df = df.copy()
    
    # Last N games rolling stats (3, 5, 10 games)
    for col in ['PTS', 'AST', 'REB']:
        for n in [3, 5, 10]:
            df[f'{col}_last_{n}'] = df.groupby('PLAYER_NAME')[col].transform(
                lambda x: x.rolling(n, min_periods=1).mean()
            )
    
    # Home/Away performance
    df['is_home'] = df['MATCHUP'].str.contains('@').map({True: 0, False: 1})
    for col in ['PTS', 'AST', 'REB']:
        df[f'{col}_home_avg'] = df.groupby(['PLAYER_NAME', 'is_home'])[col].transform('mean')
    
    # Days rest impact
    df['days_rest'] = df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days.fillna(2)
    
    # Opponent strength
    df['opp_pts_allowed'] = df.groupby('OPPONENT')['PTS'].transform('mean')
    
    return df

def create_improved_pipeline():
    """Create improved model pipeline"""
    # Define feature types
    numerical_features = ['PTS', 'AST', 'REB', 'MIN', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
                         'pts_consistency', 'ast_consistency', 'reb_consistency',
                         'avg_FG_PCT_mean', 'days_rest', 'is_home']
    
    categorical_features = ['MATCHUP', 'SEASON']
    
    return Pipeline([
        ('preprocessing', ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
        ])),
        ('model', xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

def validate_predictions(predictions, actuals, player_names):
    """Validate predictions with detailed analysis"""
    results = {}
    
    # Overall metrics
    results['overall'] = {
        'mae': mean_absolute_error(actuals, predictions),
        'rmse': np.sqrt(mean_squared_error(actuals, predictions)),
        'r2': r2_score(actuals, predictions)
    }
    
    # Player-specific analysis
    player_metrics = {}
    for player in np.unique(player_names):
        mask = player_names == player
        if sum(mask) >= 5:  # Minimum 5 games
            player_metrics[player] = {
                'mae': mean_absolute_error(actuals[mask], predictions[mask]),
                'bias': np.mean(predictions[mask] - actuals[mask]),
                'games': sum(mask)
            }
    
    results['player_metrics'] = player_metrics
    return results

def cross_validate_model(X, y, player_names, n_splits=5):
    """Time-based cross validation"""
    cv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = create_improved_pipeline()
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        val_results = validate_predictions(
            preds, y_val, player_names[val_idx]
        )
        results.append(val_results)
    
    return results

def analyze_market_inefficiencies(historical_data: Dict, f) -> None:
    """Analyze market inefficiencies and betting opportunities"""
    try:
        if 'player_stats' not in historical_data:
            raise KeyError("Missing player stats data")
            
        df = historical_data['player_stats']
        
        # Ensure required columns exist
        required_cols = ['days_rest', 'PTS', 'AST', 'REB']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise KeyError(f"Missing required columns: {missing}")
            
        # Analyze rest impact
        rest_analysis = df.groupby('days_rest')[['PTS', 'AST', 'REB']].agg(['mean', 'std'])
        
        f.write("\nRest Days Analysis:\n")
        for stat in ['PTS', 'AST', 'REB']:
            f.write(f"\n{stat} by Rest Days:\n")
            for days in sorted(df['days_rest'].unique()):
                mean = rest_analysis.loc[days, (stat, 'mean')]
                std = rest_analysis.loc[days, (stat, 'std')]
                f.write(f"- {days} days: {mean:.1f} ± {std:.1f}\n")
                
    except Exception as e:
        f.write(f"\nError in market inefficiency analysis: {str(e)}\n")

def analyze_player_consistency(historical_data: Dict, f) -> None:
    """Analyze player consistency metrics"""
    try:
        if 'player_stats' not in historical_data:
            raise KeyError("Missing player stats data")
            
        df = historical_data['player_stats']
        min_games = 20
        
        # Calculate consistency metrics
        player_stats = []
        for stat in ['PTS', 'AST', 'REB']:
            # Group by player and calculate stats
            grouped = df.groupby('PLAYER_NAME')[stat].agg(['mean', 'std', 'count'])
            grouped = grouped[grouped['count'] >= min_games]
            grouped['cv'] = grouped['std'] / grouped['mean']
            grouped = grouped.sort_values('cv')
            
            # Get top 3 most consistent players
            top_3 = grouped.head(3)
            
            f.write(f"\nMost Consistent {stat} Producers (min {min_games} games):\n")
            for idx, (name, row) in enumerate(top_3.iterrows(), 1):
                f.write(f"{idx}. {name}: {row['mean']:.1f} ± {row['std']:.1f} ({row['cv']:.3f} CV)\n")
                
    except Exception as e:
        f.write(f"\nError in player consistency analysis: {str(e)}\n")

def analyze_situations(historical_data: Dict, f) -> None:
    """Analyze performance in different situations"""
    try:
        if 'player_stats' not in historical_data:
            raise KeyError("Missing player stats data")
            
        df = historical_data['player_stats']
        
        f.write("\nSituational Performance Analysis:\n")
        
        # Home vs Away Analysis
        f.write("\nHome vs Away Impact:\n")
        for stat in ['PTS', 'AST', 'REB']:
            f.write(f"\n{stat}:")
            home_stats = df.groupby('is_home')[stat].mean()
            for is_home in [True, False]:
                f.write(f"\n- {is_home}: {home_stats[is_home]:.1f}")
                
    except Exception as e:
        f.write(f"\nError in situational analysis: {str(e)}\n")

def generate_analytics_report(historical_data: Dict, feature_sets: Dict, prediction_results: Dict) -> None:
    """Generate comprehensive analytics report"""
    try:
        report_file = 'analysis/analytics_report.txt'
        print(f"\nGenerating analytics report to {report_file}...")
        
        with open(report_file, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
            f.write("=== NBA Props Analytics Report ===\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Performance Analysis
            f.write("=== Model Performance Analysis ===\n\n")
            for target in ['PTS', 'AST', 'REB', 'PTS_REB', 'PTS_AST', 'PTS_AST_REB']:
                metrics = prediction_results.get('validation_metrics', {}).get(target, {}).get('all_predictions', {})
                f.write(f"{target} Model Insights:\n")
                mae = metrics.get('mae')
                rmse = metrics.get('rmse')
                r2 = metrics.get('r2')
                f.write(f"- MAE: {f'{mae:.2f}' if mae is not None else 'N/A'}\n")
                f.write(f"- RMSE: {f'{rmse:.2f}' if rmse is not None else 'N/A'}\n")
                f.write(f"- R²: {f'{r2:.3f}' if r2 is not None else 'N/A'}\n\n")
            
            # Feature Analysis
            f.write("=== Key Feature Analysis ===\n")
            for target in ['PTS', 'AST', 'REB']:
                feature_importance = analyze_feature_importance(feature_sets['train'], target)
                f.write(f"\nTop 10 Features for {target}:\n")
                for feat, imp in feature_importance[:10]:
                    f.write(f"- {feat}: {imp:.3f}\n")
            
            # Other analyses
            f.write("\n=== Market Inefficiency Analysis ===\n")
            analyze_market_inefficiencies(historical_data, f)
            
            f.write("\n=== Player Consistency Analysis ===\n")
            analyze_player_consistency(historical_data, f)
            
            f.write("\n=== Situational Analysis ===\n")
            analyze_situations(historical_data, f)
            
            # Add recommendations section
            generate_recommendations(prediction_results, f)
            
    except Exception as e:
        logger.error(f"Error generating analytics report: {str(e)}")
        raise

def analyze_feature_importance(data: pd.DataFrame, target: str) -> List[Tuple[str, float]]:
    """Analyze feature importance for a given target"""
    try:
        # Create XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Prepare features (exclude target and non-numeric columns)
        X = data.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
        y = data[target]
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importance
        importance = list(zip(X.columns, model.feature_importances_))
        return sorted(importance, key=lambda x: x[1], reverse=True)
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        return []

def generate_recommendations(prediction_results: Dict, f) -> None:
    """Generate specific recommendations for improvement"""
    try:
        metrics = prediction_results.get('validation_metrics', {})
        
        f.write("\nKey Recommendations:\n")
        
        # Analyze prediction accuracy patterns
        for target in ['PTS', 'AST', 'REB']:
            f.write(f"\n1. {target} Predictions:\n")
            f.write("- Consider adding more recent form indicators\n")
            f.write("- Analyze matchup-specific features\n")
            f.write("- Review outlier handling in preprocessing\n")
        
        # Add general recommendations
        f.write("\n2. Confidence Calibration:\n")
        f.write("- Adjust thresholds for high confidence predictions\n")
        f.write("- Implement dynamic confidence based on sample size\n")
        f.write("- Consider player-specific confidence adjustments\n")
        
        f.write("\n3. Feature Engineering:\n")
        f.write("- Add defensive matchup quality metrics\n")
        f.write("- Incorporate team rotation patterns\n")
        f.write("- Consider adding player rest impact features\n")
        
        f.write("\n4. Model Improvements:\n")
        f.write("- Experiment with ensemble methods\n")
        f.write("- Implement time-based cross validation\n")
        f.write("- Add recency weighting to training data\n")
        
    except Exception as e:
        f.write(f"\nError generating recommendations: {str(e)}\n")

def main():
    # Create analysis directory if it doesn't exist
    os.makedirs('analysis', exist_ok=True)
    
    print("Loading data...")
    # Load historical data
    historical_data = {}
    
    # First load player names from player_stats.csv
    try:
        players_df = pd.read_csv('data/raw/player_stats.csv')
        # Extract just the player ID and name columns we need
        players_df = players_df[['PLAYER_ID', 'PLAYER_NAME_x']].rename(columns={'PLAYER_NAME_x': 'PLAYER_NAME'})
        players_df = players_df.drop_duplicates()
        print("Loaded players database")
    except Exception as e:
        print(f"Warning: Could not load players database: {e}")
        players_df = None
    
    # Show progress bar for loading seasons
    seasons = ['2024-25', '2023-24', '2022-23']
    for season in tqdm(seasons, desc="Loading seasons"):
        print(f"\nLoading {season} data...")
        game_logs_path = f'data/historical/game_logs_{season}.csv'
        matchups_path = f'data/historical/matchups_{season}.csv'
        
        if os.path.exists(game_logs_path) and os.path.exists(matchups_path):
            game_logs = pd.read_csv(game_logs_path)
            
            # Add player names if we have them
            if players_df is not None:
                game_logs = game_logs.merge(
                    players_df,
                    left_on='Player_ID',
                    right_on='PLAYER_ID',
                    how='left'
                )
            
            historical_data[season] = {
                'game_logs': game_logs,
                'matchups': pd.read_csv(matchups_path)
            }
            print(f"Loaded {season} data successfully")
        else:
            print(f"Warning: Missing data files for {season}")
    
    print("\nPreprocessing data...")
    for season in tqdm(historical_data, desc="Preprocessing seasons"):
        # Convert GAME_DATE to datetime
        historical_data[season]['game_logs']['GAME_DATE'] = pd.to_datetime(historical_data[season]['game_logs']['GAME_DATE'])
        historical_data[season]['matchups']['GAME_DATE'] = pd.to_datetime(historical_data[season]['matchups']['GAME_DATE'])
        
        # Fill NaN values with 0 for key stats
        stats_to_fill = ['MIN', 'PTS', 'AST', 'REB', 'PLUS_MINUS']
        historical_data[season]['game_logs'][stats_to_fill] = historical_data[season]['game_logs'][stats_to_fill].fillna(0)
        
        # Remove any games where minutes played is 0 (DNPs)
        historical_data[season]['game_logs'] = historical_data[season]['game_logs'][
            historical_data[season]['game_logs']['MIN'] > 0
        ]
        
        # Sort by date and player ID
        historical_data[season]['game_logs'] = historical_data[season]['game_logs'].sort_values(
            ['Player_ID', 'GAME_DATE']
        )
    
    print("\nLoading current data...")
    current_data = {}
    for data_type in tqdm(['injury_report', 'team_clutch_stats', 'lineup_stats', 'shooting_stats'], desc="Loading current data"):
        file_path = f'data/raw/{data_type}.csv'
        if os.path.exists(file_path):
            current_data[data_type] = pd.read_csv(file_path)
            print(f"Loaded {data_type}")
        else:
            print(f"Warning: Missing {data_type} file")
            current_data[data_type] = pd.DataFrame()  # Empty DataFrame as fallback
    
    print("\nCreating features...")
    feature_engineer = NBAFeatureEngineer()
    
    # Load daily props if available
    try:
        daily_props = pd.read_csv('data/raw/daily_props.csv') if os.path.exists('data/raw/daily_props.csv') else None
        if daily_props is not None:
            print("Loaded daily props data")
    except Exception as e:
        print(f"Warning: Error loading daily props: {e}")
        daily_props = None
    
    try:
        # Get engineered features with train/val/test split
        feature_sets = feature_engineer.create_features(
            historical_data=historical_data,
            current_data=current_data,
            props_data=daily_props
        )
        
        if feature_sets is None:
            raise ValueError("Feature engineering failed to produce valid feature sets")
        
        print("\nAnalyzing data...")
        # Load position mapping from CSV
        try:
            positions_df = pd.read_csv('data/raw/player_positions.csv')
            position_mapping = dict(zip(positions_df['PLAYER_NAME'], positions_df['POSITION']))
            print("Loaded position mappings for", len(position_mapping), "players")
        except Exception as e:
            print(f"Warning: Could not load position mappings: {e}")
            position_mapping = {}
        
        analyzer = DataAnalyzer(position_mapping=position_mapping)
        
        # Run analysis steps with progress bar and timeout
        analysis_steps = [
            ("Checking data quality", lambda: analyzer.check_data_quality(feature_sets['all_features'])),
            ("Analyzing feature importance", lambda: analyzer.analyze_feature_importance(
                feature_sets['all_features'], 
                market_types=['PTS', 'AST', 'REB']
            )),
            ("Analyzing player trends", lambda: analyzer.analyze_player_trends(historical_data)),
            ("Analyzing rest patterns", lambda: analyzer.analyze_enhanced_rest_patterns(historical_data)),
            ("Analyzing home/away splits", lambda: analyzer.analyze_home_away_splits(historical_data)),
            ("Analyzing props features", lambda: analyzer.analyze_props_features(feature_sets['all_features'], daily_props))
        ]
        
        for desc, func in tqdm(analysis_steps, desc="Running analysis"):
            print(f"\n{desc}...")
            try:
                func()
            except Exception as e:
                print(f"Warning: {desc} failed: {str(e)}")
                continue

        print("\nInitializing prediction pipeline...")
        predictor = PropsPredictor()
        
        # Training models
        print("Training prediction models...")
        try:
            print("\nPreparing training data...")
            feature_sets['train'] = add_combined_targets(feature_sets['train'])
            feature_sets['validation'] = add_combined_targets(feature_sets['validation'])
            feature_sets['all_features'] = add_combined_targets(feature_sets['all_features'])
            
            predictor.train(feature_sets['train'])
            
            # Make predictions on validation set
            val_predictions = predictor.predict(feature_sets['validation'])
            
            # Evaluate predictions and store the metrics
            cv_metrics = evaluate_predictions(feature_sets, val_predictions)
            
            # Save predictions and metrics
            prediction_results = {
                'validation_metrics': cv_metrics,  # Store the detailed metrics from evaluate_predictions
                'predictions': {}
            }
            
            # Calculate and save metrics for each target
            for target in ['PTS', 'AST', 'REB', 'PTS_REB', 'PTS_AST', 'PTS_AST_REB']:
                if target in val_predictions:
                    pred_dict = val_predictions[target]
                    predictions = pred_dict['prediction']
                    confidence = pred_dict['confidence']
                    lower_bound = pred_dict.get('lower_bound', None)
                    upper_bound = pred_dict.get('upper_bound', None)
                    
                    # Store the prediction data
                    prediction_results['predictions'][target] = {
                        'prediction': predictions,
                        'confidence': confidence,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
            
            # If we have daily props, make predictions for them
            if daily_props is not None:
                print("\nMaking predictions for daily props...")
                daily_predictions = predictor.predict(feature_sets['all_features'])
                
                # Get player names from daily props
                player_names = daily_props['player'].unique()
                
                # Save predictions with confidence levels and actual player names
                prediction_results['predictions']['daily'] = {}
                for target in ['PTS', 'AST', 'REB', 'PTS_REB', 'PTS_AST', 'PTS_AST_REB']:
                    if target in daily_predictions:
                        prediction_results['predictions']['daily'][target] = {
                            'prediction': daily_predictions[target]['prediction'],
                            'confidence': daily_predictions[target]['confidence'],
                            'lower_bound': daily_predictions[target].get('lower_bound', None),
                            'upper_bound': daily_predictions[target].get('upper_bound', None),
                            'player_names': player_names
                        }
                
                # Validate predictions before saving
                for target in ['PTS', 'AST', 'REB', 'PTS_REB', 'PTS_AST', 'PTS_AST_REB']:
                    if target in daily_predictions:
                        pred = daily_predictions[target]['prediction']
                        if np.any(pred < 0):  # Check for negative predictions
                            logger.warning(f"Found negative predictions for {target}")
                            daily_predictions[target]['prediction'] = np.maximum(pred, 0)
                        
                        # Validate confidence levels
                        conf = daily_predictions[target]['confidence']
                        if not all(c in ['High', 'Medium', 'Low'] for c in conf):
                            logger.error(f"Invalid confidence levels for {target}")
            
            # Save prediction results
            print("\nSaving prediction results...")
            prediction_file = 'analysis/prediction_results.txt'
            write_prediction_results(prediction_results, prediction_file, daily_props)
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise
        
        print("\nGenerating analytics report...")
        generate_analytics_report(
            historical_data=historical_data,
            feature_sets=feature_sets,
            prediction_results=prediction_results
        )
        
        print("\nAnalysis and predictions complete! Check the 'analysis' directory for results.")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
