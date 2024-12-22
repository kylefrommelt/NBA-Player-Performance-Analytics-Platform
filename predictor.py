import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from feature_selector import FeatureSelector
from model_pipeline import PropsPredictionPipeline
from consistency_tracker import PlayerConsistencyTracker
from data_analyzer import DataAnalyzer

class PropsPredictor:
    def __init__(self):
        self.feature_selector = FeatureSelector()
        self.consistency_tracker = PlayerConsistencyTracker()
        self.models = {}
        self.logger = logging.getLogger(__name__)
        self.feature_importance = {}
        self.player_names = {}  # Add player name tracking
        
        # Initialize models for all targets including combinations
        for target in ['PTS', 'AST', 'REB', 'PTS_REB', 'PTS_AST', 'PTS_AST_REB']:
            self.models[target] = PropsPredictionPipeline(target)
            
    def _normalize_feature(self, series: pd.Series) -> pd.Series:
        """Normalize a feature to a reasonable range (-1 to 1)"""
        try:
            if series.std() == 0:
                return series * 0
            
            # For percentage features, ensure they're in [0,1]
            if any(x in series.name.lower() for x in ['pct', 'percentage', '_pct']):
                return series.clip(0, 1)
            
            # For consistency scores, keep in [0,1]
            if 'consistency' in series.name:
                return series.clip(0, 1)
            
            # For binary features, ensure 0/1
            if series.name in ['home_away', 'back_to_back', 'optimal_rest']:
                return series.astype(float).clip(0, 1)
            
            # For minutes, use a tighter clip range (-2, 2)
            if 'MIN' in series.name:
                median = series.median()
                q75, q25 = series.quantile(0.75), series.quantile(0.25)
                iqr = q75 - q25 if q75 > q25 else series.std()
                return ((series - median) / (iqr + 1e-8)).clip(-2, 2)
            
            # For recent form features, use slightly wider range (-2.5, 2.5)
            if '_recent_form' in series.name:
                median = series.median()
                q75, q25 = series.quantile(0.75), series.quantile(0.25)
                iqr = q75 - q25 if q75 > q25 else series.std()
                return ((series - median) / (iqr + 1e-8)).clip(-2.5, 2.5)
            
            # For other features, use robust scaling with adjusted clipping
            median = series.median()
            q75, q25 = series.quantile(0.75), series.quantile(0.25)
            iqr = q75 - q25 if q75 > q25 else series.std()
            
            if iqr == 0:
                return series * 0
            
            return ((series - median) / (iqr + 1e-8)).clip(-3, 3)
            
        except Exception as e:
            self.logger.error(f"Error normalizing feature {series.name}: {str(e)}")
            return series * 0
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize data with enhanced outlier handling"""
        try:
            df = df.copy()
            
            # Columns that should not be normalized
            do_not_normalize = [
                'Player_ID', 'PLAYER_ID', 'SEASON_ID', 'Game_ID',  # IDs
                'player_role', 'position'  # Categorical features
            ]
            
            # Identify numeric columns that should be normalized
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            normalize_cols = [col for col in numeric_cols if col not in do_not_normalize]
            
            # Handle numeric columns
            for col in normalize_cols:
                try:
                    # Replace infinities with NaN
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Remove extreme outliers using IQR method
                    q75 = df[col].quantile(0.75)
                    q25 = df[col].quantile(0.25)
                    iqr = q75 - q25
                    upper_bound = q75 + 2.5 * iqr
                    lower_bound = q25 - 2.5 * iqr
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    
                    # Normalize the feature
                    df[col] = self._normalize_feature(df[col])
                    
                except Exception as e:
                    self.logger.warning(f"Error cleaning column {col}: {str(e)}")
                    continue
            
            # Fill NaN values with appropriate defaults
            for col in numeric_cols:
                if col in ['MIN', 'days_rest']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)
            
            # Log summary statistics for key features
            key_features = ['PTS', 'AST', 'REB', 'MIN', 'FG_PCT', 'pts_consistency']
            self.logger.info("\nKey Feature Statistics:")
            for feat in key_features:
                if feat in df.columns:
                    stats = df[feat].describe()
                    self.logger.info(f"\n{feat}:\n{stats}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def prepare_target_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare combined target variables"""
        data = data.copy()
        
        # Create combined targets if base targets exist
        if all(target in data.columns for target in ['PTS', 'REB']):
            data['PTS_REB'] = data['PTS'] + data['REB']
        
        if all(target in data.columns for target in ['PTS', 'AST']):
            data['PTS_AST'] = data['PTS'] + data['AST']
        
        if all(target in data.columns for target in ['PTS', 'AST', 'REB']):
            data['PTS_AST_REB'] = data['PTS'] + data['AST'] + data['REB']
        
        # Log available targets
        self.logger.info(f"Available targets: {[col for col in data.columns if col in self.models]}")
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data before processing"""
        try:
            # Check for required columns
            required_cols = ['PTS', 'AST', 'REB', 'MIN', 'FG_PCT', 'FGA', 'FGM']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for sufficient data
            if len(data) < 1000:  # Arbitrary minimum size
                self.logger.warning(f"Small dataset size: {len(data)} rows")
            
            # Check for NaN values
            nan_cols = data.columns[data.isna().any()].tolist()
            if nan_cols:
                self.logger.warning(f"NaN values found in columns: {nan_cols}")
            
            # Check data types
            non_numeric = data.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                self.logger.warning(f"Non-numeric columns found: {non_numeric}")
            
            # Log data summary
            self.logger.info("\nData Validation Summary:")
            self.logger.info(f"Total rows: {len(data)}")
            self.logger.info(f"Total columns: {len(data.columns)}")
            self.logger.info(f"Memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Error validating data: {str(e)}")
            raise
    
    def train(self, data: pd.DataFrame) -> None:
        """Train models for all targets"""
        try:
            total_models = len(self.models)
            self.logger.info(f"\nTraining {total_models} models...")
            
            # Validate and clean data first
            self._validate_data(data)
            data = self._clean_data(data)
            data = self.prepare_target_data(data)
            
            for idx, (target, model) in enumerate(self.models.items(), 1):
                self.logger.info(f"\nTraining model {idx}/{total_models} for {target}")
                
                try:
                    # Enhanced feature selection
                    X = self.feature_selector.select_optimal_features(data, target).copy()
                    
                    # Only keep numeric columns for correlation calculation
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    X_numeric = X[numeric_cols]
                    
                    # Remove highly correlated features
                    corr_matrix = X_numeric.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                    X = X.drop(columns=to_drop, errors='ignore')
                    
                    # Remove low variance features (only for numeric columns)
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=0.01)
                    numeric_cols = X.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        X_numeric = X[numeric_cols]
                        selector.fit(X_numeric)
                        X = X.drop(columns=numeric_cols[~selector.get_support()], errors='ignore')
                    
                    y = data[target].copy()
                    
                    # Memory optimization
                    X = self._optimize_memory(X)
                    
                    # Train the model
                    model.train_model(X, y)
                    
                    # Store feature importance
                    self.feature_importance[target] = model.get_feature_importance()
                    
                except Exception as e:
                    self.logger.error(f"Error training model for {target}: {str(e)}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Dict]:
        try:
            predictions = {}
            data = self._clean_data(data)
            
            # Add consistency features first
            consistency_features = ['pts_consistency', 'ast_consistency', 'reb_consistency', 'is_high_consistency']
            consistency_data = self.consistency_tracker.get_consistency_features(data)
            
            # Ensure all consistency features exist
            for feat in consistency_features:
                if feat not in data.columns:
                    data[feat] = consistency_data[feat] if feat in consistency_data.columns else 0.0
            
            # Now select features for each target
            for target in self.models:
                if target not in predictions:
                    predictions[target] = {
                        'prediction': [],
                        'confidence': [],
                        'lower_bound': [],
                        'upper_bound': [],
                        'uncertainty': []
                    }
                
                # Include consistency features in feature selection
                X = self.feature_selector.select_optimal_features(data, target)
                
                # Ensure consistency features are included
                for feat in consistency_features:
                    if feat not in X.columns:
                        X[feat] = data[feat]
                
                pred_dict = self.models[target].predict(X)
                
                # Append results
                for key in pred_dict:
                    predictions[target][key].extend(pred_dict[key])
            
            # Convert lists to arrays
            for target in predictions:
                for key in predictions[target]:
                    predictions[target][key] = np.array(predictions[target][key])
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def _calculate_confidence(self, predictions: np.ndarray, features: pd.DataFrame, 
                            high_consistency_mask: np.ndarray) -> np.ndarray:
        """Calculate confidence scores with enhanced weighting and validation"""
        try:
            confidence = np.zeros(len(predictions))
            
            # Minutes correlation (30% weight)
            if 'minutes_trend' in features.columns:
                confidence += features['minutes_trend'].values * 0.30
            
            # Recent form (25% weight)
            form_cols = [col for col in features.columns if '_recent_form' in col]
            if form_cols:
                recent_form = features[form_cols].mean(axis=1)
                confidence += recent_form.values * 0.25
            
            # Consistency (25% weight)
            consistency_cols = [col for col in features.columns if '_consistency' in col]
            if consistency_cols:
                avg_consistency = features[consistency_cols].mean(axis=1)
                confidence += (avg_consistency.values * 0.15)  # Base consistency
                confidence += (high_consistency_mask.astype(float) * 0.10)  # High consistency bonus
            
            # Context factors (20% weight)
            if 'days_rest' in features.columns:
                optimal_rest = (features['days_rest'].values >= 1) & (features['days_rest'].values <= 3)
                confidence += optimal_rest.astype(float) * 0.05
            
            if 'home_away' in features.columns:
                confidence += (features['home_away'].values == 1) * 0.05
            
            # New: Performance trend (5%)
            if 'performance_trend' in features.columns:
                confidence += np.clip(features['performance_trend'].values, 0, 1) * 0.05
            
            # New: Minutes stability (5%)
            if 'minutes_stability' in features.columns:
                min_stability = 1 - np.clip(features['minutes_stability'].values / 10, 0, 1)
                confidence += min_stability * 0.05
            
            # Validate confidence scores
            confidence = np.clip(confidence, 0, 1)
            
            # Add uncertainty adjustment
            if len(self.cv_results.get('residuals', [])) > 0:
                residual_std = np.std(self.cv_results['residuals'])
                uncertainty = np.abs(predictions - np.mean(predictions)) / residual_std
                confidence *= (1 - np.clip(uncertainty * 0.1, 0, 0.3))  # Reduce confidence by up to 30% based on uncertainty
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return np.zeros(len(predictions))
    
    def _optimize_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage while preserving precision"""
        try:
            data = data.copy()
            
            # Only optimize specific columns that don't need high precision
            optimize_cols = [
                'GAME_ID', 'TEAM_ID', 'PLAYER_ID',
                'home_away', 'days_rest', 'player_role'
            ]
            
            for col in data.columns:
                if col in optimize_cols:
                    # Aggressive optimization for non-critical columns
                    if data[col].dtype.kind == 'i':
                        data[col] = pd.to_numeric(data[col], downcast='integer')
                    elif data[col].dtype.kind == 'f':
                        data[col] = pd.to_numeric(data[col], downcast='float')
                else:
                    # Preserve precision for important features
                    if data[col].dtype.kind == 'f':
                        data[col] = data[col].astype('float32')  # Use float32 instead of float64
                    
            # Use categorical for string columns
            for col in data.select_dtypes(include=['object']):
                if data[col].nunique() / len(data[col]) < 0.5:  # Only if cardinality is reasonable
                    data[col] = data[col].astype('category')
            
            # Log memory savings
            original_mem = data.memory_usage(deep=True).sum() / 1024**2
            optimized_mem = data.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"Memory optimization: {original_mem:.2f}MB -> {optimized_mem:.2f}MB")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error optimizing memory: {str(e)}")
            return data
    
    def analyze_feature_importance(self, data: pd.DataFrame) -> None:
        """Analyze feature importance with larger sample size"""
        try:
            for target in self.models:
                self.logger.info(f"\nAnalyzing feature importance for {target}")
                
                # Increase max_samples but use chunking
                chunk_size = 100000  # Process in 100k chunks
                total_samples = min(len(data), 500000)  # Use up to 500k samples
                
                importance_list = []
                
                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    chunk_data = data.iloc[start_idx:end_idx]
                    
                    chunk_importance = self.feature_selector.calculate_feature_importance(
                        data=chunk_data,
                        target=target
                    )
                    
                    importance_list.append(chunk_importance)
                
                # Combine importance scores
                if importance_list:
                    self.feature_importance[target] = pd.concat(importance_list).groupby(level=0).mean()
                    
        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {str(e)}")
    
    def _add_consistency_features(self, X: pd.DataFrame) -> None:
        """Add player consistency features to the dataset"""
        try:
            # Add consistency scores
            consistency_scores = X.index.map(
                lambda x: self.consistency_tracker.get_player_consistency(x)
            )
            consistency_scores = pd.Series(consistency_scores).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)
            X.loc[:, 'player_consistency'] = self._normalize_feature(consistency_scores)
            
            # Add binary high consistency indicator
            high_consistency = X.index.map(
                lambda x: float(self.consistency_tracker.is_consistent_player(x))
            )
            X.loc[:, 'is_high_consistency'] = pd.Series(high_consistency).fillna(0.0)
            
        except Exception as e:
            self.logger.error(f"Error adding consistency features: {str(e)}")
    
    def _add_minutes_correlation(self, X: pd.DataFrame) -> None:
        """Enhanced minutes correlation tracking"""
        try:
            # Calculate rolling minutes correlation with performance (increased window)
            for stat in ['PTS', 'AST', 'REB']:
                if stat in X.columns and 'MIN' in X.columns:
                    X[f'{stat}_min_corr'] = X.groupby(level=0).apply(
                        lambda x: x[stat].rolling(15, min_periods=5).corr(x['MIN'])
                    ).fillna(0)
                    
            # Add minutes trend feature with more weight on recent games
            X['minutes_trend'] = X.groupby(level=0)['MIN'].transform(
                lambda x: x.ewm(span=5).mean()  # Exponential weighted mean for recency
            ).fillna(X['MIN'])
            
        except Exception as e:
            self.logger.error(f"Error adding minutes correlation: {str(e)}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare all required features using DataAnalyzer results"""
        try:
            df = data.copy()
            
            # Debug print of columns
            self.logger.info(f"Available columns: {df.columns.tolist()}")
            
            # Standardize Player_ID column
            if 'PLAYER_ID' in df.columns and 'Player_ID' not in df.columns:
                df['Player_ID'] = df['PLAYER_ID']
            elif 'Player_ID' in df.columns and 'PLAYER_ID' not in df.columns:
                df['PLAYER_ID'] = df['Player_ID']
            
            # If both exist, keep PLAYER_ID and drop Player_ID
            if 'PLAYER_ID' in df.columns and 'Player_ID' in df.columns:
                df = df.drop(columns=['Player_ID'])
                df = df.rename(columns={'PLAYER_ID': 'Player_ID'})
            
            # Verify we have a Player_ID column
            if 'Player_ID' not in df.columns:
                raise ValueError("No Player_ID column found after standardization")
            
            # Debug print of Player_ID column
            self.logger.info(f"Player_ID dtype: {df['Player_ID'].dtype}")
            self.logger.info(f"Player_ID sample:\n{df['Player_ID'].head()}")
            
            # Ensure Player_ID is a single value column
            if isinstance(df['Player_ID'].iloc[0], (list, np.ndarray)):
                self.logger.info("Converting Player_ID from list/array to single value")
                df['Player_ID'] = df['Player_ID'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
            
            # Convert Player_ID to string to ensure consistency
            df['Player_ID'] = df['Player_ID'].astype(str)
            
            # Ensure GAME_DATE is datetime
            if 'GAME_DATE' in df.columns:
                df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
                
                # Calculate days rest
                df = df.sort_values(['Player_ID', 'GAME_DATE'])
                df['days_rest'] = df.groupby('Player_ID')['GAME_DATE'].diff().dt.days.fillna(1)
            
            # Remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            self.logger.error(f"DataFrame info:\n{df.info()}")
            self.logger.error(f"Columns with 'player' or 'Player': {[col for col in df.columns if 'player' in col.lower()]}")
            raise