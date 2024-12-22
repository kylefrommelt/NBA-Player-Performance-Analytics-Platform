import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, KFold
import xgboost as xgb
from typing import Tuple, List, Dict
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import gc

class PropsPredictionPipeline:
    def __init__(self, target: str):
        self.target = target
        self.pipeline = None
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.cv_results = {'residuals': []}
        
    def create_pipeline(self) -> Pipeline:
        """Create the model pipeline with preprocessing steps"""
        try:
            self.pipeline = Pipeline([
                ('scaler', RobustScaler()),
                ('model', xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=3,
                    subsample=0.7,
                    colsample_bytree=0.7,
                    min_child_weight=5,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                    random_state=42,
                    sample_weight_eval_set=True
                ))
            ])
            return self.pipeline
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline: {str(e)}")
            raise
        
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model with given features and target"""
        try:
            self.logger.info(f"Training {self.target} model...")
            
            # Store original feature names before preprocessing
            self.original_feature_names = X.columns.tolist()
            
            # Ensure all column names are strings
            X = X.rename(columns=lambda x: str(x))
            
            # Drop any non-numeric columns before converting to float32
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X = X[numeric_cols].astype('float32')
            
            # Create pipeline if not exists
            if self.pipeline is None:
                self.create_pipeline()
            
            # Fit the pipeline
            self.pipeline.fit(X, y)
            
            # Store feature names after fitting
            self.feature_names_in_ = X.columns.tolist()  # Store from cleaned DataFrame instead
            
            # Calculate residuals for confidence intervals
            predictions = self.pipeline.predict(X)
            residuals = y - predictions
            self.cv_results['residuals'] = residuals.tolist()
            
            self.logger.info(f"Model trained with {len(self.feature_names_in_)} features")
            
        except Exception as e:
            self.logger.error(f"Error training model for {self.target}: {str(e)}")
            self.pipeline = None
            raise
        
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions with enhanced confidence scoring"""
        try:
            if self.pipeline is None:
                raise ValueError(f"Pipeline for {self.target} not trained")
            
            # Ensure proper feature alignment
            X = X[self.feature_names_in_]
            X = X.astype('float32')
            
            # Make predictions
            raw_predictions = self.pipeline.predict(X)
            
            # Calculate residual-based metrics
            residual_std = np.std(self.cv_results['residuals'])
            mean_target = np.mean(raw_predictions)
            
            # Apply calibration based on stat type
            predictions = np.copy(raw_predictions)
            for i in range(len(predictions)):
                if '_' in self.target:  # Combined stats
                    if predictions[i] > mean_target * 1.8:
                        predictions[i] = mean_target + (predictions[i] - mean_target) * 0.4
                    elif predictions[i] > mean_target * 1.4:
                        predictions[i] = mean_target + (predictions[i] - mean_target) * 0.5
                    else:
                        predictions[i] = mean_target + (predictions[i] - mean_target) * 0.6
                else:  # Individual stats
                    if predictions[i] > mean_target * 1.5:
                        predictions[i] = mean_target + (predictions[i] - mean_target) * 0.3
                    elif predictions[i] > mean_target * 1.2:
                        predictions[i] = mean_target + (predictions[i] - mean_target) * 0.4
                    elif predictions[i] < mean_target * 0.5:
                        predictions[i] = mean_target - (mean_target - predictions[i]) * 0.3
                    elif predictions[i] < mean_target * 0.8:
                        predictions[i] = mean_target - (mean_target - predictions[i]) * 0.4
                    else:
                        predictions[i] = mean_target + (predictions[i] - mean_target) * 0.6
            
            # Calculate confidence levels
            confidence = np.array(['Medium'] * len(predictions))
            uncertainty = pd.Series(predictions).rolling(2, min_periods=1).std().fillna(residual_std * 0.3)
            
            # Adjust confidence thresholds
            thresholds = {
                'High': 0.15 * mean_target if '_' not in self.target else 0.20 * mean_target,
                'Medium': 0.25 * mean_target if '_' not in self.target else 0.30 * mean_target,
                'Low': float('inf')
            }
            
            # Set confidence levels
            for level, threshold in thresholds.items():
                mask = uncertainty <= threshold
                confidence[mask] = level
            
            # Calculate confidence intervals
            ci_multiplier = 1.2
            lower_bound = predictions - ci_multiplier * uncertainty
            upper_bound = predictions + ci_multiplier * uncertainty
            
            return {
                'prediction': predictions,
                'confidence': confidence,
                'uncertainty': uncertainty.values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores"""
        try:
            if self.pipeline is None:
                raise ValueError(f"Pipeline for {self.target} not trained. Call train_model() first.")
                
            model = self.pipeline.named_steps['model']
            importance = pd.Series(
                model.feature_importances_,
                index=model.feature_names_in_
            )
            return importance.sort_values(ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            return pd.Series()