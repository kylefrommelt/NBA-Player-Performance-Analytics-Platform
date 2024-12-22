import pandas as pd
import numpy as np
from typing import List, Dict
import xgboost as xgb
import logging

class FeatureSelector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Core features with importance weights
        self.key_features = {
            'PTS': [
                ('FGM', 0.65),                # Primary scoring indicator
                ('FGA', 0.11),                # Volume indicator
                ('PTS_recent_form', 0.09),    # Recent performance
                ('FTM', 0.04),                # Free throw contribution
                ('FG3M', 0.04),               # 3-point contribution
                ('MIN', 0.03),                # Playing time
                ('FG_PCT', 0.02),             # Efficiency
                ('pts_consistency', 0.01),     # Consistency metric
                ('PTS_rolling_mean', 0.01),    # Long-term average
            ],
            'AST': [
                ('AST_recent_form', 0.70),     # Recent assist performance
                ('AST_rolling_mean', 0.08),    # Long-term average
                ('MIN', 0.06),                 # Playing time
                ('PLUS_MINUS', 0.04),          # Overall impact
                ('PTS_rolling_mean', 0.03),    # Scoring correlation
                ('ast_consistency', 0.03),     # Consistency metric
                ('TOV', 0.02),                 # Ball handling
                ('USG_PCT', 0.02),             # Usage rate
                ('PACE', 0.02),                # Game pace impact
            ],
            'REB': [
                ('DREB', 0.80),               # Defensive rebounds
                ('OREB', 0.15),               # Offensive rebounds
                ('REB_recent_form', 0.02),    # Recent performance
                ('MIN', 0.01),                # Playing time
                ('HEIGHT', 0.01),             # Physical attribute
                ('reb_consistency', 0.01),    # Consistency metric
            ]
        }
        
        # Define combined prop features with weights
        self.key_features['PTS_REB'] = (
            [(f, w * 0.7) for f, w in self.key_features['PTS']] +
            [(f, w * 0.3) for f, w in self.key_features['REB']]
        )
        
        self.key_features['PTS_AST'] = (
            [(f, w * 0.7) for f, w in self.key_features['PTS']] +
            [(f, w * 0.3) for f, w in self.key_features['AST']]
        )
        
        self.key_features['PTS_AST_REB'] = (
            [(f, w * 0.6) for f, w in self.key_features['PTS']] +
            [(f, w * 0.2) for f, w in self.key_features['AST']] +
            [(f, w * 0.2) for f, w in self.key_features['REB']]
        )
        
        # Context features with importance weights
        self.context_features = [
            ('days_rest', 0.03),          # Rest impact
            ('home_away', 0.02),          # Home court advantage
            ('back_to_back', 0.02),       # B2B impact
            ('opponent_rank', 0.02),      # Opponent strength
            ('player_role', 0.01),        # Role in team
        ]
        
        # Defensive context features with weights
        self.defensive_features = {
            'DEF_RATING': 0.04,
            'OPP_PTS': 0.03,
            'OPP_FG_PCT': 0.03,
            'OPP_PTS_PAINT': 0.02,
            'OPP_PTS_FB': 0.02,
            'OPP_PTS_2ND_CHANCE': 0.01,
            'OPP_PTS_OFF_TOV': 0.01
        }

    def _calculate_feature_importance(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Calculate feature importance with optimized XGBoost parameters"""
        try:
            model = xgb.XGBRegressor(
                n_estimators=200,          # More trees for better feature importance
                learning_rate=0.05,        # Lower learning rate for stability
                max_depth=4,              # Slightly deeper trees
                subsample=0.8,            # Prevent overfitting
                colsample_bytree=0.8,     # Feature sampling
                random_state=42,
                early_stopping_rounds=20,  # Moved here from fit params
                enable_categorical=True    # Enable categorical feature support
            )
            
            X = df.select_dtypes(include=[np.number])
            y = df[target_col]
            
            # Split data for evaluation
            eval_size = min(2000, len(X) // 5)  # Use 20% or 2000 samples, whichever is smaller
            eval_idx = np.random.choice(len(X), eval_size, replace=False)
            eval_mask = np.zeros(len(X), dtype=bool)
            eval_mask[eval_idx] = True
            
            # Fit model with eval set but without early_stopping_rounds parameter
            model.fit(
                X[~eval_mask], y[~eval_mask],
                eval_set=[(X[eval_mask], y[eval_mask])],
                verbose=False
            )
            
            # Get feature importance
            importance = pd.Series(
                model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
            
            # Log top features
            self.logger.info(f"\nTop 10 important features for {target_col}:")
            for feat, imp in importance.head(10).items():
                self.logger.info(f"{feat}: {imp:.4f}")
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.Series()

    def select_optimal_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Select and weight features optimally for props prediction"""
        try:
            # Get base features with weights
            base_features = self.key_features[target_col]
            required_features = [f[0] for f in base_features]
            
            # Always include base stats and their combinations
            base_stats = ['PTS', 'AST', 'REB']
            combined_stats = ['PTS_REB', 'PTS_AST', 'PTS_AST_REB']
            
            # Add base stats and their derived features
            for stat in base_stats:
                required_features.extend([
                    stat,
                    f"{stat}_recent_form",
                    f"{stat}_rolling_mean",
                    f"{stat.lower()}_consistency"
                ])
            
            # Add combined stats
            required_features.extend(combined_stats)
            
            # For combined stats, ensure we include all component features
            if '_' in target_col:
                individual_stats = target_col.split('_')
                for stat in individual_stats:
                    if stat in self.key_features:
                        stat_features = [f[0] for f in self.key_features[stat]]
                        required_features.extend(stat_features)
            
            # Add context features
            required_features.extend([f[0] for f in self.context_features])
            
            # Remove duplicates while preserving order
            required_features = list(dict.fromkeys(required_features))
            
            # Select available features
            available_features = [col for col in required_features if col in df.columns]
            selected_df = df[available_features].copy()
            
            # For missing but required features, add them with appropriate defaults
            for feature in required_features:
                if feature not in selected_df.columns:
                    # Handle base stats
                    if feature in base_stats:
                        recent_form = f"{feature}_recent_form"
                        rolling_mean = f"{feature}_rolling_mean"
                        if recent_form in selected_df.columns:
                            selected_df[feature] = selected_df[recent_form]
                        elif rolling_mean in selected_df.columns:
                            selected_df[feature] = selected_df[rolling_mean]
                        else:
                            selected_df[feature] = 0
                    
                    # Handle combined stats
                    elif feature in combined_stats:
                        components = feature.split('_')
                        if all(comp in selected_df.columns for comp in components):
                            selected_df[feature] = sum(selected_df[comp] for comp in components)
                        else:
                            selected_df[feature] = 0
                    
                    # Handle derived features
                    elif '_recent_form' in feature:
                        base_stat = feature.replace('_recent_form', '')
                        if base_stat in selected_df.columns:
                            selected_df[feature] = selected_df[base_stat]
                        else:
                            selected_df[feature] = 0
                    
                    elif '_rolling_mean' in feature:
                        base_stat = feature.replace('_rolling_mean', '')
                        if base_stat in selected_df.columns:
                            selected_df[feature] = selected_df[base_stat]
                        else:
                            selected_df[feature] = 0
                    
                    # Default case
                    else:
                        selected_df[feature] = 0
            
            # Apply feature weights
            for feature, weight in base_features:
                if feature in selected_df.columns:
                    selected_df[feature] *= weight
            
            # Apply defensive feature weights
            for feature, weight in self.defensive_features.items():
                if feature in selected_df.columns:
                    selected_df[feature] *= weight
            
            # Handle missing values
            numeric_cols = selected_df.select_dtypes(include=[np.number]).columns
            selected_df[numeric_cols] = selected_df[numeric_cols].fillna(0)
            
            return selected_df
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            self.logger.error(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Error selecting features: {str(e)}")