import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

class PlayerConsistencyTracker:
    def __init__(self):
        self.consistency_scores = {}
        self.high_consistency_players = {}
        self.consistency_thresholds = {
            'PTS': 0.70,
            'AST': 0.65,
            'REB': 0.65,
            'default': 0.70
        }
        self.high_consistency_threshold = 0.75
        self.min_games = 10
        self.logger = logging.getLogger(__name__)
        
    def _calculate_consistency_score(self, values: np.ndarray) -> float:
        """Calculate consistency score with recency bias"""
        try:
            # Convert to numpy array if not already
            values = np.array(values)
            
            # Remove any NaN values
            values = values[~np.isnan(values)]
            
            if len(values) < self.min_games:
                return 0.0
            
            # Add exponential decay for recency bias
            weights = np.exp(-0.1 * np.arange(len(values)))[::-1]
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate weighted statistics
            weighted_mean = np.average(values, weights=weights)
            weighted_var = np.average((values - weighted_mean)**2, weights=weights)
            weighted_std = np.sqrt(weighted_var)
            
            if weighted_mean == 0:
                return 0.0
            
            # Calculate weighted coefficient of variation
            weighted_cv = weighted_std / weighted_mean
            
            # Convert to consistency score (1 - cv, bounded between 0 and 1)
            consistency = max(0.0, min(1.0, 1 - weighted_cv))
            
            return float(consistency)
            
        except Exception as e:
            self.logger.error(f"Error calculating consistency score: {str(e)}")
            return 0.0
        
    def validate_consistency_scores(self) -> bool:
        """Validate that consistency scores are properly calculated"""
        try:
            # Check if we have any scores
            if not self.consistency_scores:
                self.logger.warning("No consistency scores calculated yet")
                return False
            
            # Check score ranges
            for player, scores in self.consistency_scores.items():
                for stat, score in scores.items():
                    if not (0 <= score <= 1):
                        self.logger.error(f"Invalid consistency score for {player} {stat}: {score}")
                        return False
            
            # Log validation success
            total_players = len(self.consistency_scores)
            self.logger.info(f"Validated consistency scores for {total_players} players")
            return True
        
        except Exception as e:
            self.logger.error(f"Error validating consistency scores: {str(e)}")
            return False
        
    def update_consistency_scores(self, data: pd.DataFrame) -> None:
        """Update consistency scores for all players"""
        try:
            self.logger.info("\nUpdating consistency scores...")
            
            # Ensure we have required columns
            required_cols = ['Player_ID', 'PTS', 'AST', 'REB']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Group by player ID and calculate consistency metrics
            player_groups = data.groupby('Player_ID')
            
            # Track progress
            total_players = len(player_groups)
            processed = 0
            
            for player_id, player_data in player_groups:
                # Initialize dictionary for this player if it doesn't exist
                if player_id not in self.consistency_scores:
                    self.consistency_scores[player_id] = {}
                
                # Calculate consistency scores for each stat
                for stat in ['PTS', 'AST', 'REB']:
                    if stat in player_data.columns:
                        recent_values = player_data[stat].values[-self.min_games:]
                        if len(recent_values) >= self.min_games:
                            consistency_score = self._calculate_consistency_score(recent_values)
                            self.consistency_scores[player_id][stat] = consistency_score
                        else:
                            self.consistency_scores[player_id][stat] = 0.0
                
                # Update overall consistency flag
                self.high_consistency_players[player_id] = self._is_consistent_overall(
                    self.consistency_scores[player_id]
                )
                
                processed += 1
                if processed % 100 == 0:
                    self.logger.info(f"Processed {processed}/{total_players} players")
            
            self.logger.info(f"Completed consistency updates for {total_players} players")
            
        except Exception as e:
            self.logger.error(f"Error updating consistency scores: {str(e)}")
            self.logger.error(f"Data shape: {data.shape}")
            self.logger.error(f"Available columns: {data.columns.tolist()}")
        
    def _log_consistency_summary(self) -> None:
        """Log summary statistics for consistency scores"""
        try:
            self.logger.info("\nConsistency Score Summary:")
            for stat in ['PTS', 'AST', 'REB']:
                scores = [scores.get(stat, 0.0) 
                         for scores in self.consistency_scores.values()]
                
                if scores:
                    non_zero_scores = [s for s in scores if s > 0]
                    if non_zero_scores:
                        self.logger.info(
                            f"{stat} Consistency - "
                            f"Mean: {np.mean(non_zero_scores):.3f}, "
                            f"Std: {np.std(non_zero_scores):.3f}, "
                            f"Min: {min(non_zero_scores):.3f}, "
                            f"Max: {max(non_zero_scores):.3f}, "
                            f"Active Players: {len(non_zero_scores)}, "
                            f"High Consistency Players: {sum(s > self.high_consistency_threshold for s in scores)}"
                        )
                    else:
                        self.logger.warning(f"No non-zero consistency scores found for {stat}")
        except Exception as e:
            self.logger.error(f"Error logging consistency summary: {str(e)}")
        
    def get_player_consistency(self, player_id) -> float:
        """Get overall consistency score for a player"""
        try:
            # Get scores for each stat
            scores = []
            for stat in ['PTS', 'AST', 'REB']:
                if stat in self.consistency_scores and player_id in self.consistency_scores[stat]:
                    scores.append(self.consistency_scores[stat][player_id])
            
            # Return average if we have scores, otherwise 0
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error getting consistency for player {player_id}: {str(e)}")
            return 0.0
        
    def is_consistent_player(self, player_id) -> bool:
        """Determine if a player is consistently performing"""
        try:
            consistency = self.get_player_consistency(player_id)
            return consistency > 0.65  # Updated threshold
        except Exception as e:
            self.logger.warning(f"Error checking consistency for player {player_id}: {str(e)}")
            return False
        
    def get_consistency_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create consistency features for all players"""
        try:
            # First update the consistency scores
            self.update_consistency_scores(data)
            
            # Create a new DataFrame with just Player_ID
            result = data.copy()
            
            # Initialize ALL required consistency columns with zeros
            consistency_cols = {
                'pts_consistency': 0.0,
                'ast_consistency': 0.0,
                'reb_consistency': 0.0,
                'is_high_consistency': 0.0
            }
            
            for col, default in consistency_cols.items():
                result[col] = default
            
            # Update consistency scores for each player
            for player_id, scores in self.consistency_scores.items():
                player_mask = result['Player_ID'] == player_id
                
                # Update individual stat consistencies
                for stat, score in scores.items():
                    col_name = f"{stat.lower()}_consistency"
                    result.loc[player_mask, col_name] = score
                
                # Update high consistency flag
                is_high_consistency = float(self._is_consistent_overall(scores))
                result.loc[player_mask, 'is_high_consistency'] = is_high_consistency
            
            # Ensure all consistency columns are float32
            for col in consistency_cols:
                result[col] = result[col].astype('float32')
            
            # Log summary statistics
            self._log_consistency_summary()
            
            # Validate the results
            for col in consistency_cols:
                if col not in result.columns:
                    raise ValueError(f"Missing consistency column: {col}")
                if result[col].isna().any():
                    self.logger.warning(f"Found NaN values in {col}, filling with 0.0")
                    result[col] = result[col].fillna(0.0)
            
            self.logger.info("Successfully created consistency features")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating consistency features: {str(e)}")
            self.logger.error(f"Data shape: {data.shape}")
            self.logger.error(f"Available columns: {data.columns.tolist()}")
            
            # Return original data with ALL required consistency columns
            result = data.copy()
            for col, default in {
                'pts_consistency': 0.0,
                'ast_consistency': 0.0,
                'reb_consistency': 0.0,
                'is_high_consistency': 0.0
            }.items():
                result[col] = default
            return result
        
    def _is_consistent_overall(self, player_scores: Dict[str, float]) -> bool:
        """Determine if a player is consistent overall based on their scores"""
        try:
            if not player_scores:
                return False
            
            # Check each stat against its threshold
            for stat, score in player_scores.items():
                threshold = self.consistency_thresholds.get(stat, self.consistency_thresholds['default'])
                if score < threshold:
                    return False
                
            # If all scores are above their thresholds
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking overall consistency: {str(e)}")
            return False