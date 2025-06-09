import json
import joblib
import numpy as np
import pandas as pd
import random
from datetime import datetime
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union

# Import feature engineering functions from common_functions.py
from Classifier import parse_condition,parse_condition_recursive, check_condition, get_temporal_multiplier, load_conditions

# Load the feature names to ensure consistent ordering
print("Loading feature names...")
with open('feature_names.json', 'r') as f:
    FEATURE_NAMES = json.load(f)

# Define MODEL_FEATURE_KEYS to ensure consistency
MODEL_FEATURE_KEYS = ["login_streak", "posts_created", "comments_written", "upvotes_received", 
                     "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned_today"]

class RewardEngine:
    def __init__(self):
        """Initialize the reward engine with the trained model and configurations."""
        self.model = load_model()
        self.conditions = load_conditions()
        self.config = self._load_config()
        # Define expected feature names (raw + rule-based + temporal)
        self.expected_feature_names = MODEL_FEATURE_KEYS + [f"rule_{i}" for i in range(len(self.conditions))] + ["temporal_multiplier"]
    
    def _load_config(self):
        """Load and validate the configuration."""
        with open('config.json', 'r') as f:
            config = json.load(f)
        # Validate required fields
        required_fields = ['reward_probability_threshold', 'reward_rules', 'box_types', 'karma_min', 'karma_max']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        return config
    
    def _evaluate_rule(self, rule_conditions, metrics):
        """Evaluate if all conditions in a rule are met."""
        if not isinstance(rule_conditions, list):
            rule_conditions = [rule_conditions]
            
        for condition in rule_conditions:
            parsed_condition = parse_condition(condition)
            if not check_condition(metrics, parsed_condition):
                return False
        return True
    
    def _determine_box_type(self, metrics):
        """Determine the type of box based on user metrics and reward rules."""
        for box_type, conditions in self.config['reward_rules'].items():
            if self._evaluate_rule(conditions, metrics):
                return box_type
        return "mystery"  # Default box type if no rules match
    
    def _calculate_rarity(self, box_type, prediction_probability, seed=None):
        """Calculate the rarity of the reward based on probability and box type."""
        if box_type not in self.config['box_types']:
            box_type = 'mystery'
            
        box_config = self.config['box_types'][box_type]
        rng = random.Random(seed) if seed is not None else random
        
        # Adjust weights based on prediction probability
        weights = list(box_config['rarity_weights'].values())
        adjusted_weights = [w * (1 + prediction_probability) for w in weights]
        total = sum(adjusted_weights)
        adjusted_weights = [w/total for w in adjusted_weights]
        
        # Select rarity based on weights
        rarities = list(box_config['rarity_weights'].keys())
        return rng.choices(rarities, weights=adjusted_weights, k=1)[0]
    
    def _calculate_reward_karma(self, box_type, rarity, metrics):
        """Calculate the karma reward based on box type, rarity, and user metrics."""
        if box_type not in self.config['box_types']:
            box_type = 'mystery'
            
        base_karma = self.config['box_types'][box_type]['base_karma']
        
        # Apply rarity multiplier
        rarity_multipliers = {
            'common': 1.0,
            'rare': 1.5,
            'elite': 2.0,
            'legendary': 3.0
        }
        
        # Apply activity bonus (0-50% based on overall activity)
        activity_score = sum(metrics.values()) / len(metrics) if metrics else 0
        activity_bonus = 1.0 + (activity_score / 100) * 0.5
        
        # Calculate final karma
        karma = int(base_karma * rarity_multipliers[rarity] * activity_bonus)
        
        # Ensure karma is within bounds
        return max(self.config['karma_min'], min(self.config['karma_max'], karma))

    def _get_deterministic_seed(self, user_id: str, date: str) -> int:
        """Generate a deterministic seed based on user_id and date."""
        seed_str = f"{user_id}_{date}"
        hash_value = hashlib.md5(seed_str.encode()).hexdigest()
        return int(hash_value[:8], 16)
    
    def _prepare_features(self, daily_metrics: Dict[str, Any], date: str) -> np.ndarray:
        """
        Prepare the feature vector for model prediction, including raw, rule-based, and temporal features.
        """
        # Create DataFrame with raw features and ensure feature names are preserved
        features = {key: [daily_metrics.get(key, 0)] for key in MODEL_FEATURE_KEYS}
        X = pd.DataFrame(features)
        
        # Add rule-based features
        rule_features = np.zeros((1, len(self.conditions)))
        for j, cond in enumerate(self.conditions):
            parsed_condition = parse_condition(cond["condition"])
            rule_features[0, j] = check_condition(daily_metrics, parsed_condition)
        rule_cols = [f"rule_{i}" for i in range(len(self.conditions))]
        rule_df = pd.DataFrame(rule_features, columns=rule_cols, index=X.index)
        X = pd.concat([X, rule_df], axis=1)
        
        # Add temporal feature
        day_dt = datetime.strptime(date, "%Y-%m-%d")
        temporal_mult = get_temporal_multiplier(day_dt.weekday(), day_dt.month)
        X["temporal_multiplier"] = temporal_mult
        
        # Ensure correct column order and feature names
        X = X[self.expected_feature_names]
        
        return X
    
    def _find_matching_condition(self, 
                               daily_metrics: Dict[str, Any], 
                               prediction: int,
                               prediction_probability: float) -> Optional[Dict[str, Any]]:
        """Find the condition that best matches the user's metrics."""
        # Filter conditions by prediction label
        matching_conditions = [
            cond for cond in self.conditions 
            if cond.get('label') == prediction
        ]
        
        if not matching_conditions:
            return None
        
        # Find all conditions that match the user's metrics
        matched_conditions = []
        for cond in matching_conditions:
            parsed_condition = cond.get('parsed_condition')
            if parsed_condition is None:
                parsed_condition = parse_condition(cond.get('condition', ''))
            if parsed_condition is not None and check_condition(daily_metrics, parsed_condition):
                matched_conditions.append(cond)
        
        if not matched_conditions:
            if prediction == 1:
                # Return the condition with highest probability if no match found but prediction is 1
                return max(matching_conditions, key=lambda x: x.get('probability', 0))
            return None
        
        # Select a condition based on probability weights
        probabilities = [cond.get('probability', 0) for cond in matched_conditions]
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(matched_conditions)] * len(matched_conditions)
        
        rng = random.Random(self._get_deterministic_seed(
            daily_metrics.get('user_id', ''), 
            daily_metrics.get('date', '')
        ))
        
        return rng.choices(matched_conditions, weights=probabilities, k=1)[0]
    
    def check_surprise_box(self, user_id: str, date: str, daily_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a user qualifies for a surprise box and calculate the reward.
        
        Args:
            user_id: The ID of the user
            date: The date of the check (YYYY-MM-DD)
            daily_metrics: Dictionary containing the user's daily metrics
            
        Returns:
            Dictionary containing reward details
        """
        try:
            # Generate deterministic seed for reproducibility
            seed = self._get_deterministic_seed(user_id, date)
            random.seed(seed)
            
            # Check if user qualifies for any reward
            box_type = self._determine_box_type(daily_metrics)
            
            # Get prediction probability from model
            features = self._prepare_features(daily_metrics, date)
            prediction_probability = self.model.predict_proba(features)[0][1]
            
            # Check if probability meets threshold
            if prediction_probability < self.config['reward_probability_threshold']:
                return {
                    "user_id": user_id,
                    "surprise_unlocked": False,
                    "status": "missed",
                    "reason": "Activity level below reward threshold"
                }
            
            # Determine reward details
            rarity = self._calculate_rarity(box_type, prediction_probability, seed)
            reward_karma = self._calculate_reward_karma(box_type, rarity, daily_metrics)
            
            # Get box display name
            box_name = self.config['box_types'].get(box_type, {}).get('name', 'Mystery Box')
            
            return {
                "user_id": user_id,
                "surprise_unlocked": True,
                "reward_karma": reward_karma,
                "box_type": box_type,
                "box_name": box_name,
                "rarity": rarity,
                "status": "delivered",
                "reason": f"Earned {rarity} {box_name} for your activity!"
            }
            
        except Exception as e:
            print(f"Error in check_surprise_box: {str(e)}")
            return {
                "user_id": user_id,
                "surprise_unlocked": False,
                "status": "error",
                "reason": f"Error processing request: {str(e)}"
            }

def load_model():
    """Load the trained classifier model."""
    try:
        with open('classifier_bal_1.pkl', 'rb') as f:
            model = joblib.load(f)
            print('model loaded')
        return model
    # try:
    #     with open('classifier.pkl', 'rb') as f:
    #         model = joblib.load(f)
    #         print('model loaded')
    #     return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_reward_engine():
    """Get or create the reward engine singleton."""
    return RewardEngine()