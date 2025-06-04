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

# Load the configuration
print("Loading configuration...")
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

# Define MODEL_FEATURE_KEYS to ensure consistency
MODEL_FEATURE_KEYS = ["login_streak", "posts_created", "comments_written", "upvotes_received", 
                     "quizzes_completed", "buddies_messaged", "karma_spent", "karma_earned_today"]

class RewardEngine:
    def __init__(self):
        """Initialize the reward engine with the trained model and configurations."""
        print('sadadsaas')
        self.model = load_model()
        print('model loaded',self.model)
        self.conditions = load_conditions()
        print('conditions loaded',self.conditions)
        self.config = CONFIG
        print('config loaded',self.config)
        # Define expected feature names (raw + rule-based + temporal)
        self.expected_feature_names = MODEL_FEATURE_KEYS + [f"rule_{i}" for i in range(len(self.conditions))] + ["temporal_multiplier"]

    def _get_deterministic_seed(self, user_id: str, date: str) -> int:
        """Generate a deterministic seed based on user_id and date."""
        seed_str = f"{user_id}_{date}"
        hash_value = hashlib.md5(seed_str.encode()).hexdigest()
        return int(hash_value[:8], 16)
    
    # def _prepare_features(self, daily_metrics: Dict[str, Any], date: str) -> np.ndarray:
    #     """
    #     Prepare the feature vector for model prediction, including raw, rule-based, and temporal features.
    #     """
    #     # Create DataFrame with raw features
    #     features = {key: daily_metrics.get(key, 0) for key in MODEL_FEATURE_KEYS}
    #     X = pd.DataFrame([features])
        
    #     # Add rule-based features
    #     rule_features = np.zeros((1, len(self.conditions)))
    #     for j, cond in enumerate(self.conditions):
    #         parsed_condition = parse_condition(cond["condition"])
    #         rule_features[0, j] = check_condition(daily_metrics, parsed_condition)
    #     rule_cols = [f"rule_{i}" for i in range(len(self.conditions))]
    #     X = pd.concat([X, pd.DataFrame(rule_features, columns=rule_cols, index=X.index)], axis=1)
        
    #     # Add temporal feature
    #     day_dt = datetime.strptime(date, "%Y-%m-%d")
    #     temporal_mult = get_temporal_multiplier(day_dt.weekday(), day_dt.month)
    #     X["temporal_multiplier"] = temporal_mult
        
    #     # Ensure correct column order
    #     X = X[self.expected_feature_names]
        
    #     return X.values.reshape(1, -1)
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
    def _calculate_reward_karma(self, 
                               prediction_probability: float, 
                               daily_metrics: Dict[str, Any],
                               matched_condition: Dict[str, Any]) -> int:
        """Calculate the reward karma based on prediction probability and user metrics."""
        base_reward = matched_condition['reward_score']
        activity_score = sum([
            daily_metrics.get('login_streak', 0) * 0.5,
            daily_metrics.get('posts_created', 0) * 1.5,
            daily_metrics.get('comments_written', 0) * 0.8,
            daily_metrics.get('upvotes_received', 0) * 0.3,
            daily_metrics.get('quizzes_completed', 0) * 2.0,
            daily_metrics.get('buddies_messaged', 0) * 0.7,
            daily_metrics.get('karma_spent', 0) * 0.1
        ]) / 10.0
        modifier = 1.0 + (prediction_probability - 0.5) * 0.4 + (activity_score / 50.0)
        reward = int(base_reward * modifier)
        reward = max(self.config['karma_min'], min(self.config['karma_max'], reward))
        return reward
    
    def _determine_rarity(self, 
                         prediction_probability: float, 
                         matched_condition: Dict[str, Any],
                         seed: int) -> str:
        """Determine the rarity of the reward box."""
        base_rarity = matched_condition['rarity']
        rng = random.Random(seed)
        rarity_upgrade_chance = prediction_probability * 0.3
        rarity_levels = ['common', 'rare', 'elite', 'legendary']
        if base_rarity in rarity_levels:
            base_index = rarity_levels.index(base_rarity)
            if base_index < len(rarity_levels) - 1 and rng.random() < rarity_upgrade_chance:
                return rarity_levels[base_index + 1]
            return base_rarity
        rarity_dist = self.config['target_rarity_dist']
        rand_val = rng.random()
        cumulative = 0
        for rarity, prob in rarity_dist.items():
            cumulative += prob
            if rand_val <= cumulative:
                return rarity
        return 'common'
    
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
        """Check if a user qualifies for a surprise box and calculate the reward."""
        try:
            metrics_with_id = daily_metrics.copy()
            metrics_with_id['user_id'] = user_id
            metrics_with_id['date'] = date
            
            # Prepare features for the model
            features = self._prepare_features(daily_metrics, date)
            
            # Get prediction and probability
            try:
                prediction = self.model.predict(features)[0]
                prediction_probability = self.model.predict_proba(features)[0][1]
            except Exception as e:
                print(f"Model prediction error: {e}")
                activity_score = sum([
                    daily_metrics.get('login_streak', 0) * 0.5,
                    daily_metrics.get('posts_created', 0) * 1.5,
                    daily_metrics.get('comments_written', 0) * 0.8,
                    daily_metrics.get('upvotes_received', 0) * 0.3,
                    daily_metrics.get('quizzes_completed', 0) * 2.0,
                    daily_metrics.get('buddies_messaged', 0) * 0.7,
                    daily_metrics.get('karma_spent', 0) * 0.1
                ])
                prediction = 1 if activity_score > 10 else 0
                prediction_probability = min(0.95, activity_score / 20)
            
            # Find matching condition
            matched_condition = self._find_matching_condition(daily_metrics, prediction, prediction_probability)
            print('matched_condition',matched_condition)
            # Generate deterministic seed
            seed = self._get_deterministic_seed(user_id, date)
            
            # Default response
            response = {
                "user_id": user_id,
                "surprise_unlocked": False,
                "reward_karma": 0,
                "reason": "Not enough activity",
                "rarity": "none",
                "box_type": "none",
                "status": "missed"
            }
            
            if prediction == 1 and matched_condition is not None:
                try:
                    reward_karma = self._calculate_reward_karma(prediction_probability, daily_metrics, matched_condition)
                    rarity = self._determine_rarity(prediction_probability, matched_condition, seed)
                    reason = matched_condition.get('display_reason', "Great activity!")
                    box_type = matched_condition.get('box_type', "mystery")
                    response.update({
                        "surprise_unlocked": True,
                        "reward_karma": reward_karma,
                        "reason": reason,
                        "rarity": rarity,
                        "box_type": box_type,
                        "status": "delivered"
                    })
                except Exception as e:
                    print(f"Error calculating reward details: {e}")
                    response.update({
                        "surprise_unlocked": True,
                        "reward_karma": self.config['karma_min'],
                        "reason": "Daily activity bonus",
                        "rarity": "common",
                        "box_type": "mystery",
                        "status": "delivered"
                    })
            
            return response
        except Exception as e:
            print(f"Unexpected error in check_surprise_box: {e}")
            return {
                "user_id": user_id,
                "surprise_unlocked": False,
                "reward_karma": 0,
                "reason": "Error processing request",
                "rarity": "none",
                "box_type": "none",
                "status": "missed"
            }

def load_model():
    """Load the trained classifier model."""
    try:
        with open('classifier.pkl', 'rb') as f:
            model = joblib.load(f)
            print('model loaded')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_reward_engine():
    """Get or create the reward engine singleton."""
    return RewardEngine()