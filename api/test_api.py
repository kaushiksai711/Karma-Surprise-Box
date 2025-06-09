import unittest
import json
import os
import tempfile
import shutil
from fastapi.testclient import TestClient
from fastapi import status
from main import app, CONFIG_FILE
from pathlib import Path

class TestRewardAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create test client
        cls.client = TestClient(app)
        
        # Create a backup of the original config file
        cls.config_backup = Path('config.json.bak')
        if CONFIG_FILE.exists():
            shutil.copy2(CONFIG_FILE, cls.config_backup)
        
        # Create a test config file
        cls.test_config = {
            "reward_probability_threshold": 0.5,
            "karma_min": 10,
            "karma_max": 1000,
            "reward_rules": {
                "daily_activity": {
                    "conditions": ["login_streak > 3", "posts_created >= 1"],
                    "description": "Daily active user"
                }
            },
            "box_types": {
                "mystery": {
                    "name": "Mystery Box",
                    "base_karma": 50,
                    "rarity_weights": {
                        "common": 0.6,
                        "rare": 0.25,
                        "elite": 0.1,
                        "legendary": 0.05
                    }
                }
            }
        }
        
        # Save test config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cls.test_config, f, indent=2)
    
    @classmethod
    def tearDownClass(cls):
        # Restore original config file
        if cls.config_backup.exists():
            shutil.move(cls.config_backup, CONFIG_FILE)
        elif CONFIG_FILE.exists():
            os.remove(CONFIG_FILE)
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("message", response.json())
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["status"], "ok")
    
    def test_check_surprise_box_success(self):
        """Test checking for surprise box with valid data"""
        test_data = {
            "user_id": "test123",
            "date": "2024-06-09",
            "daily_metrics": {
                "login_streak": 5,
                "posts_created": 2,
                "comments_written": 3,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 35,
                "karma_earned_today": 50
            }
        }
        
        response = self.client.post("/check-surprise-box", json=test_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn("user_id", data)
        self.assertIn("surprise_unlocked", data)
        self.assertIn("status", data)
        
        if data["surprise_unlocked"]:
            self.assertIn("reward_karma", data)
            self.assertIn("reason", data)
    
    # Configuration Management Tests
    def test_get_config(self):
        """Test getting the current configuration"""
        response = self.client.get("/config")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        config = response.json()
        self.assertIn("reward_probability_threshold", config)
        self.assertIn("karma_min", config)
        self.assertIn("karma_max", config)
        self.assertIn("reward_rules", config)
        self.assertIn("box_types", config)
    
    def test_update_config(self):
        """Test updating the configuration"""
        update_data = {
            "reward_probability_threshold": 0.6,
            "karma_min": 20,
            "karma_max": 2000
        }
        
        response = self.client.patch("/config", json=update_data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify the update
        response = self.client.get("/config")
        config = response.json()
        self.assertEqual(config["reward_probability_threshold"], 0.6)
        self.assertEqual(config["karma_min"], 20)
        self.assertEqual(config["karma_max"], 2000)
    
    def test_list_rules(self):
        """Test listing all reward rules"""
        response = self.client.get("/rules")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        data = response.json()
        self.assertIn("rules", data)
        self.assertIn("daily_activity", data["rules"])
    
    def test_add_or_update_rule(self):
        """Test adding or updating a reward rule"""
        rule_data = {
            "conditions": ["quizzes_completed > 0"],
            "description": "Completed a quiz"
        }
        
        # Add a new rule
        response = self.client.post("/rules/quiz_completion", json=rule_data)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # Verify the rule was added
        response = self.client.get("/rules")
        self.assertIn("quiz_completion", response.json()["rules"])
        
        # Update the rule
        updated_rule = {"conditions": ["quizzes_completed > 1"]}
        response = self.client.post("/rules/quiz_completion", json=updated_rule)
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
    
    def test_delete_rule(self):
        """Test deleting a reward rule"""
        # First add a rule to delete
        rule_data = {"conditions": ["login_streak > 7"]}
        self.client.post("/rules/weekly_streak", json=rule_data)
        
        # Delete the rule
        response = self.client.delete("/rules/weekly_streak")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        # Verify the rule was deleted
        response = self.client.get("/rules")
        self.assertNotIn("weekly_streak", response.json()["rules"])
    
    def test_delete_nonexistent_rule(self):
        """Test deleting a non-existent rule"""
        response = self.client.delete("/rules/nonexistent_rule")
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
    
    def test_check_surprise_box_invalid_date(self):
        """Test checking for surprise box with invalid date format"""
        test_data = {
            "user_id": "test123",
            "date": "invalid-date",
            "daily_metrics": {
                "login_streak": 1,
                "posts_created": 0,
                "comments_written": 0,
                "upvotes_received": 0,
                "quizzes_completed": 0,
                "buddies_messaged": 0,
                "karma_spent": 0,
                "karma_earned_today": 0
            }
        }
        
        response = self.client.post("/check-surprise-box", json=test_data)
        self.assertEqual(response.status_code, 400)
    
    def test_check_surprise_box_missing_fields(self):
        """Test checking for surprise box with missing required fields"""
        test_data = {
            "user_id": "test123",
            # Missing date
            "daily_metrics": {
                # Missing some metrics
                "login_streak": 1,
                "posts_created": 0
            }
        }
        
        response = self.client.post("/check-surprise-box", json=test_data)
        self.assertEqual(response.status_code, 422)  # Validation error

if __name__ == "__main__":
    unittest.main()
