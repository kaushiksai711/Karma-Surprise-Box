import unittest
import json
from fastapi.testclient import TestClient
from main import app, fake_users_db, get_password_hash

class TestRewardAPI(unittest.TestCase):    
    @classmethod
    def setUpClass(cls):
        # Create test client
        cls.client = TestClient(app)
        
        # Add a test user if not exists
        if "testuser" not in fake_users_db:
            fake_users_db["testuser"] = {
                "username": "testuser",
                "hashed_password": get_password_hash("testpassword"),
                "disabled": False,
            }
    
    def get_auth_token(self):
        """Helper method to get an auth token for test user"""
        response = self.client.post(
            "/token",
            data={"username": "testuser", "password": "testpassword"},
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
        return response.json()["access_token"]
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_check_reward_unauthorized(self):
        """Test check-reward without authentication"""
        test_data = {
            "user_id": "test123",
            "date": "2023-06-06",
            "daily_metrics": {
                "login_streak": 5,
                "posts_created": 2,
                "comments_written": 3,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 5,
                "karma_earned_today": 15
            }
        }
        response = self.client.post("/check-reward/", json=test_data)
        self.assertEqual(response.status_code, 401)  # Unauthorized
    
    def test_check_reward_authorized(self):
        """Test check-reward with authentication"""
        # Get auth token
        token = self.get_auth_token()
        
        # Test data
        test_data = {
            "user_id": "test123",
            "date": "2023-06-06",
            "daily_metrics": {
                "login_streak": 5,
                "posts_created": 2,
                "comments_written": 3,
                "upvotes_received": 10,
                "quizzes_completed": 1,
                "buddies_messaged": 2,
                "karma_spent": 5,
                "karma_earned_today": 15
            }
        }
        
        # Make authenticated request
        response = self.client.post(
            "/check-reward/",
            json=test_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("qualifies", data)
        self.assertIn("message", data)
        
        # If qualifies, check for additional fields
        if data["qualifies"]:
            self.assertIn("reward_karma", data)
            self.assertIn("box_rarity", data)
            self.assertIn("condition_matched", data)

if __name__ == "__main__":
    unittest.main()
